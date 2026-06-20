"""Build population-level (cell_line, drug) CENTROIDS from a subset of Tahoe-x1 embedding shards.

DESIGN (matches the brief):
  * The measurement is destructive, so cells differ between control and treated; we work at the
    POPULATION level. For each (cell_line, drug) we store the centroid of the 2560-d mosaicfm-3b
    embedding plus the cell count. The vehicle/DMSO populations per cell line are the control
    baselines.
  * Min-cell filter (refinement #4): drop (cell_line, drug) populations with fewer than --min_cells
    cells so centroids are stable. We report the threshold and #surviving pairs.

Streaming: each shard is ~15 GB. We hf_hub_download it (resumable), then pyarrow.iter_batches over
ALL columns once, accumulating per-group float64 sums + counts (no full-data copy in RAM). The local
parquet is deleted after processing unless --keep_parquet.

Output npz (examples/tahoe_probe/data/centroids.npz):
  treated_centroid [P,2560] f32, treated_cell_line [P] int, treated_drug [P] int, treated_n [P] int
  control_centroid [C,2560] f32, control_cell_line [C] int, control_n [C] int
  cell_line_names [n_cl] str, drug_names [n_drug] str, control_labels [..] str, meta (json str)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import pyarrow.parquet as pq

REPO = "tahoebio/Tahoe-x1-embeddings"
EMB_COL = "mosaicfm-3b-prod-cont-MFMv2"
FILE_TMPL = "data/mosaicfm_3b_tahoe_100m_embeddings_shard-%03d.parquet"
DEFAULT_CONTROLS = ["DMSO_TF"]  # verified label is filled in from the footer scan; override via --controls


def run(
    shards: str = "0",
    controls: str = ",".join(DEFAULT_CONTROLS),
    min_cells: int = 50,
    out: str = "examples/tahoe_probe/data/centroids.npz",
    batch_size: int = 20000,
    keep_parquet: bool = True,
    cache_dir: str = "/tmp/tahoe_shards",
    parquet_paths: str = "",
):
    """If parquet_paths (comma-sep local files) is given, read those directly instead of downloading
    from HF (used on the cluster where a shard already lives on lustre)."""
    local_paths = None
    if parquet_paths:
        if isinstance(parquet_paths, (list, tuple)):
            local_paths = [str(p) for p in parquet_paths]
        else:
            local_paths = [p for p in str(parquet_paths).split(",") if p]
    if isinstance(shards, (list, tuple)):
        shard_list = [int(s) for s in shards]
    elif isinstance(shards, int):
        shard_list = [shards]
    else:
        shard_list = [int(s) for s in str(shards).strip("() ").replace(" ", "").split(",") if s != ""]
    if isinstance(controls, (list, tuple)):
        control_set = set(str(c) for c in controls)
    else:
        control_set = set(c for c in str(controls).split(",") if c != "")
    print(f"shards={shard_list} controls={sorted(control_set)} min_cells={min_cells}", flush=True)

    # accumulators keyed by (cell_line, drug) -> [sum(2560) float64, count]
    sums: dict = {}
    counts: dict = defaultdict(int)

    def acc(key, emb_block, n):
        if key not in sums:
            sums[key] = emb_block.sum(axis=0).astype(np.float64)
        else:
            sums[key] += emb_block.sum(axis=0)
        counts[key] += n

    iter_items = list(enumerate(local_paths)) if local_paths else [(s, None) for s in shard_list]
    if local_paths:
        shard_list = list(range(len(local_paths)))
    for shard, path in iter_items:
        if path is None:
            print(f"[shard {shard}] downloading...", flush=True)
            from huggingface_hub import hf_hub_download  # lazy: cluster reads local files, no HF needed
            local = hf_hub_download(REPO, FILE_TMPL % shard, repo_type="dataset", cache_dir=cache_dir)
        else:
            print(f"[shard {shard}] reading local {path}", flush=True)
            local = path
        pf = pq.ParquetFile(local)
        nrows = pf.metadata.num_rows
        seen = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=["drug", "cell_line", EMB_COL]):
            drugs = batch.column("drug").to_pylist()
            cls = batch.column("cell_line").to_pylist()
            # emb is FixedSizeList<float>[2560] -> fast zero-copy reshape (to_pylist would be very slow)
            col = batch.column(EMB_COL)
            emb = col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1)  # [b,2560] f32
            # group within batch by (cell_line, drug)
            key_arr = list(zip(cls, drugs))
            order = defaultdict(list)
            for i, k in enumerate(key_arr):
                order[k].append(i)
            for k, idxs in order.items():
                acc(k, emb[idxs], len(idxs))
            seen += len(drugs)
            if seen % 200000 < batch_size:
                print(f"  [shard {shard}] {seen:,}/{nrows:,} rows", flush=True)
        if not keep_parquet and path is None:  # only delete files WE downloaded, never pre-existing
            Path(local).unlink(missing_ok=True)
            print(f"  [shard {shard}] processed; deleted local parquet", flush=True)
        # incremental save after EACH shard so a usable dataset exists early and grows
        done = shard_list[: shard_list.index(shard) + 1]
        _assemble_and_save(out, sums, counts, control_set, done, min_cells)

    print("\nALL SHARDS DONE.", flush=True)
    return _assemble_and_save(out, sums, counts, control_set, shard_list, min_cells, final=True)


def _assemble_and_save(out, sums, counts, control_set, shard_list, min_cells, final=False):
    # -- assemble, apply min-cell filter, split control vs treated
    cell_lines = sorted({k[0] for k in counts})
    drugs_all = sorted({k[1] for k in counts})
    cl_ix = {c: i for i, c in enumerate(cell_lines)}
    dr_ix = {d: i for i, d in enumerate(drugs_all)}

    treated_c, treated_cl, treated_dr, treated_n = [], [], [], []
    control_c, control_cl, control_n = [], [], []
    n_dropped = 0
    for k, cnt in counts.items():
        cl, dr = k
        centroid = (sums[k] / cnt).astype(np.float32)
        if cnt < min_cells:
            n_dropped += 1
            continue
        if dr in control_set:
            control_c.append(centroid); control_cl.append(cl_ix[cl]); control_n.append(cnt)
        else:
            treated_c.append(centroid); treated_cl.append(cl_ix[cl]); treated_dr.append(dr_ix[dr]); treated_n.append(cnt)

    cn = np.asarray(control_n) if control_n else np.asarray([0])
    tn = np.asarray(treated_n) if treated_n else np.asarray([0])
    meta = {
        "shards": shard_list, "controls": sorted(control_set), "min_cells": min_cells,
        "final": final,
        "n_cell_lines": len(cell_lines), "n_drugs_total": len(drugs_all),
        "n_treated_pairs": len(treated_c), "n_control_pops": len(control_c),
        "n_dropped_below_min_cells": n_dropped,
        "control_cells_min": int(cn.min()), "control_cells_median": int(np.median(cn)),
        "treated_cells_min": int(tn.min()), "treated_cells_median": int(np.median(tn)),
        "emb_dim": 2560, "emb_col": EMB_COL,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        treated_centroid=np.asarray(treated_c, dtype=np.float32),
        treated_cell_line=np.asarray(treated_cl, dtype=np.int64),
        treated_drug=np.asarray(treated_dr, dtype=np.int64),
        treated_n=np.asarray(treated_n, dtype=np.int64),
        control_centroid=np.asarray(control_c, dtype=np.float32),
        control_cell_line=np.asarray(control_cl, dtype=np.int64),
        control_n=np.asarray(control_n, dtype=np.int64),
        cell_line_names=np.asarray(cell_lines),
        drug_names=np.asarray(drugs_all),
        control_labels=np.asarray(sorted(control_set)),
        meta=json.dumps(meta),
    )
    tag = "FINAL COVERAGE" if final else f"incremental coverage (shards done: {shard_list})"
    print(f"\n=== {tag} (MEASURED) ===", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"saved -> {out}", flush=True)
    return meta


if __name__ == "__main__":
    fire.Fire(run)
