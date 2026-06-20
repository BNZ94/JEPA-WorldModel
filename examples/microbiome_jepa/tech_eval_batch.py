"""Batched + parallel fresh-probe eval for the tech-invariance sweep.

Loads the corpus / ProkBERT h5 / tokens / z-score ONCE (workers inherit them via fork —
no 60x re-stream), computes the constant baselines (raw mean-pool, random encoder) once,
then for each checkpoint encodes the frozen latent and runs FRESH probes (linear + MLP)
for technology (down=good) and biome (keep) IN PARALLEL across CPU cores. Per-tag JSONs
already on disk are reused (skip). Aggregates to sweep_results.json + tradeoff.

INTEGRITY: identical metric to tech_invariance.run (same probe, same balanced-acc, same
StratifiedKFold seed) — just batched for speed; never uses the training adversary.

Run (cluster CPU):
  python -m examples.microbiome_jepa.tech_eval_batch --n_workers 14 \
      --losses vicreg,bcs --coeffs 0,0.3,1.0,3.0,10.0 --seeds 0,1,2,3,4,5
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import numpy as np
import torch

from eb_jepa.datasets.microbiome.otu_data import (
    build_otu_key_resolver, load_otu_rename_map, load_prokbert_embeddings)
from eb_jepa.datasets.microbiome.transforms import PerDimZScore
from eb_jepa.training_utils import load_config
from examples.microbiome_jepa.tech_invariance import (
    F, STRATEGIES, build_tokens, encode, load_runid_labels, probe, raw_meanpool,
    stream_labeled_communities)
from examples.microbiome_jepa.tech_sweep import _csv, _tag, summarize, _print_tradeoff

# globals filled in the parent before the pool forks (workers inherit via COW)
_D = {}


def _tags(losses, coeffs, seeds):
    out = []
    for loss in losses:
        for c in coeffs:
            for s in seeds:
                out.append((loss, c, s, _tag(loss, c, s)))
    return out


def _eval_tag(args):
    """Worker: encode one frozen checkpoint and fresh-probe tech + biome (linear + MLP)."""
    from eb_jepa.architectures import SetTransformerEncoder
    torch.set_num_threads(1)
    loss, coeff, seed, tag = args
    ckpt = os.path.join(_D["ckpt_root"], tag, "latest.pth.tar")
    out_dir = os.path.join(_D["out_root"], tag)
    jpath = os.path.join(out_dir, "tech_invariance.json")
    if os.path.exists(jpath):  # reuse a previously-computed result
        try:
            with open(jpath) as fh:
                res = json.load(fh)
            res["_meta"] = {"loss": loss, "coeff": float(coeff), "seed": int(seed), "tag": tag}
            return res
        except Exception:
            pass
    if not os.path.exists(ckpt):
        return None
    enc = SetTransformerEncoder(token_dim=F, d_model=_D["d_model"], n_heads=_D["n_heads"],
                                n_layers=_D["n_layers"], dim_feedforward=_D["dim_ff"],
                                dropout=0.0, pool=_D["pool"])
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("encoder_state_dict") or ck.get("model_state_dict") or ck
    sd = {k.replace("encoder.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    enc.load_state_dict(sd, strict=False)
    Z = encode(enc, _D["tokens"], _D["masks"], _D["zscore"], torch.device("cpu"))
    hb = _D["has_biome"]
    res = {
        "tech_probe": {"jepa_pretrained": probe(Z, _D["strat"], kind="linear")},
        "tech_probe_mlp": {"jepa_pretrained": probe(Z, _D["strat"], kind="mlp")},
        "biome_probe": {"jepa_pretrained": probe(Z[hb], _D["biome_y"], kind="linear")},
        "biome_probe_mlp": {"jepa_pretrained": probe(Z[hb], _D["biome_y"], kind="mlp")},
        "_meta": {"loss": loss, "coeff": float(coeff), "seed": int(seed), "tag": tag},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(jpath, "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def run(losses="vicreg,bcs", coeffs="0,0.3,1.0,3.0,10.0", seeds="0,1,2,3,4,5",
        d_model=128, n_max=256, per_class_cap=2500, n_workers=14,
        fname="examples/microbiome_jepa/cfgs/layerA_real.yaml", data_dir=None,
        ckpt_root=None, out_root=None):
    work = os.environ.get("WORK", ".")
    data_dir = data_dir or os.path.join(os.environ.get("EBJEPA_DSETS", "."), "susagi", "data")
    ckpt_root = ckpt_root or os.path.join(work, "checkpoints/microbiome_jepa/tech_sweep")
    out_root = out_root or os.path.join(work, "checkpoints/microbiome_jepa/tech_sweep_eval")
    losses, coeffs, seeds = _csv(losses), _csv(coeffs), _csv(seeds)  # robust to fire tuple-parsing
    cfg = load_config(fname, {"model.d_model": d_model}, quiet=True)
    mb = os.path.join(data_dir, "microbeatlas")

    # ---- load EVERYTHING once (workers inherit via fork) ----
    runid_labels = load_runid_labels(os.path.join(mb, "sample_terms_mapping_combined_dany_og_biome_tech.txt"))
    samples = stream_labeled_communities(os.path.join(mb, "samples-otus.97.mapped"),
                                         runid_labels, n_max, per_class_cap)
    emb, otu_id_to_row = load_prokbert_embeddings(os.path.join(data_dir, "model", "prokbert_embeddings.h5"))
    rename_map = load_otu_rename_map(os.path.join(mb, "otus.rename.map1"))
    all_ids = [oid for s in samples for (oid, _) in s["otus"]]
    resolver = build_otu_key_resolver(all_ids, rename_map, set(otu_id_to_row))
    tokens, masks = build_tokens(samples, emb, otu_id_to_row, resolver, n_max)
    keep = masks.any(1).numpy()
    tokens, masks = tokens[keep], masks[keep]
    samples = [s for s, k in zip(samples, keep) if k]
    strat = np.array([s["strat"] for s in samples])
    biome = [s["biome"] for s in samples]
    has_biome = np.array([b is not None for b in biome])
    biome_y = [b for b in biome if b is not None]
    zscore = PerDimZScore().fit(tokens.reshape(-1, F), mask=masks.reshape(-1))

    _D.update(dict(ckpt_root=ckpt_root, out_root=out_root, d_model=cfg.model.d_model,
                   n_heads=cfg.model.n_heads, n_layers=cfg.model.n_layers,
                   dim_ff=cfg.model.dim_feedforward, pool=cfg.model.get("pool", "mean"),
                   tokens=tokens, masks=masks, zscore=zscore, strat=strat,
                   has_biome=has_biome, biome_y=biome_y))

    # constant baselines (computed once, for the report)
    baselines = {
        "raw_meanpool": {"tech": probe(raw_meanpool(tokens, masks, zscore), strat, kind="linear"),
                         "biome": probe(raw_meanpool(tokens, masks, zscore)[has_biome], biome_y, kind="linear")},
    }

    tags = _tags(losses, coeffs, seeds)
    print(f"[batch-eval] {len(tags)} tags | n_workers={n_workers} | n={len(strat)} "
          f"biome_n={int(has_biome.sum())}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_eval_tag, t): t for t in tags}
        for fut in as_completed(futs):
            r = fut.result()
            if r is None:
                print(f"[eval] SKIP {futs[fut][3]} (no checkpoint)", flush=True)
                continue
            tp = r["tech_probe"]["jepa_pretrained"]["balanced_acc_mean"]
            bp = r["biome_probe"]["jepa_pretrained"]["balanced_acc_mean"]
            print(f"[eval] {r['_meta']['tag']}: TECH bal {tp:.3f} | BIOME bal {bp:.3f}", flush=True)
            results.append(r)

    summary = summarize(results)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "sweep_results.json"), "w") as fh:
        json.dump({"config": {"d_model": d_model, "per_class_cap": per_class_cap, "n": int(len(strat))},
                   "baselines": baselines, "runs": results, "summary": summary}, fh, indent=2)
    print(f"\n[batch-eval] saved -> {out_root}/sweep_results.json ({len(results)} runs)", flush=True)
    _print_tradeoff(summary)
    return summary


if __name__ == "__main__":
    fire.Fire(run)
