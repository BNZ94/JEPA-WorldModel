"""Qualitative tech-invariance figure: 2-D projection (UMAP, PCA fallback) of community
representations, colored by TECHNOLOGY (amplicon/wgs) and by BIOME.

Compares, as rows:
  * raw mean-pooled input tokens   (the "base" space / input distribution),
  * the baseline JEPA latent       (--baseline ckpt, e.g. invariance_coeff=0),
  * the tech-invariant JEPA latent (--dann ckpt, e.g. invariance_coeff=3/10).
Columns: [colored by technology] | [colored by biome].

Story to look for: technology should go from CLUSTERED (raw + baseline JEPA) to MIXED
(DANN JEPA), while biome structure is PRESERVED across all three. Every point is a real
corpus community encoded by a frozen encoder; the 2-D coords are saved for reproducibility.

Reuses the eval's data loaders so the samples / join / z-score are identical to the probe.

Run (cluster, CPU):
  python -m examples.microbiome_jepa.plot_latent_umap \
      --baseline $WORK/checkpoints/microbiome_jepa/tech_sweep/techinv_vicreg_c0_s0/latest.pth.tar \
      --dann     $WORK/checkpoints/microbiome_jepa/tech_sweep/techinv_vicreg_c10p0_s0/latest.pth.tar \
      --data_dir $EBJEPA_DSETS/susagi/data --d_model 128 --per_class_cap 1500 \
      --out examples/microbiome_jepa/results/tech_umap.png
"""
import json
import os

import fire
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eb_jepa.architectures import SetTransformerEncoder
from eb_jepa.datasets.microbiome.otu_data import (
    build_otu_key_resolver, load_otu_rename_map, load_prokbert_embeddings)
from eb_jepa.datasets.microbiome.transforms import PerDimZScore
from eb_jepa.training_utils import load_config
from examples.microbiome_jepa.tech_invariance import (
    F, STRATEGIES, build_tokens, encode, load_runid_labels, raw_meanpool,
    stream_labeled_communities)


def _project(X, method, seed=0):
    """2-D embedding of X [N,D]. UMAP if available, else PCA."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed).fit_transform(Xs), "UMAP"
        except Exception as e:
            print(f"[umap unavailable: {type(e).__name__}: {e} -> PCA]", flush=True)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(Xs), "PCA"


def _load_encoder(ckpt, cfg, dev):
    enc = SetTransformerEncoder(token_dim=F, d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
                                n_layers=cfg.model.n_layers, dim_feedforward=cfg.model.dim_feedforward,
                                dropout=0.0, pool=cfg.model.get("pool", "mean")).to(dev)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = ck.get("encoder_state_dict") or ck.get("model_state_dict") or ck
    sd = {k.replace("encoder.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    enc.load_state_dict(sd, strict=False)
    return enc


def run(baseline=None, dann=None, fname="examples/microbiome_jepa/cfgs/layerA_real.yaml",
        data_dir=None, d_model=128, n_max=256, per_class_cap=1500, method="umap",
        max_points=4000, out="examples/microbiome_jepa/results/tech_umap.png"):
    dev = torch.device("cpu")
    data_dir = data_dir or os.path.join(os.environ.get("EBJEPA_DSETS", "."), "susagi", "data")
    mb = os.path.join(data_dir, "microbeatlas")
    cfg = load_config(fname, {"model.d_model": d_model}, quiet=True)

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
    biome = np.array([s["biome"] if s["biome"] else "?" for s in samples])
    zscore = PerDimZScore().fit(tokens.reshape(-1, F), mask=masks.reshape(-1))

    reps = {"raw input (base)": raw_meanpool(tokens, masks, zscore)}
    if baseline and os.path.exists(baseline):
        reps["JEPA baseline (coeff=0)"] = encode(_load_encoder(baseline, cfg, dev), tokens, masks, zscore, dev)
    if dann and os.path.exists(dann):
        reps["JEPA + DANN (tech-invariant)"] = encode(_load_encoder(dann, cfg, dev), tokens, masks, zscore, dev)

    # optional subsample for legible scatter (stratified by tech)
    n = len(strat)
    rng = np.random.default_rng(0)
    idx = np.arange(n) if n <= max_points else rng.choice(n, max_points, replace=False)
    strat_s, biome_s = strat[idx], biome[idx]

    biomes = sorted(set(biome_s) - {"?"})
    bcolors = {b: plt.cm.tab10(i % 10) for i, b in enumerate(biomes)}
    bcolors["?"] = (0.8, 0.8, 0.8, 0.4)
    tcolor = {"amplicon": "#1f77b4", "wgs": "#d62728"}

    nrows = len(reps)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.6 * nrows), squeeze=False)
    coords_out = {}
    for r, (name, X) in enumerate(reps.items()):
        XY, used = _project(X[idx], method)
        coords_out[name] = {"xy": XY.tolist(), "method": used}
        # by technology
        ax = axes[r][0]
        for s in STRATEGIES:
            m = strat_s == s
            ax.scatter(XY[m, 0], XY[m, 1], s=6, alpha=0.5, c=[tcolor[s]], label=s, linewidths=0)
        ax.set_title(f"{name} — by TECHNOLOGY [{used}]")
        ax.legend(markerscale=2, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        # by biome
        ax = axes[r][1]
        for b in biomes + ["?"]:
            m = biome_s == b
            if m.any():
                ax.scatter(XY[m, 0], XY[m, 1], s=6, alpha=0.5, c=[bcolors[b]], label=b, linewidths=0)
        ax.set_title(f"{name} — by BIOME [{used}]")
        ax.legend(markerscale=2, fontsize=7, ncol=2); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Community latent space: technology (want MIXED after fix) vs biome (want PRESERVED)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=130)
    with open(out.replace(".png", "_coords.json"), "w") as fh:
        json.dump({"n_plotted": int(len(idx)), "reps": list(reps),
                   "strat": strat_s.tolist(), "biome": biome_s.tolist(), "coords": coords_out}, fh)
    print(f"saved -> {out} (+ _coords.json)", flush=True)


if __name__ == "__main__":
    fire.Fire(run)
