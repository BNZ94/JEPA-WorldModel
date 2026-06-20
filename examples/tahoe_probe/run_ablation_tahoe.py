"""IDM-ablation driver for the Tahoe perturbation probe (the gLV run_ablation.py analog).

For each seed and each arm (idm_coeff = 1.0 'on' vs 0.0 'off') and each split (pairs, celllines):
  train the predictor, eval shift-space metrics vs the three baselines, and the action-decodability
  of true vs predicted shifts. Aggregate mean +/- standard error per arm, print tables, save JSON +
  a figure.

FRAMING (refinement #3): the encoder is frozen, so this is NOT a replication of the gLV M4
collapse-rescue. It tests whether an IDM auxiliary that pushes the predictor toward action-aware
transitions improves predictor quality / decodability. We report the effect honestly either way.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np

from examples.tahoe_probe.data import make_split
from examples.tahoe_probe.train_probe import BASELINES, eval_arm, train_arm

ARMS = [("idm_on", 1.0), ("idm_off", 0.0)]


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    n = len(a)
    return (float(a.mean()), float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)


def _seedlist(seeds):
    if isinstance(seeds, (list, tuple)):
        return [int(s) for s in seeds]
    if isinstance(seeds, int):
        return [seeds]
    return [int(s) for s in str(seeds).strip("() ").split(",") if str(s).strip()]


def run(npz="examples/tahoe_probe/data/centroids.npz", seeds="0,1,2,3,4", splits="pairs,celllines",
        epochs=300, action_dim=64, idm_hidden=256, lr=3e-4, weight_decay=1e-3, test_frac=0.2,
        out="examples/tahoe_probe/results/ablation", device="cpu"):
    seed_list = _seedlist(seeds)
    split_kinds = [s for s in str(splits).strip("() ").replace(" ", "").split(",") if s] \
        if not isinstance(splits, (list, tuple)) else [str(s) for s in splits]
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    coverage = None

    for kind in split_kinds:
        for seed in seed_list:
            split, info = make_split(npz, kind=kind, test_frac=test_frac, seed=seed)
            coverage = info
            for arm, coeff in ARMS:
                model = train_arm(split, idm_coeff=coeff, seed=seed, action_dim=action_dim,
                                  idm_hidden=idm_hidden, epochs=epochs, lr=lr,
                                  weight_decay=weight_decay, device=device)
                res = eval_arm(model, split, seed=seed, device=device)
                rec = {"split": kind, "seed": seed, "arm": arm, "idm_coeff": coeff,
                       "n_train": info["n_train"], "n_test": info["n_test"], **_flatten(res)}
                records.append(rec)
                print(f"[{kind} seed{seed} {arm}] R2_shift={rec['model.R2_shift']:.3f} "
                      f"per_drug_R2_shift={rec['per_drug_mean_shift.R2_shift']:.3f} "
                      f"decode_pred_top1={rec['decode_pred_shift.top1']:.3f}", flush=True)

    summary = _summarize(records, split_kinds)
    _print_tables(summary, split_kinds, seed_list, coverage)
    res_path = out_dir / "ablation_results.json"
    json.dump({"seeds": seed_list, "splits": split_kinds, "epochs": epochs,
               "coverage": coverage, "records": records, "summary": summary},
              open(res_path, "w"), indent=2)
    print(f"\nsaved raw -> {res_path}", flush=True)
    try:
        _make_figure(summary, split_kinds, seed_list, epochs, out_dir / "ablation_figure.png")
        print(f"saved figure -> {out_dir / 'ablation_figure.png'}", flush=True)
    except Exception as e:  # matplotlib may be absent on a compute node; JSON has everything
        print(f"[figure skipped: {e}] — render later from {res_path}", flush=True)
    return {"summary": summary, "results_json": str(res_path)}


def _flatten(res):
    flat = {}
    for grp, d in res.items():
        for k, v in d.items():
            flat[f"{grp}.{k}"] = v
    return flat


# metrics we aggregate per (split, arm)
AGG_KEYS = (["model.R2_shift", "model.cos_shift", "model.R2_abs", "model.cos_abs"]
            + [f"{b}.R2_shift" for b in BASELINES] + [f"{b}.cos_shift" for b in BASELINES]
            + ["decode_true_shift.top1", "decode_true_shift.top5",
               "decode_pred_shift.top1", "decode_pred_shift.top5",
               "decode_pred_shift.chance_top1"])


def _summarize(records, split_kinds):
    summary = {}
    for kind in split_kinds:
        summary[kind] = {}
        for arm, _ in ARMS:
            rs = [r for r in records if r["split"] == kind and r["arm"] == arm]
            summary[kind][arm] = {k: _agg([r.get(k) for r in rs]) for k in AGG_KEYS}
    return summary


def _print_tables(summary, split_kinds, seed_list, coverage):
    print("\n================ TAHOE IDM ABLATION (MEASURED) ================")
    print(f"seeds={seed_list}  coverage={json.dumps(coverage)}")
    for kind in split_kinds:
        print(f"\n--- split: {kind} ---")
        hdr = "metric".ljust(28) + "".join(a.ljust(20) for a, _ in ARMS)
        print(hdr)
        for k in AGG_KEYS:
            row = k.ljust(28)
            for arm, _ in ARMS:
                m, se = summary[kind][arm][k]
                row += f"{m:+.3f}±{se:.3f}".ljust(20)
            print(row)


def _make_figure(summary, split_kinds, seed_list, epochs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(split_kinds), figsize=(6.5 * len(split_kinds), 5), squeeze=False)
    bar_keys = ["model.R2_shift", "per_drug_mean_shift.R2_shift",
                "global_mean_shift.R2_shift", "no_op.R2_shift"]
    labels = ["model\n(idm arm)", "per-drug\nmean-shift", "global\nmean-shift", "no-op"]
    for ax, kind in zip(axes[0], split_kinds):
        x = np.arange(len(bar_keys)); w = 0.36
        for i, (arm, _) in enumerate(ARMS):
            means = [summary[kind][arm][k][0] for k in bar_keys]
            ses = [summary[kind][arm][k][1] for k in bar_keys]
            ax.bar(x + (i - 0.5) * w, means, w, yerr=ses, capsize=4, label=arm,
                   color=("#2a7" if arm == "idm_on" else "#c44"))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("R²(shift) on held-out  (higher=better)")
        ax.set_title(f"split: {kind}")
        ax.legend()
    fig.suptitle(f"Tahoe drug-perturbation probe — shift-space R² vs baselines "
                 f"({len(seed_list)} seeds, {epochs} ep)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    fire.Fire(run)
