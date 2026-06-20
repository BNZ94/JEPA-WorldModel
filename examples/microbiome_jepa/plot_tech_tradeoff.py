"""Plot the tech-invariance tradeoff curve from tech_sweep's sweep_results.json.

X = invariance_coeff; left Y = TECH balanced-acc (down = more invariant = good),
right Y = BIOME balanced-acc (kept = good). Horizontal guides: tech chance (0.5),
the confounding floor (biome->tech bal-acc), and the before-baseline tech-acc.
Every plotted point is MEASURED (mean±s.e. over seeds) from the sweep JSON.

  python -m examples.microbiome_jepa.plot_tech_tradeoff \
      --sweep_json <out_root>/sweep_results.json \
      --confounding_json <...>/tech_confounding/confounding.json \
      --out examples/microbiome_jepa/results/tech_tradeoff.png
"""
import json

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(sweep_json, confounding_json=None, out="examples/microbiome_jepa/results/tech_tradeoff.png"):
    with open(sweep_json) as fh:
        sweep = json.load(fh)
    summary = sweep["summary"]
    floor = None
    if confounding_json:
        with open(confounding_json) as fh:
            c = json.load(fh)
        floor = c["confounding"]["predict_tech_from_biome"]["balanced_acc_mean"]

    losses = sorted({s["loss"] for s in summary})
    fig, axes = plt.subplots(1, len(losses), figsize=(6 * len(losses), 4.5), squeeze=False)
    for ax, loss in zip(axes[0], losses):
        rows = sorted([s for s in summary if s["loss"] == loss], key=lambda r: r["coeff"])
        xs = [r["coeff"] for r in rows]
        tech = [r["tech_balacc"]["mean"] for r in rows]
        tech_se = [r["tech_balacc"]["se"] or 0 for r in rows]
        biome = [r["biome_balacc"]["mean"] for r in rows]
        biome_se = [r["biome_balacc"]["se"] or 0 for r in rows]
        ax.errorbar(xs, tech, yerr=tech_se, marker="o", color="C3", label="TECH bal-acc (↓ good)")
        ax.set_xlabel("invariance_coeff (DANN strength)")
        ax.set_ylabel("TECH balanced acc", color="C3")
        ax.axhline(0.5, ls=":", c="grey", lw=1, label="tech chance 0.5")
        if floor:
            ax.axhline(floor, ls="--", c="C1", lw=1, label=f"confounding floor {floor:.2f}")
        ax2 = ax.twinx()
        ax2.errorbar(xs, biome, yerr=biome_se, marker="s", color="C0", label="BIOME bal-acc (keep)")
        ax2.set_ylabel("BIOME balanced acc", color="C0")
        ax.set_title(f"{loss}: tech-invariance tradeoff")
        ax.set_ylim(0.45, 1.0)
        ax2.set_ylim(0.45, 1.0)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    fire.Fire(run)
