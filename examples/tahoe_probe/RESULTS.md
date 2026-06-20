# Tahoe perturbation probe — RESULTS (all numbers MEASURED)

Generality check: does the gLV action-conditioned latent predictor + IDM recipe transfer to real
single-cell drug-perturbation data (Tahoe-x1 frozen 2560-d mosaicfm-3b embeddings)? One-step,
population/centroid level. Branch `tahoe-perturbation-probe` (local only; not folded into `bnz`).

**Provenance.** Centroids built from `emb0.parquet` (Tahoe-x1 shard-0, 1,494,131 cells) read on the
Dalia cluster; ablation trained on a GB200 GPU (aarch64 venv), 5 seeds. Raw numbers:
`examples/tahoe_probe/results/ablation/ablation_results.json`. Figure:
`examples/tahoe_probe/results/ablation/ablation_figure.png`. Every table below is MEASURED from those
runs (no expected/assumed values).

## Data coverage (MEASURED)
50 cell lines · 95 drugs (non-control) · **4,198** treated (cell_line, drug) pairs · 48 DMSO_TF control
populations · 500 pairs dropped (<50 cells) · control cells/pop min 50 / median 638 · treated cells/pop
min 50 / median 312. Controls = `DMSO_TF` (exact; `Trametinib (DMSO_TF solvate)` excluded). Cell lines
are shared across the dataset, so the single shard gives controls + many drugs over the same 50 lines.

## Splits, metrics, baselines
- **Splits.** `pairs` (primary): random 80/20 over (cell_line, drug) pairs — both seen, the combination
  held out (n_test ≈ 840). `celllines` (secondary): whole cell lines held out (n_test ≈ 755).
- **Metric.** Shift-space R² and cosine on Δ = z_treated − z_control (standardized latent; standardizer
  fit on train only). Absolute-z metrics are reported but uninformative (see below).
- **Baselines.** no-op (Δ̂=0); global mean-shift (mean train Δ); **per-drug mean-shift** (PRIMARY — mean
  train Δ for that drug over the *other* cell lines; beating it requires cell-line-specific modulation).
- Training: GRU predictor (action=input, z_control=hidden) + drug-embedding action encoder; objective
  `MSE + idm_coeff·CE(idm(z_control, ẑ_treated), drug)`; ablation idm_coeff ∈ {1.0, 0.0}; validation
  early-stopping (15% of train; no test tuning); seeds drive split + init (paired arms).

### The absolute-z trap (why we report the shift)
Absolute-z metrics are high for *everything* including no-op: R²_abs ≈ 0.93–0.96, cos_abs ≈ 0.97 across
all methods and arms. The drug effect is ~29% of state magnitude (standardized), so z_treated ≈ z_control
and absolute metrics are dominated by cell-line identity. All conclusions below use the **shift**.

## Result 1 — Held-out PAIRS (primary): the predictor (no IDM) beats every baseline
MEASURED, mean ± s.e. over 5 seeds.

| method | R²(shift) | cos(shift) | drug decodable from predicted shift (top-1) |
|---|---|---|---|
| **model, idm_off** | **+0.201 ± 0.004** | **+0.568 ± 0.001** | 0.455 ± 0.008 |
| model, idm_on | +0.017 ± 0.002 | +0.384 ± 0.002 | 0.007 ± 0.003 |
| per-drug mean-shift (primary baseline) | +0.149 ± 0.004 | +0.479 ± 0.001 | — |
| global mean-shift | −0.001 ± 0.000 | +0.346 ± 0.002 | — |
| no-op | −0.125 ± 0.002 | n/a (Δ̂=0) | — |

**The action-conditioned predictor without IDM beats the per-drug mean-shift baseline**
(0.201 vs 0.149, ~13 s.e. apart) on both R² and cosine → it captures **cell-line-specific modulation**
of the drug effect, not just the per-drug average. It also far exceeds global mean-shift and no-op.

## Result 2 — IDM ablation: on real data the IDM auxiliary HURTS (the gLV-M4 cross-check)
Adding the IDM auxiliary collapses the predictor: pairs R²(shift) **0.201 → 0.017**, and drug
decodability of the *predicted* shift **0.455 → 0.007** (top-1). This is the **opposite** of the gLV
M4 result, where IDM *helped* by rescuing action information under representational collapse.

Interpretation (as pre-registered in the brief — this is NOT an M4 replication): the gLV M4 benefit was
**collapse-specific**. Here the encoder is frozen (real mosaicfm-3b embeddings) and there is **no
collapse to fight**, so the IDM term is an unnecessary auxiliary that competes with the supervised
prediction objective and degrades it. A clean cross-check that *sharpens* the gLV claim — the IDM term
earns its keep only in the collapse regime it was introduced for.

**`idm_coeff` sensitivity (MEASURED, held-out pairs, 3 seeds)** — degradation is monotonic in the IDM
weight, so the on/off result is not an artifact of one large coefficient (`results/coeff_sweep.log`):

| idm_coeff | 0.0 | 0.01 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|
| R²(shift) | **+0.199 ± 0.002** | +0.155 ± 0.003 | +0.103 ± 0.003 | +0.071 ± 0.003 | +0.015 ± 0.003 |
| drug decodable from pred. shift (top-1) | 0.450 | 0.141 | 0.002 | 0.001 | 0.010 |

(per-drug baseline = +0.153 for this 3-seed subset.) Any positive IDM weight lowers held-out R²; only
`idm_coeff=0` clears the per-drug baseline.

## Result 3 — Held-out CELL LINES (secondary): does not transfer to unseen lines
MEASURED, mean ± s.e. over 5 seeds.

| method | R²(shift) | cos(shift) |
|---|---|---|
| model, idm_on | −0.149 ± 0.005 | +0.089 ± 0.004 |
| model, idm_off | −0.208 ± 0.020 | +0.102 ± 0.003 |
| per-drug mean-shift | **+0.133 ± 0.011** | +0.483 ± 0.006 |
| global mean-shift | −0.044 ± 0.004 | +0.327 ± 0.005 |
| no-op | −0.156 ± 0.005 | n/a |

For **entirely unseen cell lines**, both model arms are worse than no-op, while the per-drug mean-shift
stays positive (+0.133). The predictor cannot place a novel baseline state (novel z_control) and its
shift extrapolates poorly; the per-drug average still works because the drug is seen. **Honest
limitation: the predictor composes a seen cell line with a seen drug (Result 1) but does not generalize
to unseen cell lines.** (A learned per-drug action encoder also cannot generalize to unseen *drugs* —
SMILES would be needed; not attempted here.)

## Result 4 — Action information is in the embedding (data property)
A linear probe recovers the drug from the **true** latent shift at top-1 **0.753 ± 0.006** (pairs) /
**0.841 ± 0.014** (cell lines), top-5 0.95–0.98, vs chance ≈ 0.01–0.02. The latent transition strongly
encodes the intervention — the premise the IDM/action-conditioning exploits clearly holds on real data.

## What transfers, what does not (summary)
- ✅ Drug identity is strongly encoded in the real latent shift (Result 4).
- ✅ The action-conditioned predictor (no IDM) captures real drug responses **and** beats the per-drug
  baseline on held-out (cell_line, drug) pairs — it learns cell-line-specific modulation (Result 1).
- ✅/❗ IDM effect is the **opposite** of gLV-M4: it hurts here, consistent with M4 being collapse-specific
  (Result 2). Not a replication; a sharpening cross-check.
- ❌ No generalization to entirely unseen cell lines (Result 3).

## Reproduce
```bash
# centroids (cluster): build_centroids.py --parquet_paths emb0.parquet --controls DMSO_TF --min_cells 50
python -m examples.tahoe_probe.run_ablation_tahoe --npz <centroids.npz> \
    --seeds 0,1,2,3,4 --splits pairs,celllines --epochs 300 --lr 3e-4 --weight_decay 1e-3 \
    --action_dim 64 --device cuda
```
