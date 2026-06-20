# Tahoe perturbation probe — generality check for the gLV action-conditioned predictor + IDM

**What this is.** A real-data validation that the *action-conditioning* and *inverse-dynamics (IDM)*
components of our gLV world model transfer to real single-cell drug-perturbation data
(Tahoe-x1 embeddings). It is **not** a new perturbation-response model and does **not** compete with
CPA/GEARS/scGPT. The gLV remains the dynamical-planning headline. We test whether the same
action-conditioned latent predictor + IDM recipe that enabled multi-step planning on gLV also
(a) captures real drug responses on Tahoe and (b) shows an IDM effect on real data.

This work is isolated on branch `tahoe-perturbation-probe` and is **not** folded into `bnz`.

## Recipe reuse (faithful light reimplementation)
The gLV Layer-B pipeline (`examples/microbiome_jepa/train_worldmodel.py`) is entangled with the 5-D
temporal JEPA contract (`SetTransformerEncoder → [B,D,T,1,1]`, `jepa.unroll`, sequence regularizers).
Tahoe is **one-step**, uses **frozen** 2560-d mosaicfm-3b embeddings (no encoder to train), and the
prediction target is a real embedding centroid. We therefore reuse the *component designs*, not the
temporal scaffolding (`model.py` documents every deviation):
- `GRUPredictor` — GRU with **action = input, z_control = hidden**, exactly the wiring of
  `eb_jepa.architectures.RNNPredictor`. One step: `z_treated = f(z_control, action)`. Update-gate bias
  initialized high so the predictor starts at the "no-change" prior and learns the small drug delta.
- `IDMClassifier` — same MLP shape as `eb_jepa.architectures.InverseDynamicsModel`, on
  `concat(z_control, z_treated)`. The gLV action is a continuous vector (IDM regresses it); here the
  action is a categorical **drug**, so IDM is a **classifier** (cross-entropy). Forced by the modality.
- Objective mirrors `train_worldmodel.run`: `pred_loss (MSE) + idm_coeff · IDM`, ablation
  `idm_coeff ∈ {1.0, 0.0}`. VICReg anti-collapse is dropped (supervised target on a frozen latent →
  no collapse to fight).

### IDM framing (important; not an M4 replication)
On gLV the IDM benefit (M4) was **collapse-specific** — IDM rescued action information under
representational collapse. Here the encoder is frozen, so there is **no collapse**. To make
`idm_coeff` a real knob on a frozen-input model, our IDM reads the **predictor's output**
`idm(z_control, ẑ_treated)`, i.e. it is an auxiliary that regularizes the predictor toward an
action-aware transition. We therefore **do not claim M4 replicates**; we test a different mechanism
and report the effect honestly either way.

## Data (MEASURED coverage)
Source: `tahoebio/Tahoe-x1-embeddings`, the precomputed 2560-d `mosaicfm-3b-prod-cont-MFMv2` vectors
(we do **not** train an encoder). We used **1 shard** (`emb0.parquet`, shard-0, 1,494,131 cells) read
locally on the Dalia cluster (the same file is HF shard-0). Controls = `DMSO_TF` (exact match; the
drug `Trametinib (DMSO_TF solvate)` is **not** a control and is excluded). Population centroids per
`(cell_line, drug)`; populations with `< 50` cells dropped (refinement: stable centroids).

| quantity | value |
|---|---|
| cells | 1,494,131 |
| cell lines | 50 |
| drugs (non-control) | 95 |
| treated (cell_line, drug) pairs (≥50 cells) | 4,198 |
| control (DMSO_TF) populations | 48 |
| pairs dropped (<50 cells) | 500 |
| control cells per pop (min / median) | 50 / 638 |
| treated cells per pop (min / median) | 50 / 312 |

One shard already gives broad coverage with controls + many drugs across the **same** 50 cell lines
(cell lines are shared across all shards), so coverage refinement #5 is satisfied: for the cell lines
used, both DMSO controls and many drug treatments are present with adequate cell counts.

## Metrics & baselines (the discriminating choices)
The drug effect is small relative to cell-line identity (standardized shift std ≈ 0.29 vs state ≈ 1),
so **absolute** z_treated is ≈ z_control and even no-op scores high on absolute R²/cosine. We therefore
report metrics on the **shift** Δ = z_treated − z_control, against three baselines:
- **no-op**: ẑ_treated = z_control (Δ̂ = 0).
- **global mean-shift**: Δ̂ = average train shift (ignores drug identity).
- **per-drug mean-shift** (PRIMARY): Δ̂ = average train shift *for that drug* over the other (train)
  cell lines. Beating it requires capturing **cell-line-specific** modulation of the drug effect.

Splits: **held-out (cell_line, drug) pairs** (primary; drug and cell line both seen, the combination
not) and **held-out cell lines** (secondary). Multiple seeds; mean ± standard error.
Action-decodability: a linear probe predicting the drug from the latent shift (true shift = property of
the embedding space; predicted shift = idm_on vs idm_off comparison).

## Run
```bash
# 1) Build centroids (cluster, reads local shard; or HF download locally via --shards)
python -m examples.tahoe_probe.build_centroids \
    --parquet_paths /path/to/emb0.parquet --controls DMSO_TF --min_cells 50 \
    --out examples/tahoe_probe/data/centroids.npz

# 2) IDM ablation (held-out pairs + cell lines, multi-seed)
python -m examples.tahoe_probe.run_ablation_tahoe \
    --npz examples/tahoe_probe/data/centroids.npz --seeds 0,1,2,3,4 --splits pairs,celllines
```
Seeds drive both the split and the model init (paired idm_on vs idm_off per seed). All result tables in
`RESULTS.md` are MEASURED from real runs.
