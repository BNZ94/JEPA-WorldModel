# Microbiome-JEPA — Complete Work Report (git author BNZ94)

**Scope note.** This report documents my contribution (git author BNZ94) within the eb_jepa fork. The fork baseline (encoder/predictor/loss infrastructure of eb_jepa) is pre-existing; everything below is my added work. The document is an exhaustive quantitative inventory of every branch, experiment, measured number, figure, and file I produced. All quantitative results are quoted verbatim from committed JSON / figures / commit messages, with provenance. Values are labelled **MEASURED** (from a real run, with job ID where available), **EXPECTED** (a hypothesis or literature figure), or **ASSUMED/ESTIMATE**. Where two source sections disagree on a number, both are kept and flagged "(NOTE: sources disagree — verify)".

---

## Section 0 — Repository & branch map

| Branch | Purpose | Key artifacts | Headline result (MEASURED, exact) | Status |
|---|---|---|---|---|
| `main` | Fork baseline (pre-existing eb_jepa) | encoder/predictor/loss infra | — | baseline |
| `bnz` | Integration trunk: master REPORT.md / README.md / PLAN.md + all results JSON/figures folded in | `REPORT.md`, `README.md`, `/PLAN.md`, all `results/*` | Consolidates everything below | trunk |
| `bnz` (folded) — gLV IDM ablation | Collapse-and-recovery headline (M4) | `ablation_collapse.json/.png`, `ablation_default.json/.png` | `fast_r2_action` **0.748 ± 0.051 (IDM on) vs 0.520 ± 0.021 (off)**, Δ +0.229, on>off all 3 seeds | POSITIVE |
| `m2-realdata` (folded into `bnz`) | Real-data Layer A frozen probe vs Susagi MLP | `realdata_infants*.json`, `realdata.py` | Frozen JEPA + linear probe **acc 0.509 ± 0.014 / AUC 0.896 ± 0.001** vs Susagi MLP 0.527 / 0.890 (AUC tie) | DIAGNOSTIC (honest tie) |
| `bigbet-planning` (folded into `bnz`) | Controllability→representation→readout-fidelity planning diagnosis + tech-invariance | `oracle_K_sweep.*`, `planning_decoded_*.json`, `planning_diagnosis*.json`, `tech_invariance.json` | Planning **0% all methods**; decoded-MPPI weak-reg final **3.01 / best 2.58** (R²=0.89); tech-acc JEPA **0.952** (least invariant) | NEGATIVE (fully diagnosed) |
| `sigreg-rep` | SIGReg(LeJEPA)-vs-VICReg "one rep fixes all three" test | `realdata_infants_sigreg*.json`, `planning_diagnosis_k24_sigreg.json`, `tech_invariance_sigreg.json`, `losses.py` | M2 **win**; M3 **not fixed** (corr −0.23/−0.41); tech **not fixed** (0.967) | MIXED (1/3 positive) |
| `exp2-genejepa` | GeneJepa EMA-teacher + matched VICReg-d256 tech | `tech_invariance_vicreg_d256.json`, `tech_invariance.json` | EMA hurts M2; VICReg-d256 tech 0.960 | NEGATIVE (EMA) |
| `m3-learned-cost` | Learned monotonic cost head + horizon sweep | `planning_learned_{lowreg,d256,h2,h3,h4}.json` | Spearman 0.71→0.81 (capacity), planning flat 0% | NEGATIVE/diagnostic |
| `m3-model-fidelity` | Epistemic gate + on-policy (DAgger/MBPO) loop | `m3_ensemble_gate.json`, `m3_onpolicy.json` | Not epistemic; dist-shift real (3.7×) but on-policy still 0% | NEGATIVE/diagnostic |
| `m3-multistep-rollout` | Multi-step free-running rollout objective | `m3_multistep.json` | Free-run err 0.611→0.086 (−86%), planning still 0% | NEGATIVE/diagnostic |
| `m3-metric-loss-hybrid` | HYBRID isometry auxiliary (true-state supervision) to close the planning loop | `m3_metric_gate*.json`, `planning_learned_metric_mc*.json`, `m3_recognition*.json`, `metric_sweep_consolidated.{json,md}`, `metric_hybrid.png` | Raw-latent MPPI **0% → 100% ± 0%**, final **0.804 ≈ oracle 0.79**; Spearman 0.085 → 0.990 | POSITIVE (HYBRID, not pure JEPA) |
| `m3-generalization` | Exp1: closure across diverse gLV instances | `exp1_instance_screen.json`, `exp1_generalization.{json,md}`, `BONUS_SESSION_RESULTS.md` | 4/4 instances cross tol; 2/4 match oracle 100% | POSITIVE |
| `m3-idm-selfsup` | Exp2: IDM-reweight self-supervised closure | `exp2_idm.{json,md}` | 0% all idm_coeff; raw latent↔true Spearman driven negative (−0.313) | NEGATIVE (sharpens claim) |
| `m3-bottleneck` | Exp3: shrink d_model toward true-state dim | `exp3_dim.{json,md}` | 0% all dims; marginal Spearman ~0.2; costs recognition | NEGATIVE |
| `m2-tech-invariance` (current branch) | Confounding pre-screen, DANN adversary, CORAL/MMD alignment, UMAP/PCA | `REPORT_tech_invariance.md`, `tech_*.json`, `tech_*.png`, `main.py`, GRL in `architectures.py` | DANN FAILS (tech never <0.90); CORAL partial fix (SIGReg+CORAL −0.090 tech / ≈0 biome) | NEGATIVE (DANN) + PARTIAL POSITIVE (CORAL) |
| `tahoe-perturbation-probe` | Real-data generality check of action-conditioned predictor + IDM (drug perturbations) | `examples/tahoe_probe/*`, `results/ablation/ablation_results.json/.png` | Predictor (no IDM) beats per-drug baseline R²(shift) 0.201 vs 0.149; IDM HURTS (0.201→0.017) | POSITIVE (predictor) + NEGATIVE (IDM) + NEGATIVE (cross-line) |
| `figures-assets` | (figure/asset staging branch) | — | — | support |

---

## Section 1 — Methodology & environment

**Cluster / compute.** All GPU runs were on the **Dalia GB200 cluster (IDRIS, aarch64)** over SSH; SLURM jobs submitted via `sbatch` (partition `defq`, reservation `Vivatech`, account `vivatech-dynamics`). Evaluation/diagnosis frequently done on a CPU-only venv (`.venv-cpu`, torch 2.12.1) for fast shape/logic iteration. The gLV experiments are fully synthetic (no data download). Real-corpus work requires the cluster-only ~20–22 GB MicrobeAtlas / Susagi corpus.

**SLURM job IDs as provenance (collected across sections).** M4 ablation: 74504→74554 (divergence sweep), 74595 (failed, 8s), **74610** (headline). Real-data probe: **74841** (first), **74984** (fair re-run), 74933, 74966, 74996, 75032. Planning: **74610, 74718, 74841, 74933**. SIGReg M2: **40729fc** run 100ep/50k/d256; VICReg-100ep "pending (job 75195)". HYBRID metric: **75773 / 75774 / 75775**. Tech confounding: **76596**. DANN sweep: train 76698/76925/77350, eval **77533**. CORAL sweep: train **77650** / eval **77777**. Tech-invariance eval batches: jobs in `tech_eval_batch.py`. (Bonus-session Exp1–3 commit messages do not record explicit job IDs.)

**Scientific unit.** The recommended unit is the **three-seed sweep** with mean ± standard error (s.e. = std(ddof=1)/√n). Ablations and planning evals: 3 seeds × 12 episodes (planning) or 80-epoch trains (M4); tech sweeps: up to 6 seeds; real-data probes: 5-fold StratifiedKFold (s.e. over 5 folds); Tahoe: 5 seeds × 300 epochs.

**Integrity practices (followed throughout).** Every quantity labelled MEASURED / EXPECTED / ASSUMED. Negative and surprising results reported as-is, never papered over. Reversing seeds kept in figures. The central integrity catch (the unseeded-decoder 2.8% fluke corrected to 0% before folding) is documented in Section 6.2. Provenance gaps explicitly flagged (see "Open items").

---

## Section 2 — Layer A: set-JEPA encoder + gLV simulator + data pipeline + losses + eb_jepa integration

**Primary branch:** `bnz` (integration branch where Layers A+B were built; fork baseline is `main`). All paths read with `git show bnz:<path>`.

### 2.1 What was added to eb_jepa core (`git diff main bnz -- eb_jepa/`)

The diff against `main` adds **11 files / 2493 insertions, 0 deletions** (purely additive):

| File (`bnz:`) | Lines added | Role |
|---|---|---|
| `eb_jepa/architectures.py` | +174 | New class `SetTransformerEncoder` (only new class; `RNNPredictor`/`InverseDynamicsModel` pre-existing) |
| `eb_jepa/losses.py` | +140 | New classes `SIGReg_IDM_Sim_Regularizer` and `ImposterRepulsionLoss` |
| `eb_jepa/datasets/utils.py` | +8 | `init_data` early-return branch for `env_name == "microbiome"` |
| `eb_jepa/datasets/microbiome/glv.py` | +576 | gLV simulator |
| `eb_jepa/datasets/microbiome/otu_data.py` | +566 | Static OTU-set loader (Layer A) + real-data Susagi parsers + `init_microbiome_data` dispatch |
| `eb_jepa/datasets/microbiome/traj.py` | +320 | gLV trajectory `TrajDataset` (Layer B) |
| `eb_jepa/datasets/microbiome/transforms.py` | +247 | CLR, per-dim z-score, two-view augmentation, I-JEPA OTU masking |
| `eb_jepa/datasets/microbiome/__init__.py` | +10 | package init |
| `eb_jepa/_smoke_encoder.py`, `.../microbiome/_smoke_data.py`, `.../microbiome/_smoke_glv.py` | +154/+173/+125 | contract smoke tests |

The one real dispatch change (`utils.py`) is a clean early return:

```python
if env_name == "microbiome":
    from eb_jepa.datasets.microbiome.otu_data import init_microbiome_data
    return init_microbiome_data(cfg_data, device)
```

### 2.2 The permutation-invariant set-transformer encoder

**File:** `bnz:eb_jepa/architectures.py` (class `SetTransformerEncoder`, lines 457–629).

**Shape contract (verified vs `ImpalaEncoder`).** Input is an obs dict `{"otu": [B, T, N_max, F], "mask": [B, T, N_max]}`, `F = token_dim` (default 385 = 384 ProkBERT/species dims + 1 CLR log-abundance). Output is exactly **`[B, D, T, 1, 1]`** (H′=W′=1), matching the conv-encoder convention so the temporal predictor, `VC_IDM_Sim_Regularizer` ([B,C,T,H,W]), and the planning machinery work unchanged. `D == self.mlp_output_dim`; `self.final_ln` (LayerNorm) exposed for the predictor.

**Permutation invariance by construction.** Tokens embedded by a single `nn.Linear(token_dim → d_model)` with **no positional encoding**; passed through `nn.TransformerEncoder` (`norm_first=True`, GELU, `batch_first=True`) with `src_key_padding_mask = ~mask`; pooled with masked **mean** (default `pool="mean"`) or learned single-query **attention (PMA)** pool — both order-independent. Padded slots zeroed before input projection. T folded into batch (`[B*T, N_max, F]`), mirroring `TemporalBatchMixin`.

Smoke test `bnz:eb_jepa/_smoke_encoder.py` asserts: output shape exactly `[B,D,T,1,1]`; permutation-invariance (max-abs-diff < 1e-5); mask-invariance over padded slots (< 1e-5); composition with `VC_IDM_Sim_Regularizer` and `RNNPredictor`; finite `ImposterRepulsionLoss`.

### 2.3 The data pipeline (CLR, per-dim z-score, masking, augmentation)

**File:** `bnz:eb_jepa/datasets/microbiome/transforms.py`.

- **`clr(abund, pseudocount=1e-6)`** — centered log-ratio over OTU axis: `CLR(x)_i = log(x_i+pc) − mean_j log(x_j+pc)`. Removes sum-to-1 constraint; pseudocount handles sparsity/zeros.
- **`PerDimZScore`** — per-feature standardization over all F dims, fit on train tokens (optionally masked), serializable (`state_dict`/`load_state_dict`), `eps=1e-6` floors near-constant dims. Implements the mandate that the 384 embedding dims *and* the abundance dim be z-scored so abundance cannot dwarf the VICReg per-dim variance term.
- **`augment_community(...)`** (two-view SSL) — OTU subsample (`subsample_frac=0.8`), OTU dropout (`dropout_p=0.1`), Gaussian jitter on log-abundance (`jitter_std=0.1`); always keeps ≥1 real OTU.
- **`mask_otus(..., frac=0.5)`** — I-JEPA-style split of real OTUs into visible context + masked target (keeps ≥1 visible).

Token contract: `token = concat(z(ProkBERT_384 or species_emb), z(CLR_log_abundance))`, F=385.

**Static loader** (`bnz:eb_jepa/datasets/microbiome/otu_data.py`): `OTUSampleDataset` with modes `{two_view, masked, single}`; synthetic fallback (`synth_vocab=200` deterministic 384-d embeddings, sparse exponential abundances) for CPU smoke; real-data Susagi parsers (`load_prokbert_embeddings` reading h5 `embeddings` group, `parse_samples_otus_mapped`, A97/B97 key resolver) — *unverified on this Mac* (22 GB corpus cluster-only, flagged in docstring). `init_microbiome_data` dispatches `task="otu"` (Layer A) vs `task∈{glv,traj,temporal}` (Layer B).

### 2.4 The gLV simulator (the "Two Rooms" of microbiome)

**File:** `bnz:eb_jepa/datasets/microbiome/glv.py` (`GLVConfig`, `GLVSimulator`, `demonstrate_non_monotonicity`). Pure NumPy, deterministic from `config.seed`.

**Dynamics:** `dx/dt = x·(r + A·x) + m`, RK4-integrated, clamped non-negative; `m = immigration = 1e-3`.

**Default `GLVConfig`:** `n_species=32` (S), `n_candidate=8` (**K = action_dim**), `dt=0.05`, `steps_per_action=1`, `noise_std=0.0`, **`n_guilds=3`**, `self_lim=-1.0`, `within_frac=0.4`, `comp_strong=-2.5`, `comp_weak=-0.4`, `growth=1.0`, `action_max=0.5`, `init_seed_abundance=0.05`.

**Multistability:** species partitioned into `n_guilds` competitive guilds (within-guild weak mutualism; between-guild competition). Competitive exclusion makes each single-guild-dominant corner a stable attractor — verified via Jacobian (`jacobian_max_real_eig < 0`).

**Non-monotonic reachability (rubric-defining):** between-guild competition is **cyclic** (rock-paper-scissors): guild *g* strongly suppresses *(g+1)*, weakly suppressed back. Reaching a target "against the cycle" requires first blooming a gate guild — moving distance-to-target *up* before down. Greedy planner gets stuck; a detour succeeds. The Two-Rooms analog.

**K-candidate action panel:** `a ∈ R^K` (K = `n_candidate`) is a non-negative probiotic-dose delta on a fixed K-species panel (round-robin across guilds), clipped to `±action_max`. Continuous, low-dimensional, biologically meaningful; clean ground-truth interventions.

**MEASURED non-monotonicity** (provenance: commit `ef608b3`, restated in `bnz:REPORT.md` line 31): *"a greedy 'reduce distance every step' policy fails on 6/6 tested pairs."* With `n_guilds=3` there are 3×2 = 6 ordered (init,target) attractor pairs ⇒ `fraction_nonmonotonic = 6/6 = 1.000`. Smoke test `bnz:.../microbiome/_smoke_glv.py` asserts ≥2 attractors, local stability, stability under zero action, deterministic regeneration, `fraction_nonmonotonic > 0`.

### 2.5 gLV trajectory dataset (Layer B substrate)

**File:** `bnz:eb_jepa/datasets/microbiome/traj.py` (`GLVTrajConfig`, `GLVTrajDataset`, `init_microbiome_traj_data`). Wraps `GLVSimulator.generate_trajectories` into eb_jepa `TrajDataset` **5-tuple** `(obs_dict, act[T,K], state[T,S], reward[T], extra)`. **N_max = S** (every species a token slot; mask gates presence at `eps_present=1e-4`). Each species gets a fixed seeded 384-d species embedding (ProkBERT analog) so the *same* `SetTransformerEncoder` consumes gLV communities. Tokens CLR'd + z-scored as in Layer A. Exposes `action_dim=K`, `state_dim=S`, `token_dim=385`, `n_max=S`. A shape-faithful `_StubGLVSimulator` used only if `glv.py` absent (ablation JSONs confirm real sim ran: every record `"used_stub_glv": false`). `init_microbiome_traj_data` slices via `TrajSlicerDataset`/`get_train_val_sliced` into `num_frames`-windows, like `two_rooms`.

### 2.6 Regularizers / losses wiring

**World-model regularizers** (`forward(state[B,C,T,H,W], actions[B,K,T]) → (weighted, unweighted, dict)`):
- **`VC_IDM_Sim_Regularizer`** (pre-existing core) — VICReg variance (`std_coeff`) + covariance (`cov_coeff`) + temporal-similarity (`sim_coeff_t`, L_sim) + **inverse-dynamics** (`idm_coeff`, the M4 knob). Default Layer-B regularizer.
- **`SIGReg_IDM_Sim_Regularizer`** (NEW, `bnz:eb_jepa/losses.py` lines 306–379) — replaces VICReg std+cov with SIGReg/LeJEPA Epps–Pulley isotropy over `num_slices=256` random 1-D projections, applied to the **encoder output (no projector)**; keeps L_sim + IDM identical.

**Two-view SSL losses** (Layer A, `forward(z1,z2) → dict{"loss",...}`): `VICRegLoss(std_coeff, cov_coeff)` and `BCS` (SIGReg, `epps_pulley` + `lmbd=10.0`) — pre-existing core; Layer-A loop logs every extra dict key.

**Imposter-repulsion (NEW, optional creativity term, `bnz:eb_jepa/losses.py`):** `ImposterRepulsionLoss(margin, distance∈{l2,cosine})` — triplet hinge `relu(d(pred,real) − d(pred,imposter) + margin)` carrying Susagi's imposter-discrimination idea into JEPA. Composes with encoder in the smoke test; **not** part of the measured M4 ablation.

**action_dim plumbing:** `K` flows `GLVConfig.n_candidate → GLVTrajDataset.action_dim → data_config.action_dim`; builders set `RNNPredictor(hidden_size=D, action_dim=K)` (GRU: action=input, state=hidden) and `InverseDynamicsModel(state_dim=D, hidden_dim=256, action_dim=K)` from the loader's K (default K=6 in `layerB_worldmodel.yaml`), avoiding hardcoded `action_dim=2`. Actions transposed `[B,T,K] → [B,K,T]` for `unroll`/IDM.

### 2.7 Configs

**`bnz:examples/microbiome_jepa/cfgs/layerA_vicreg.yaml`** (static two-view): `token_dim=385`, `d_model=256`, `n_heads=4`, `n_layers=4`, `dim_feedforward=512`, `pool=mean`, `use_projector=true`; loss `vicreg` (`std_coeff=1.0`, `cov_coeff=25.0`); `n_max=256`, `emb_dim=384`, `batch_size=64`, `size=4096`, augmentation `subsample_frac=0.8/jitter_std=0.1/dropout_p=0.1`; `epochs=100`, `lr=1e-3`. Commit `ef608b3` CPU smoke: **loss 1.41→1.29, feat_std ≈ 0.91 stable (no collapse)** (2-epoch synthetic).

**`bnz:examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml`** (gLV world model): `task=glv`, `n_traj=256`, `T=24`, `n_species=24`, `n_candidate=6` (**K=action_dim=6**), `dt=0.05`, `noise_std=0.01`, `num_frames=8`; `d_model=256` (=predictor hidden), `nsteps=7`; regularizer `std_coeff=1.0, cov_coeff=25.0, sim_coeff_t=1.0, idm_coeff=1.0` (ablation knob, `idm_hidden=256`, `use_proj=true`); `epochs=60`, `lr=1e-3`, `grad_clip=1.0`.

---

## Section 3 — Collapse / IDM ablation (M4 headline)

### 3.1 Hypothesis (Sobal et al., arXiv 2211.10831) — EXPECTED

A JEPA preferentially encodes **slow** features. On gLV trajectories the slow feature is the trajectory's static **identity** (initial composition / basin); the **fast** signal is the time-varying state, its one-step change, and the applied **intervention**. A temporal JEPA trained only to predict next-latent should **collapse onto slow features** and discard fast dynamics. **Adding the IDM term** (encoder must let an inverse model recover the action from consecutive latents) should **restore fast decodability**, slow-identity decodability ≈ unchanged.

### 3.2 Method

**Driver:** `bnz:examples/microbiome_jepa/run_ablation.py`. For each seed × arm (`idm_on`: `idm_coeff=1.0`; `idm_off`: `idm_coeff=0.0`) trains the Layer-B world model (`train_worldmodel.py`), then **freezes the encoder** and runs `bnz:examples/microbiome_jepa/eval_collapse.py` on **held-out** gLV trajectories (held-out `sim_seed = 10000 + seed`). **Probes** are Ridge regressions with trajectory-level train/test split (no timepoint leakage) and floored per-dim standardization (sd<1e-4 → 1.0, fixed in commit `6907019`):
- `fast_r2_action`: decode applied intervention `a_t` from `[z_t, z_{t+1}]` (a fresh probe, not the trained IDM — random-init encoder ≈0, cleaned in commit `2423ef0`),
- `fast_r2_delta`: decode one-step change `x_{t+1}−x_t`,
- `fast_r2_state`: decode current state `x_t`,
- `slow_r2_init`: decode initial state `x_0` (slow feature),
- `feat_std`: mean per-dim std of latents.

Both arms: **3 seeds (0,1,2), 80 epochs, n_traj=256, eval_n_traj=128, d_model=128, use_amp=false**, real gLV (`used_stub_glv=false`).

### 3.3 MEASURED results

**A) Induce-collapse regime (THE HEADLINE)** — weak variance-reg `sim_coeff_t=4, cov_coeff=1, std_coeff=0.25`.
Provenance: `bnz:.../results/ablation_collapse.json`; figure `ablation_collapse.png`; commit `0e542b7` (job **74610**).

| probe (R², held-out, mean ± s.e., 3 seeds) | IDM on | IDM off | Δ (on−off) |
|---|---|---|---|
| `fast_r2_action` (intervention) | **0.7483 ± 0.0512** | **0.5197 ± 0.0207** | **+0.2286** |
| `fast_r2_delta` (one-step dynamics) | 0.8185 ± 0.0226 | 0.7362 ± 0.0120 | +0.0823 |
| `fast_r2_state` (current state) | 0.9736 ± 0.0032 | 0.9626 ± 0.0021 | +0.0110 |
| `slow_r2_init` (identity, slow) | 0.9934 ± 0.0008 | 0.9888 ± 0.0027 | +0.0046 (saturated) |
| `fast_minus_slow` | −0.1749 ± 0.0234 | −0.2526 ± 0.0135 | +0.0777 |
| `feat_std` | 0.01074 ± 0.00057 | 0.00738 ± 0.00011 | +0.00336 |

Per-seed `fast_r2_action` (on / off): seed0 0.7332 / 0.4787; seed1 0.8435 / 0.5456; seed2 0.6681 / 0.5347 — **on > off all 3 seeds, error bars non-overlapping**. Training: IDM-on `train_pred ≈ 7e-5`, `train_idm_loss ≈ 0.010–0.015`; IDM-off `train_idm_loss = 0.0`.
**Interpretation (POSITIVE):** in the collapse-prone regime the encoder drops the intervention without IDM (0.52 vs 0.75); IDM rescues the **fast** signal while slow identity stays saturated (~0.99). Textbook collapse-and-recovery.

**B) Default regime** — strong VICReg `std_coeff=1.0, cov_coeff=25.0, sim_coeff_t=1.0`.
Provenance: `bnz:.../results/ablation_default.json`; figure `ablation_default.png`; commit `0e542b7`.

| probe (R², mean ± s.e., 3 seeds) | IDM on | IDM off | Δ (on−off) |
|---|---|---|---|
| `fast_r2_action` (intervention) | 0.3641 ± 0.0199 | 0.2910 ± 0.0409 | +0.0731 |
| `fast_r2_delta` | 0.6679 ± 0.0085 | 0.6433 ± 0.0124 | +0.0246 |
| `fast_r2_state` | 0.9452 ± 0.0028 | 0.9381 ± 0.0051 | +0.0072 |
| `slow_r2_init` | 0.9769 ± 0.0015 | 0.9827 ± 0.0029 | −0.0058 (saturated) |
| `fast_minus_slow` | −0.3090 ± 0.0086 | −0.3394 ± 0.0112 | +0.0304 |
| `feat_std` | 0.01597 ± 0.00086 | 0.01424 ± 0.00046 | +0.00173 |

Per-seed `fast_r2_action` (on/off): seed0 0.3754 / 0.2229; seed1 0.3915 / 0.2859; **seed2 0.3253 / 0.3642 (reversed)** → only 2/3 seeds favor IDM.
**Interpretation (honest, modest):** strong VICReg partially substitutes for IDM; gap shrinks to +0.073 and is seed-noisy (1 reversal). The controlled-comparison conclusion (`bnz:REPORT.md`, commit `0e542b7`): **the IDM effect is regime-dependent** — large/robust in the collapse-prone regime, small/noisy when the variance regularizer is already strong. Not overclaimed.

### 3.4 Figures
- **`bnz:.../results/ablation_collapse.png`** — grouped bars (3 probe categories × IDM on/off, mean±s.e., 3 seeds, 80 ep), induce-collapse regime: IDM-on (green) >> IDM-off (red) on fast:action (~0.75 vs ~0.52), smaller gap on fast:Δstate, slow:init saturated equal (~0.99).
- **`bnz:.../results/ablation_default.png`** — same layout, strong-VICReg default: small action gap (~0.36 vs ~0.29, overlapping error bars), Δstate near-equal, identity saturated.

### 3.5 Provenance chain (commits)
`ef608b3` (Layer A + gLV + encoder/data/losses, CPU-green; non-monotonicity 6/6) → `cd75b67` (Layer B world-model + IDM-ablation harness; not yet a result) → `a0bba72` (M4 collapse-metric eval + driver; random-init sanity slow 0.97 ≫ fast −0.30) → `c5e939c` (CLI sweep knobs + GPU sweep) → `bfd3feb` (perf: feat_std once/epoch; d_model=128/60ep/256traj) → `2423ef0` (standardize-X-before-Ridge fix; jobs 74504→74554: default 0.241 vs 0.128, collapse 0.277 vs 0.172) → `6907019` (fix tuple-seeds parsing after job 74595 failed in 8s; floor per-dim std) → `0e542b7` (**M4 HEADLINE**, job 74610) → `fc89f2b` (hands off to M3 planning).

---

## Section 4 — M2 downstream probe + real DIABIMMUNE/Susagi data + Susagi baselines + fairness corrections

**Branches:** `bnz` (superset), `m2-realdata` (plumbing), `sigreg-rep` (subset), `exp2-genejepa` (subset).
**Key commits:** `ef7162d` (plumbing), `cd30593` (probe + baseline port + tech-invariance), `95b82c2` (fairness: MLP probe + corpus z-score + finetune), `e66e6d7` (big-run launcher), `40729fc` (SIGReg M2 JSON), `d85ec6f` (EMA-teacher gate), `091ddda` (EXP2 fold: EMA hurts, d384 best), `6a82f96` (tech-invariance + Susagi imposter rep), `c9144d3` (M2 folded into report).

### 4.1 What M2 tested

A **frozen** set-JEPA community embedding, probed with a simple classifier on the real **Susagi infant-environment** task, should match/beat the Susagi MLP baseline that uses the same classifier on raw abundance features.

**Task (`infants_env`):** predict `Env` (**12 classes**) for **2036** infant gut-microbiome samples. Self-contained in `data/infants/` (`infants_otus.tsv`, `meta_withbirth.csv`, Susagi's `abundance.csv`) — does NOT need the 20 GB corpus, only the corpus-pretrained encoder checkpoint. Verified (commit `ef7162d`): "2036 samples, 12 Env classes."

**Code (on `bnz`):**
- **`probe_downstream.py`** (commit `cd30593`) — original WS4 probe. `encode_samples` runs frozen `SetTransformerEncoder` over `mode="single"` OTU samples to `Z=[N,D]`; `linear_probe` does Susagi-matched CV (`GroupKFold` if host groups, else `StratifiedKFold(shuffle, random_state=42)`), reports accuracy + macro OVR ROC-AUC; `_probe_logreg` = `LogisticRegression(lbfgs, l2, C=10, multinomial, max_iter=2000)` — identical to Susagi's `predict_env.py`. `tech_invariance` decodes sequencing technology. Real-data label loaders flagged `# UNVERIFIED`.
- **`baselines_port.py`** (commit `cd30593`) — faithful Susagi port. `make_susagi_logreg`/`make_susagi_mlp` copy hyperparameters from `scripts/infants/predict_env.py` (LogReg) and `scripts/diabimmune/base_lines_mlp.py` (`MLPClassifier(hidden_layer_sizes=(128,), relu, adam, alpha=1e-4, lr=1e-3, max_iter=200, seed=42)`). `macro_roc_auc` = Susagi metric (OVR macro for >2 classes). `SUSAGI_SEED=42`.
- **`realdata.py`** (commits `ef7162d`, `95b82c2`) — self-contained driver used for measured numbers. Loads infant communities, resolves B97 OTU ids → ProkBERT embeddings (`prokbert_embeddings.h5` + `otus.rename.map1`), CLR + z-score, frozen-encoder, runs both linear probe (`LogisticRegression(C=10)`) and **MLP probe** (`MLPClassifier(128)`) on the *same* embeddings, plus the **Susagi MLP baseline on `abundance.csv`** (`standardize=False`), and an optional supervised `finetune_upper_bound`. s.e. = std(ddof=1)/√5 over 5 folds.

**Configs/launchers (`bnz`):** `cfgs/layerA_real.yaml` (two-view VICReg, `token_dim=385`, `d_model=128` overridable, VICReg `std_coeff=1.0, cov_coeff=25.0, lmbd=10.0`, `epochs=30`, `lr=1e-3`, bf16). Launchers: `run_realdata.sh` (VICReg 30ep/20k/d128), `run_realdata_eval.sh` (CPU fairness re-eval, no retrain), `run_realdata_big.sh` (VICReg 100ep/50k/d256 + finetune), `run_realdata_sigreg.sh` (SIGReg via `--loss.type bcs`, 100ep/50k/d256).

### 4.2 The fairness story

The first M2 run (`realdata_infants.json`, commit `c9144d3`) compared **JEPA linear probe (LogReg)** vs **Susagi MLP baseline** on an **infant-token z-score** — two confounds: (a) probe-class mismatch (linear vs MLP), (b) z-score inconsistent with pretraining corpus. JSON flag: `"zscore_note": "z-score fit on infant tokens (corpus z-score not persisted) — approximation"`.

Fairness fixes (commit `95b82c2`, `run_realdata_eval.sh` → `realdata_infants_fair.json`):
1. **Apples-to-apples probe:** add frozen **MLP probe** (`MLPClassifier(128)`, same class as baseline) on the same embeddings. Encoder stays frozen.
2. **Corpus z-score:** fit `PerDimZScore` on 5000 **corpus** samples (`zscore_source` → `"corpus_5000_samples"`).
3. **Optional supervised finetune upper bound** (labeled non-headline).

### 4.3 Exact measured numbers

Literal contents of `bnz:.../results/*.json`. Format: `acc_mean ± acc_se` / `auc_mean ± auc_se`. All: `task=infants_env`, `n_samples=2036`, `n_classes=12`, `n_max=256`, `pretrained_encoder=true`, 5-fold StratifiedKFold.

| Run (file) | Loss / setup | d_model | z-score | JEPA **linear** (acc / AUC) | JEPA **MLP** (acc / AUC) | **Finetune** upper bound (acc / AUC) |
|---|---|---|---|---|---|---|
| `realdata_infants.json` (run#1, `c9144d3`) | VICReg 30ep/20k | 128 | infant-token (approx) | **0.5079 ± 0.0069** / **0.8960 ± 0.0028** | — | — |
| `realdata_infants_fair.json` (fairness, `6a82f96`) | VICReg 30ep/20k | 128 | corpus_5000 | 0.5088 ± 0.0137 / 0.8961 ± 0.0015 | 0.4995 ± 0.0082 / 0.8884 ± 0.0016 | — |
| `realdata_infants_sigreg.json` (EXP1, `40729fc`) | **SIGReg** 100ep/50k | 256 | corpus_5000 | 0.5138 ± 0.0054 / 0.8912 ± 0.0012 | **0.5255 ± 0.0075** / 0.8936 ± 0.0024 | **0.5899 ± 0.0083** / **0.9180 ± 0.0036** |
| `realdata_infants_sigreg_d384.json` (EXP2, `091ddda`) | **SIGReg** 100ep/50k | **384** | corpus_5000 | 0.5044 ± 0.0052 / 0.8894 ± 0.0026 | **0.5309 ± 0.0040** / **0.8986 ± 0.0027** | 0.5855 ± 0.0097 / 0.9063 ± 0.0015 |
| `realdata_infants_sigreg_ema.json` (EXP2, `091ddda`) | SIGReg **+EMA teacher** 100ep/50k | 256 | corpus_5000 | 0.4474 ± 0.0094 / 0.8534 ± 0.0029 | 0.5074 ± 0.0089 / 0.8848 ± 0.0058 | 0.5516 ± 0.0079 / 0.8965 ± 0.0038 |

**Susagi MLP baseline** (MLP on true `abundance.csv`; identical in every JSON): **acc 0.5270 ± 0.0102 / macro-AUC 0.8898 ± 0.0020**.
**Susagi reported reference** (their `env_predictions.txt`, EXPECTED/reference, not a run we executed): **acc 0.549 / macro-AUC 0.912**.

> **Filename caveat:** `realdata_infants_sigreg.json` holds the **d256** run (`"d_model": 256`); d384 lives in `realdata_infants_sigreg_d384.json`.

### 4.4 Comparison & what beats the baseline

- VICReg 30ep/20k/d128 MLP probe **0.4995 / 0.8884** → **below** the Susagi MLP baseline.
- SIGReg 100ep/50k/d256 MLP probe **0.5255 / 0.8936** → **matches** baseline acc (0.5255 vs 0.5270, overlapping SE) and **beats** AUC (0.8936 > 0.8898). Commit `40729fc`: "MLP 0.526/0.894 (matches Susagi MLP)."
- SIGReg 100ep/50k/d384 MLP probe **0.5309 / 0.8986** → **best frozen result**, **beats** baseline on both. Commit `091ddda`: "d384 capacity best (0.531/0.899, beats Susagi MLP)."

**d256 vs d384 (SIGReg):** capacity helps frozen MLP probe (0.5255→0.5309 acc; 0.8936→0.8986 AUC) but linear probe flat/down (0.5138→0.5044) and finetune ceiling lower at d384 (0.5899→0.5855 acc; 0.9180→0.9063 AUC).

**EMA on/off (SIGReg, d256):** EMA **hurts** — linear 0.5138→**0.4474** (AUC 0.8912→0.8534), MLP 0.5255→0.5074, finetune 0.5899→0.5516. Commit `091ddda`: "EMA teacher hurts M2 (0.507 vs 0.526) ... EMA negative."

**Linear vs MLP vs finetune:** linear never clears the baseline (best linear acc 0.5138). MLP probe matches (d256) then beats (d384). Finetune upper bound (acc **0.5899 / AUC 0.9180** at SIGReg-d256) **beats the Susagi *reported* reference** (0.549/0.912). Labeled in every JSON as a SUPERVISED upper bound, not the label-free claim.

### 4.5 Honest verdict
- **Negative (initial):** unfair VICReg-d128 and fair VICReg-d128 MLP (0.4995/0.888) did NOT beat baseline.
- **Positive (frozen):** SIGReg + scale flips it — matches at d256, beats at d384. Headline Layer-A frozen-rep win.
- **Positive (supervised ceiling):** finetune (0.5899/0.9180 at SIGReg-d256) beats published Susagi reference — explicitly not the label-free claim.
- **Negative (diagnostic):** EMA teacher consistently degrades M2.
- **N/A:** tech-invariance N/A on infant task (Instrument = 100% Illumina MiSeq, single class); tech-invariance probe runs on a multi-tech corpus subset (Section 7).

### 4.6 Figures
No M2-specific figure exists. The only `results/*.png` (`ablation_collapse`, `ablation_default`, `metric_hybrid`, `oracle_K_sweep`, `planning_diagnosis`, `planning_success_rate`) belong to Layer-B planning/ablation. M2 is reported as the numbers table.

---

## Section 5 — SIGReg vs VICReg (across M2/M3/tech)

**Branches:** `sigreg-rep` (primary), `exp2-genejepa` (GeneJepa EMA + matched VICReg-d256). M2 capacity/EMA files (`realdata_infants_sigreg_d384.json`, `realdata_infants_sigreg_ema.json`) live on `bnz`.

**Core hypothesis (EXPECTED, commit `3aef7f9`):** SIGReg/LeJEPA replaces VICReg's per-dim variance hinge + off-diagonal covariance penalty with an **Epps–Pulley characteristic-function test** pushing the latent toward an isotropic standard Gaussian along random 1-D slices. The bet: an isotropic-Gaussian latent should (a) be a better frozen rep, (b) make Euclidean distance meaningful so planning works, (c) be more technology-invariant. **Outcome: SIGReg wins on M2; negative on M3 geometry and tech-invariance; EMA add-on hurts. 2 of 3 legs negative.**

### 5.1 Code added
- **`eb_jepa/losses.py` (sigreg-rep), `SIGReg_IDM_Sim_Regularizer` (L306–379, commit `98be526`)** — world-model regularizer swapping VICReg std+cov for SIGReg isotropy on the encoder output (no projector); keeps `sim_coeff_t` + `idm_coeff`. `_sigreg(z)` draws `num_slices=256` random unit projections (seeded by `self.step`), returns `epps_pulley(z @ A).mean()`. Knobs: `sigreg_coeff, sim_coeff_t, idm_coeff, num_slices, first_t_only`.
- **`epps_pulley(x, t_min=-3, t_max=3, n_points=10)` and `BCS(num_slices=256, lmbd=10.0)` (L435–477)** — two-view SSL path (`loss.type=bcs`): `total_loss = MSE(z1,z2) + lmbd * mean(epps_pulley(view1,view2))`, dict with `loss`/`bcs_loss`/`invariance_loss`.
- **`train_worldmodel.py` (sigreg-rep), `reg_type` knob (L147):** `reg_type = str(rcfg.get("type","vicreg")).lower()`; `if reg_type == "sigreg"` → `SIGReg_IDM_Sim_Regularizer`, else `VC_IDM_Sim_Regularizer`.
- **`diagnose_planning.py`, DIAG2 `feat_std` (commit `9f4194c`):** `feat_std = float(torch.stack(zs_all).std(dim=0).mean())` ("~1 isotropic, ~0 squished").
- **EMA teacher (GeneJepa-style, config-gated `use_ema`, commit `d85ec6f`):** `ema_decay=0.996`; view2 target from stop-grad EMA copy (BYOL/DINO/I-JEPA style). Launched via `run_realdata_ema.sh` with `--loss.type bcs --model.use_ema true --model.ema_decay 0.996`.

### 5.2 Leg 1 — M2 downstream probe: SIGReg WINS
(See Section 4.3 table.) References: Susagi MLP baseline acc 0.5270 ± 0.0102 / AUC 0.8898 ± 0.0020; Susagi reported 0.549 / 0.912. SIGReg MLP probe 0.5255 (d256) / **0.5309 (d384)** matches/beats baseline; d384 MLP AUC 0.8986 beats 0.8898; finetune (0.5899/0.9180) beats reference. Win scales with capacity. Commits `40729fc`, `091ddda`. **Verdict: POSITIVE.**

### 5.3 Leg 2 — M3 planning geometry gate: SIGReg NEGATIVE
File `planning_diagnosis_k24_sigreg.json` (sigreg-rep, commit `e2e6b4f`), K=24, `reg_type=sigreg`.

| Quantity | SIGReg WM (MEASURED) | VICReg ref (from commit msgs) |
|---|---|---|
| Encoder `feat_std` (isotropy gate) | **0.5045** | ~0.01 |
| Latent-vs-true distance Pearson | **−0.2338** | ≈ 0 |
| Latent-vs-true distance Spearman | **−0.4068** (n=120) | ≈ 0 |
| Rollout `norm_div_t1` | 0.0803 | — |
| Rollout `norm_div_mean` (H=20) | **0.6103** | ~0.01 |
| Rollout `norm_div_tH` | 0.8085 | — |
| Oracle planner success_rate | 1.0 (final 0.777, start 6.639, tol 0.996, n_pairs 6) | — |
| Learned-MPPI success_rate | **0.0** (job 74718: final 4.88 ≥ random 4.58) | — |

SIGReg de-squished the latent (`feat_std` 0.01→0.50) but distances stay anti-correlated and rollout degrades (0.01→0.61). VICReg's low rollout divergence was partly a *collapse artifact*. M3 stopped per geometry-gate rule. **Verdict: NEGATIVE/diagnostic.** Commit `e2e6b4f`.

### 5.4 Leg 3 — Technology-invariance: SIGReg NEGATIVE
Files `tech_invariance_sigreg.json` (sigreg-rep, `18691f8`), `tech_invariance_vicreg_d256.json`, `tech_invariance.json` (exp2-genejepa). Setup: n=4960, per-class cap 2500, amplicon 2461 / wgs 2499; tech probe 2-class (chance 0.5038); biome probe 8-class on n=4062 (chance 0.5325). Lower tech bal-acc = better; higher biome = better.

**Tech (LOWER better):** SIGReg 0.9669 ± 0.0022 > VICReg-d256 0.9599 ± 0.0027 > VICReg-d128 0.9524 ± 0.0022 > raw_meanpool 0.9380 ± 0.0049 > random_encoder 0.9231 ± 0.0043 > Susagi imposter 0.8914 ± 0.0030.
**Biome (HIGHER better):** VICReg-d256 0.8432 ± 0.0230 > SIGReg 0.8163 ± 0.0206 > VICReg-d128 0.8050 ± 0.0320 > raw 0.7937 ± 0.0236 > random 0.7634 ± 0.0327 > Susagi 0.7192 ± 0.0216.

Every learned JEPA makes technology *more* decodable than raw/random; **SIGReg encodes the MOST technology** (opposite of the goal), and does not buy extra biology. **Verdict: NEGATIVE.** Commit `18691f8`. This drove the DANN/CORAL work (Section 7).

### 5.5 EMA-teacher (GeneJepa, EXP2): NEGATIVE
Measured on M2 infants (`realdata_infants_sigreg_ema.json`, `091ddda`): linear acc 0.4474 (vs 0.5138 no-EMA), MLP acc 0.5074 (vs 0.5255, below Susagi MLP 0.5270), MLP AUC 0.8848. Commit `091ddda`: "EMA teacher hurts M2 (0.507 vs 0.526)." Plain SIGReg + extra capacity (d384) was the better lever.

### 5.6 SIGReg verdict
- M2 (probe): **POSITIVE** (`40729fc`, `091ddda`).
- M3 (geometry gate): **NEGATIVE/diagnostic** (feat_std 0.01→0.50 but Pearson −0.23, Spearman −0.41, rollout 0.01→0.61, MPPI 0%) (`e2e6b4f`).
- Tech: **NEGATIVE** (tech bal-acc 0.967, most of all) (`18691f8`).
- EMA: **NEGATIVE** (`091ddda`).

Net: isotropic-Gaussian latent is a better downstream representation but does not buy metric planning geometry or technology-invariance; EMA does not help. 2/3 legs (+EMA) negative.

Commit chain: `3aef7f9` → `98be526` → `9f4194c` → `e2e6b4f` → `8bbeb3e`/`a86be48`/`f6e15a0` → `18691f8` → `40729fc`/`005145c` → `d85ec6f` → `091ddda`.

---

## Section 6 — Layer B / Planning saga

**Branch:** `bnz` (all Part-1 artifacts; `bigbet-planning` fully contained in `bnz`). Task: given a trained gLV world model (`SetTransformerEncoder` f_θ + GRU `RNNPredictor` g_φ), plan a sequence of continuous K-dim interventions to drive the simulated community from a start attractor to a target. Success on the **true gLV state**: executed plan applied to the real `GLVSimulator`, episode succeeds iff true final state within `tol = 0.9957860399240593` (= `tol_frac=0.15` × inter-attractor distance `attr_scale = 6.6386`).

**Code inventory (`git show bnz:<path>`):**

| File | Role |
|---|---|
| `examples/microbiome_jepa/plan_glv.py` | Latent-space MPPI planner + baselines (`random`, `greedy`, `final_only`, `mppi`) |
| `examples/microbiome_jepa/plan_glv_decoded.py` | Decoded planner: fit z→x̂ readout (Ridge + 1-hidden MLP), score MPPI in decoded space |
| `examples/microbiome_jepa/oracle_K_sweep.py` | Pure-numpy oracle (state-space MPPI on true gLV) across action-panel sizes K |
| `examples/microbiome_jepa/diagnose_planning.py` | 3-layer diagnosis: oracle / latent-cost alignment / rollout accuracy |
| `examples/microbiome_jepa/make_planning_figure.py` | Builds planning figure from committed JSONs only |
| Launchers | `run_glv_plan.sh`, `run_glv_plan_k24.sh`, `run_glv_k24_big.sh`, `run_glv_k24_lowreg.sh`, `run_glv_k24_d256.sh`, `run_glv_final.sh` |

**Planners.** `random` (random doses); `greedy` (1-step oracle-ish in true state space, K candidates × 2 dose levels + no-op, picks max distance-reduction — the baseline non-monotonicity is designed to defeat); `final_only` (latent MPPI, `cumulative=False`); `mppi` (latent, encode start once, roll GRU forward H steps in latent space, score cumulative L2 latent distance to target rep, exp-weighted elite refit). MPPI default (`plan_glv.py`): `horizon=8, n_samples=256, n_elites=32, n_iters=4, temperature=1.0, init_std=0.25, min_std=0.02, cumulative=True`; actual runs used `horizon=6, n_samples=128, n_elites=16–32, n_iters=3`. Decoded MPPI (`mppi_decoded_linear`, `mppi_decoded_mlp`) = same latent rollout, cost in decoded-state space.

### 6.1 Oracle controllability — task uncontrollable below K=24, fully controllable at K=24

**File:** `bnz:.../results/oracle_K_sweep.json` · commits `ab975bf` (single-seed), `0102309` (3-seed). Perfect-model state-space MPPI on true gLV, swept over K. K is pure actuation (attractors depend only on guild structure), so tol/targets identical across K (`same_tol_across_K: true`). Settings: `n_species=24, n_guilds=3, action_max=0.5, horizon=6, mpc_steps=20, n_samples=96, n_elites=16, n_iters=3, tol_frac=0.15`, seeds [0,1,2], 6 pairs.

| K | Oracle success (mean ± SE) | Per-seed | Mean final dist (mean ± SE) |
|---|---|---|---|
| 6 | 0.000 ± 0.000 | [0,0,0] | 4.0889 ± 0.0017 |
| 9 | 0.000 ± 0.000 | [0,0,0] | 3.7194 ± 0.0006 |
| 12 | 0.000 ± 0.000 | [0,0,0] | 3.3234 ± 0.0011 |
| 15 | 0.000 ± 0.000 | [0,0,0] | 2.8867 ± 0.0014 |
| 18 | 0.000 ± 0.000 | [0,0,0] | 2.3766 ± 0.0019 |
| 21 | 0.000 ± 0.000 | [0,0,0] | 1.7310 ± 0.0022 |
| **24** | **1.000 ± 0.000** | **[1,1,1]** | **0.7901 ± 0.0164** |

`best_K=24`, `best_success_rate=1.0`, `controllability_onset_K=24`, `controllable=true`. With a perfect model, mean final distance decreases monotonically (4.09→0.79), but success only crosses tol once all 24 species are dose-able. Proves the first K=6 negative was **structural uncontrollability of the task spec**, not model failure. Motivated retraining the world model at K=24.

### 6.2 Initial latent-MPPI (NEGATIVE) + the integrity fix

**Result 1 — first latent-MPPI: honest NEGATIVE (all methods 0%).** File `bnz:.../results/planning_results.json`, commit `d9d0d54`, job 74718, K=6 model, 3 seeds × 12 episodes (36/method), `mpc_steps=20`, `tol=0.99579`, `attr_scale=6.6386`, `mean_start_dist=6.6386`.

| Method | Success (mean ± SE) | Per-seed | Mean final dist | Mean best dist |
|---|---|---|---|---|
| random | 0.000 ± 0.000 | [0,0,0] | 4.5752 | 4.4989 |
| greedy | 0.000 ± 0.000 | [0,0,0] | 4.5130 | 4.5099 |
| final_only | 0.000 ± 0.000 | [0,0,0] | 4.8740 | 4.8664 |
| **mppi (latent)** | **0.000 ± 0.000** | [0,0,0] | **4.8814** | 4.8600 |

Every method 0%; learned latent-MPPI final (4.88) *worse* than random (4.58) and greedy (4.51). Reported as "honest NEGATIVE."

**THE INTEGRITY FIX (commit `00a05a1`) — the 2.8% "success" was an unseeded decoder-init fluke.** An earlier decoded-planning run (`bnz:.../results/planning_decoded.json`, stale pre-fix copy committed as a "stray" at `6a6fc54`, weak-reg `plan_model_k24_lowreg`, MLP R²=0.8939) reported `mppi_decoded_mlp` success_rate = **0.027778 ± 0.027778**, **per_seed = [0.08333, 0.0, 0.0]**, final 2.7840, best 2.5910 — one success in one seed only. Traced to an **unseeded MLP decoder initialization** (Adam, no fixed seed). The fix is a single line in `plan_glv_decoded.py:fit_decoders`:

```python
# ---- mlp (1 hidden layer), trained with Adam on standardized z (seeded for reproducibility) ----
torch.manual_seed(0)
```

With the decoder seeded the result is reproducibly **0% success**. Report, README, and figure corrected to seeded numbers before folding. Commit message: *"INTEGRITY FIX: seed decoder; the 2.8% planning success was an unseeded-init fluke … The robust readout-fidelity trend is in FINAL DISTANCE (4.12→3.01), not success."* Logged verbatim in PLAN.md (`c2a275c`).

### 6.3 Decoded-state planning — readout fidelity improves *distance*, not success

Canonical, integrity-fixed (seeded) numbers in `planning_decoded_default.json` (default-reg K=24, `plan_model_k24`) and `planning_decoded_lowreg.json` (weak-reg K=24, `plan_model_k24_lowreg`) — both rewritten in `00a05a1`. Both: `action_dim=24`, 3 seeds, 12 episodes, `mpc_steps=20`, `mppi_cfg={horizon 6, n_samples 128, n_elites 16, n_iters 3, cumulative true}`, `tol=0.99579`, start 6.6386.

| File / regime | Readout R² (lin / MLP) | Method | Success (mean ± SE) | Mean final dist | Mean best dist |
|---|---|---|---|---|---|
| **default** (`plan_model_k24`) | **0.7561 / 0.7784** | random | 0.000 ± 0.000 | 3.9939 | 3.5801 |
| | | greedy | 0.000 ± 0.000 | 3.2612 | 3.2611 |
| | | mppi_latent | 0.000 ± 0.000 | 4.2584 | 4.0277 |
| | | mppi_decoded_linear | 0.000 ± 0.000 | 4.0722 | 3.7886 |
| | | mppi_decoded_mlp | **0.000 ± 0.000** | 4.1218 | 3.7420 |
| **lowreg** (`plan_model_k24_lowreg`) | **0.8420 / 0.8921** | random | 0.000 ± 0.000 | 3.9939 | 3.5801 |
| | | greedy | 0.000 ± 0.000 | 3.2612 | 3.2611 |
| | | mppi_latent | 0.000 ± 0.000 | 4.4894 | 4.3695 |
| | | mppi_decoded_linear | 0.000 ± 0.000 | 3.1275 | 2.9291 |
| | | **mppi_decoded_mlp** | **0.000 ± 0.000** | **3.0147** | **2.5778** |

Higher readout fidelity (R² 0.78→0.89) improves how close decoded-MPPI gets (default-reg `mppi_decoded_mlp` final 4.12 → weak-reg 3.01; best 3.74 → 2.58); weak-reg decoded planner is the only method to beat greedy's final distance (3.01 < 3.26). **But success stays 0% across all seeds/regimes** at tol≈1.0. The robust readout-fidelity trend lives in **final distance (4.12→3.01)**, not success.

### 6.4 Layered diagnosis — bottleneck = representation geometry

**Result 3 — learned latent-MPPI at K=24 still fails (job 74933).** File `bnz:.../results/planning_diagnosis_k24.json`, commit `8f7b992` (K=24 default-reg).

| Diagnostic layer | Metric | Value | Reading |
|---|---|---|---|
| (1) Oracle / controllability | success_rate | **1.000** (final 0.7771, start 6.6386, n_pairs 6) | task solvable |
| (2) Latent-cost alignment | Pearson / Spearman (n=120) | **−0.0023 / +0.0247** | latent distance **uninformative** |
| (3) Rollout accuracy (H=20) | norm_div t=1 / t=H / mean | 0.0225 / 0.0196 / **0.0193** | world model **faithful** (~2%) |
| Learned latent-MPPI | success_rate | **0.000** | planner fails despite (1)+(3) |

(K=6 default-reg `planning_diagnosis.json`/`6a6fc54`: oracle 0% [uncontrollable at K=6, final 4.0870], latent Pearson −0.1911 / Spearman −0.1895, rollout 0.0125. Weak-reg K=24 `planning_diagnosis_k24_lowreg.json`: oracle 1.0, Pearson −0.0551 / Spearman −0.0979, rollout 0.0082. SIGReg K=24 `planning_diagnosis_k24_sigreg.json`: oracle 1.0, Pearson −0.2338 / Spearman −0.4068, `feat_std=0.5045`, rollout 0.6103, t=H div 0.8085.)

**The layered story:** Layer 1 — oracle proves gLV unreachable at K≤21, fully reachable at K=24 (success 100%, final 0.79). Layer 2 — after retraining at K=24, learned latent-MPPI still 0%; latent-distance-to-target essentially uncorrelated with true distance (Pearson −0.00, Spearman +0.02) though rollout faithful (~2% / 20 steps). Layer 3 — explicit z→x readout improves R² (0.78→0.89), tightens final distance (4.12→3.01, beating greedy), never reaches tol. **The gLV is controllable, dynamics learned faithfully, but the JEPA latent does not expose a plannable cost surface.**

**Figures:** `oracle_K_sweep.png` (controllability curve), `planning_diagnosis.png` (3-layer diagnosis), `planning_success_rate.png` (per-method success bars).

### 6.5 M3 levers — three targeted attacks (all NEGATIVE / diagnostic)

**Branches:** `m3-learned-cost`, `m3-model-fidelity`, `m3-multistep-rollout` (learned-cost JSONs mirrored on `bnz`). All commits BNZ94, 2026-06-20. None crossed `tol = 0.9957860399240593`. Substrate: **weak-reg (lowreg) world model** (`checkpoints/plan_model_k24_lowreg/latest.pth.tar`, decoder R² ~0.89). Config: `action_dim=24`, `d_model=128` (except d256 probe), regularizer `sim_coeff_t=4 / cov_coeff=1 / std_coeff=0.25`, MPPI `n_samples=128, n_elites=16, n_iters=3, temperature=1.0, init_std=0.25, min_std=0.02, cumulative=true`, 3 seeds [0,1,2], 12 episodes, 20 MPC steps.

**Lever 1 — Learned monotonic cost head** (`m3-learned-cost`; `plan_glv_learned.py`; commits `4268d4e`, `fc5d2d7`, `956dbc5`; results `planning_learned_{lowreg,d256,h2,h3,h4}.json`). `RankHead` (symmetric pair features → Softplus scalar) trained to rank candidate next-states by true gLV distance; MPPI cost = `Σ_t h(z_rolled_t, z_target)`. Head trained 3000 steps, hidden=256, on 6400 train states.

1a. Capacity sweep:

| Substrate | head Spearman (held-out) | decoder R² (lin/mlp) | mppi_learned final | best | success |
|---|---|---|---|---|---|
| weak-reg d128 (`planning_learned_lowreg.json`) | **0.7099** | 0.8425 / 0.8927 | **3.0605** | 2.8083 | 0% |
| weak-reg d256 (`planning_learned_d256.json`) | **0.8077** | 0.8791 / 0.8914 | **3.1460** | 2.7568 | 0% |

Full d128 baseline comparison (all 0%, start 6.6386): random 3.9922/3.5788; greedy 3.2581/3.2580; mppi_latent 4.5311/4.4157; mppi_decoded 3.1240/2.5571; **mppi_learned 3.0605/2.8083** (best final of any method, yet still 0%). Better cost-monotonicity (Spearman 0.71→0.81) does NOT close the loop (`4268d4e`, `fc5d2d7`); d512 judged not warranted.

1b. Horizon sweep (compounding-error test):

| MPPI horizon | mppi_learned final | mppi_decoded final | success |
|---|---|---|---|
| h2 (`planning_learned_h2.json`) | 3.4062 | 2.8233 | 0% |
| h3 (`planning_learned_h3.json`) | 3.2285 | 2.8375 | 0% |
| h4 (`planning_learned_h4.json`) | 3.1532 | 2.9992 | 0% |
| h6 (`planning_learned_lowreg.json`) | **3.0605** | 3.1240 | 0% |

Learned-cost final improves monotonically with horizon (3.41→3.23→3.15→3.06) — longer rollout is *better*, refuting the compounding-error hypothesis (`956dbc5`).

**Lever 2 — Model-fidelity push** (`m3-model-fidelity`; `m3_ensemble_gate.py`, `m3_onpolicy.py`; commits `34b6e86`, `02c4e10`, `6d582c3`; results `m3_ensemble_gate.json`, `m3_onpolicy.json`).

Step 1 — Epistemic gate (`m3_ensemble_gate.json`, n_models=5, 720 steps / 36 episodes): planner_success 0.0; corr(disagreement,error) Pearson 0.3770 / Spearman 0.3935; mean disagreement 0.011037; mean true 1-step error 0.264039; disagreement far/near 0.010982/0.011062; error far/near 0.263847/0.264504. Verdict: *"NOT epistemic exploitation"* — disagreement uniform; no exploitable pockets. Commit `34b6e86` adds the decisive number: predictor 1-step error **0.072 on training distribution (random-dose) vs 0.264 on planner OOD trajectories — 3.7× gap** → real wall is distribution shift.

Step 2 — On-policy DAgger/MBPO loop (`m3_onpolicy.json`, 4 rounds):

| round | n_train | train MSE | err on planner traj | success | mean final dist |
|---|---|---|---|---|---|
| 0 | 4608 | 4.683e-05 | 0.13783 | 0% | 3.4913 |
| 1 | 4848 | 4.910e-05 | 0.14133 | 0% | 4.0362 |
| 2 | 5088 | 5.112e-05 | 0.13553 | 0% | 4.2680 |
| 3 | 5328 | 6.019e-05 | 0.13523 | 0% | 4.4192 |

`final_success_rate=0.0`, `final_mean_dist=4.6985` (3 seeds), `crossed_tol=false`, baseline_learned_cost_final=3.06, oracle_final=0.79. Error flat ~0.135; final distance worsens (3.49→4.42). Fresh on-policy predictor has *lower* 1-step error (0.135) than original (0.264) yet plans *worse* (3.49 vs 3.06) → 1-step accuracy decoupled from planning quality (`02c4e10`/`6d582c3`).

**Lever 3 — Multi-step free-running rollout loss** (`m3-multistep-rollout`; `m3_multistep.py`; commits `048577a`, `732358d`; results `m3_multistep.json`). Retrain predictor with multi-step free-running objective (unroll k=6, feeding own predicted latents, scheduled sampling; encoder + learned cost frozen).

| quantity | 1-step-trained | multi-step-trained |
|---|---|---|
| held-out 1-step error | 0.23745 | **0.07241** |
| held-out free-running 6-step error | 0.61117 | **0.08607** |

`gate_bites_freerun_reduced=true`. Planning: `multistep_plan_success=0.0`, `multistep_plan_final=4.1255`, `multistep_plan_err_on_traj=0.17228`, `crossed_tol=false`. Baselines in-file: oracle 0.79, learned_cost_1step 3.06. Free-running 6-step error cut **0.611→0.086 (−86%)**, 1-step **0.237→0.072**, yet planning 0%. **Dynamics fidelity (1-step and multi-step) definitively ruled out** (`732358d`).

**Sharpened claim (commit `732358d`, M3 "8 levers"):** (1) cost-head quality not the wall (Spearman 0.71→0.81, flat 0%); (2) compounding error not the wall (longer horizon better); (3) optimizer exploitation not the wall (uniform disagreement 0.011); (4) distribution shift real (3.7× 0.072 vs 0.264) but not sufficient (on-policy err→0.135, planning worsens); (5) dynamics fidelity exonerated (−86% free-run, 0% success). Residual bottleneck: **frozen encoder's latent cost/representation precision near the target** (rank Spearman ~0.81, decode R² ~0.89 can't target to tol; oracle true-state cost reaches 0.79). Every M3 lever kept as an honest negative, never folded into `bnz` as a positive.

*Note: `bnz` also carries a related lever family (`m3_metric_gate.py`, `m3_recognition.py` with `m3_metric_gate{,_mc03,_mc10,_mc30}.json`, `m3_recognition.json`, `planning_learned_metric_mc{03,10,30}.json`) — covered in 6.6.*

### 6.6 HYBRID metric-loss (the one POSITIVE planning result)

**Branches:** `m3-metric-loss-hybrid` (full results), `bnz` (subset). Key commits: `8ccd1c3` (loss + scripts), `26f75b4` (positive result), `19387f5` (full-eval sweep + sim caveat), `283ca9e` (consolidated table). Training jobs **75773 / 75774 / 75775** (eval on CPU).

This turned the pure-JEPA M3 planning negative (0% success) into **100% planning success** by adding a **metric-preserving (isometry) auxiliary loss** to gLV world-model training, gated by `model.regularizer.metric_coeff` (default `0` ⇒ bit-identical to pure-JEPA `bnz` baseline). Explicitly **HYBRID, not pure JEPA** — uses ground-truth gLV-state supervision.

**The loss** (`m3-metric-loss-hybrid:examples/microbiome_jepa/train_worldmodel.py`, `isometry_metric_loss`, L79–99; wiring L207–276). For latents `z` and true gLV states `x` (loader's `batch[2]`, `[B, Tw, S]`), sample `n_pairs` (default **4096**) seeded pairs, penalize the gap between latent and true-state Euclidean distances up to one learned global scale `exp(metric_scale)`:

```
dz = ||z_i - z_j||,  dx = ||x_i - x_j||           # clamp_min(1e-12) before sqrt
loss += metric_coeff * mean( (dz - exp(metric_scale) * dx)^2 )
```

`metric_scale` is a single learnable parameter in the AdamW group. Active VICReg std/cov terms prevent collapse-to-a-point. CPU smoke (`8ccd1c3`): `metric_loss` 4.5→0.4, healthy feat_std.

**Diagnostic scripts:** `m3_metric_gate.py` (CPU gate: latent-vs-true distance Pearson+Spearman, to-target and pairwise; decode R²; feat_std; 1-step + free-running 6-step rollout error; recomputes pure-JEPA "before" via `--ref_checkpoint`); `m3_recognition.py` (frozen-latent logistic-regression probe on `dominant_guild` and `basin` labels; full-train acc, few-shot, majority baseline).

Substrate: K=24, `d_model=128`, weak-reg, IDM on, 80 epochs (`run_glv_k24_metric.sh`). Planning eval: 3 seeds × 12 episodes = 36, tol 0.9958, 20 MPC steps; `mppi_latent` = raw latent-distance MPPI (clean test); MPPI horizon 6, 128 samples, 16 elites, 3 iters, cumulative.

**Gate** (`m3_metric_gate.json`, `gate_pass: true`):

| quantity | pure JEPA (mc=0) | metric HYBRID (mc=0.3) | Δ |
|---|---|---|---|
| corr-to-target Spearman | **+0.0845** | **+0.9904** | +0.906 |
| corr-to-target Pearson | +0.1128 | +0.9771 | — |
| pairwise Spearman | +0.5160 | +0.9726 | — |
| pairwise Pearson | +0.5291 | +0.9827 | — |
| decode R² (linear) | 0.8433 | 0.9761 | — |
| decode R² (MLP) | 0.8927 | 0.9788 | +0.086 |
| feat_std | 0.00758 | 0.07236 | — |
| 1-step rollout err | 0.0727 | 0.2117 | — |
| free-run 6-step rollout err | 0.0843 | 0.2834 | +0.199 (worse) |

**Planning** (`planning_learned_metric_mc03.json` vs `planning_learned_lowreg.json`), 36 episodes, start 6.6386:

| method | pure JEPA (mc=0) success / final | metric HYBRID (mc=0.3) success / final |
|---|---|---|
| random | 0% / 3.992 | 0% / 3.994 |
| greedy | 0% / 3.258 | 0% / 3.264 |
| **mppi_latent (clean test)** | **0% / 4.531** | **100.0% ± 0.0 / 0.804** |
| mppi_decoded | 0% / 3.124 | 100.0% ± 0.0 / 0.844 |
| mppi_learned | 0% / 3.060 | 80.6% ± 7.3 / 1.024 |
| **ORACLE** (from `oracle_K_sweep.json`) | — | **100% / 0.790** |

Raw-latent planner reaches the target in **100% of 36 episodes**, final distance (0.804) ≈ oracle (0.790). Learned-cost head held-out Spearman 0.9954. **The single positive planning result.**

**`metric_coeff` sweep** (`metric_sweep_consolidated.json` + `.md`, `283ca9e`; oracle = K=24 row):

| metric_coeff | success (mppi_latent) | final | free-run 6-step err | latent↔true Spearman | recog dom-guild | recog basin |
|---|---|---|---|---|---|---|
| pure JEPA (mc=0) | **0.0%** | 4.531 | 0.0843 | +0.0845 | 0.899 | 0.690 |
| HYBRID mc=0.3 | **100.0% ± 0.0** | 0.804 | 0.2834 | +0.9904 | **0.971** | **0.812** |
| HYBRID mc=1.0 | 97.2% ± 2.8 | 0.840 | 0.3646 | +0.9918 | 0.967 | 0.771 |
| HYBRID mc=3.0 | 91.7% ± 4.8 | 0.866 | 0.4148 | +0.9906 | 0.970 | 0.779 |
| ORACLE | 100% | 0.790 | — | — | — | — |

**Mechanism:** cost geometry (Spearman) saturates ~0.99 for every mc>0 — not why higher coeff hurts. Success erodes 100.0%→97.2%→91.7%, tracking rising free-running rollout error (0.283→0.365→0.415). Binding tension: **metric-fidelity vs rollout-predictability**; mc=0.3 is the sweet spot on all three axes. (`mppi_learned` falls to 38.9%±7.3 at mc=1.0 and 50.0%±8.3 at mc=3.0; `mppi_decoded` non-monotonic 100%→75%→97.2%.)

**Recognition tradeoff — REFUTED on gLV** (`m3_recognition.json`, `_mc10.json`, `_mc30.json`):

| task | mc=0 acc_full | mc=0.3 | mc=1.0 | mc=3.0 | majority |
|---|---|---|---|---|---|
| dominant_guild (n_test=1280) | 0.8992 | **0.9711** | 0.9672 | 0.9695 | 0.35 |
| basin (n_test=480) | 0.6896 | **0.8125** | 0.7708 | 0.7792 | 0.3667 |

Recognition *improves* with the metric (peaks at mc=0.3: dom-guild +0.072, basin +0.123 over pure-JEPA; basin few-shot 0.475→0.658). On gLV the abstract labels are functions of the state aligned with the true metric — sim-specific, need not hold on real data (would need Bray-Curtis; future work).

**Honest caveat (load-bearing, `meta.hybrid_caveat`):** `metric_coeff>0` uses TRUE gLV-state distances as supervision ⇒ HYBRID, not pure JEPA. The pure-JEPA M3 negative (0% planning) remains the project's primary M3 result; the metric-HYBRID positive is presented alongside, clearly labelled; it is sim-specific (depends on a ground-truth state metric).

**Figure:** `m3-metric-loss-hybrid:examples/microbiome_jepa/results/metric_hybrid.png` (built by `make_metric_figure.py`, suptitle flags "uses TRUE-state supervision — NOT pure JEPA"). Left: planning success by method (M3 negative all 0% vs HYBRID raw-latent 100%, oracle=1.00 line). Right: `metric_coeff` tradeoff (success + Spearman saturating ~0.99 vs rising free-run rollout error; mc=0.3 sweet spot).

### 6.7 Bonus experiments (Exp1 / Exp2 / Exp3) — robustness & falsification of the metric closure

**Branches:** `m3-generalization` (Exp1), `m3-idm-selfsup` (Exp2), `m3-bottleneck` (Exp3). Session 2026-06-20 (GB200 train + CPU eval). All MEASURED, seeded. `bnz` not modified this session. Consolidated write-up: `examples/microbiome_jepa/results/BONUS_SESSION_RESULTS.md` (commit `047a9a9`, on `m3-generalization`). Common substrate: K=full-actuation, `d_model=128` (unless swept), weak-reg, IDM on, 80 epochs; planning eval 3 seeds × 12 episodes, tol = 0.15 × mean inter-attractor distance; `mppi_latent` = clean raw-latent test.

**Exp 1 — Generalization across gLV instances (POSITIVE).** Branch `m3-generalization`; commits `bbda923`, `94f5e43`, `c021bee`, `93ab649`; code `screen_instances.py`, `eval_generalization.py`; results `exp1_instance_screen.json`, `exp1_generalization.{json,md}`. Honesty note: the headline result's 3 "seeds" vary only planning eval on one fixed gLV; new instances require varying structural knobs (`n_guilds`, `comp_strong`, `comp_weak`, `n_species`, `growth`).

Pre-screen (`exp1_instance_screen.json`) — validity = stable ∧ distinct ∧ n_attr≥2 ∧ greedy<0.5 ∧ oracle≥0.99:

| instance | knobs (S/guilds/comp_strong/comp_weak/growth) | K | n_attr | stable (max eig) | min inter-attr dist | tol | greedy | oracle | oracle final | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_g3_s24 | 24/3/−2.5/−0.4/1.0 | 24 | 3 | yes (−0.9889) | 6.6386 | 0.9958 | 0.0 | 1.0 | 0.7785 | yes |
| g4_s24 | 24/4/−2.5/−0.4/1.0 | 24 | 4 | yes (−0.9846) | 5.7431 | 0.8615 | 0.3333 | 1.0 | 0.7223 | yes |
| g3_s18 | 18/3/−2.5/−0.4/1.0 | 18 | 3 | yes (−0.9859) | 5.7467 | 0.8620 | 0.3889 | 1.0 | 0.6191 | yes |
| g5_s30 | 30/5/−2.5/−0.4/1.0 | 30 | 5 | yes (−0.9833) | 5.7394 | 0.8610 | 0.2222 | 1.0 | 0.6600 | yes |
| g3_s24_strongcomp | 24/3/−3.5/−0.25/1.0 | 24 | 3 | yes (−0.9589) | 6.5882 | 0.9882 | 0.3889 | 1.0 | 0.7984 | yes |
| g3_s32_fastgrow | 32/3/−2.5/−0.4/**1.5** | 32 | 3 | yes (−1.4921) | 11.1000 | 1.6954 | 0.0 | 0.9444 | 1.5185 | **no (rejected, oracle<0.99)** |

Matched eval (`exp1_generalization.json`), HYBRID mc=0.3 per instance, 3 seeds × 12 episodes:

| instance | guilds/S/K | tol | random | greedy | oracle | mppi_latent succ ± SE / final | crosses tol | near-oracle |
|---|---|---|---|---|---|---|---|---|
| g3_s18 | 3/18/18 | 0.862 | 0%/3.323 | 41.7%/2.654 | 100%/0.610 | **100% ± 0.0 / 0.738** | yes | yes |
| g5_s30 | 5/30/30 | 0.861 | 0%/3.728 | 25.0% ± 4.8/3.025 | 100%/0.671 | **100% ± 0.0 / 0.654** | yes | yes |
| g4_s24 | 4/24/24 | 0.861 | 0%/3.607 | 33.3% ± 4.8/2.812 | 100%/0.648 | 88.9% ± 7.3 / 0.778 | yes | no |
| g3_s24_strongcomp | 3/24/24 | 0.988 | 0%/4.007 | 41.7%/2.769 | 100%/0.790 | 88.9% ± 7.3 / 0.953 | yes | no |

Closure robust: **4/4** new instances cross tol (pure JEPA 0%); **2/4** match oracle at 100%, **2/4** reach 88.9% (32/36 episodes) near oracle. Not cherry-picked.

**Exp 2 — IDM-reweight self-supervised closure (NEGATIVE — sharpens claim).** Branch `m3-idm-selfsup`; commits `4305751`, `8a111d7`, `0844b1e`; code `eval_pure_variants.py`, `run_glv_idm.sh`; results `exp2_idm.{json,md}`. At `metric_coeff=0`, sweep `idm_coeff ∈ {2,5,10}`.

| variant | d | mppi_latent succ/final | decoded | learned | latent↔true Spearman | free-run 6-step | one-step | decode R²(MLP) | feat_std | recog guild | recog basin | head Spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pure-JEPA idm=1.0 (ref) | 128 | 0%/4.531 | — | — | +0.0845 | 0.0843 | — | — | — | 0.8992 | 0.6896 | — |
| HYBRID mc=0.3 (upper bar) | 128 | 100%/0.804 | — | — | +0.9904 | 0.2834 | — | — | — | 0.9711 | 0.8125 | — |
| idm=2 | 128 | 0%/4.325 | 0% | 0% | **−0.0167** | 0.0844 | 0.0738 | 0.9027 | 0.00764 | 0.8961 | 0.7313 | 0.5328 |
| idm=5 | 128 | 0%/4.263 | 11.1% | 0% | **−0.1962** | 0.0961 | 0.0798 | 0.9269 | 0.00915 | 0.9008 | 0.7250 | 0.7328 |
| idm=10 | 128 | 0%/4.381 | 0% | 0% | **−0.3127** | 0.0856 | 0.0769 | 0.8893 | 0.00755 | 0.8875 | 0.6958 | 0.7392 |

Raw-latent MPPI 0% at every weight; latent↔true Spearman goes *more negative* (+0.085→−0.313). Learned-cost head Spearman climbs (0.53→0.74) — IDM induces a *control* metric but not the Euclidean state metric. **Privileged true-state metric is necessary.**

**Exp 3 — Bottleneck shrink (NEGATIVE).** Branch `m3-bottleneck`; commits `10627d1`, `aafdaaf`; code `eval_pure_variants.py --which dim`, `run_glv_dim.sh`; results `exp3_dim.{json,md}`. At `metric_coeff=0`, sweep `d_model ∈ {16,24,32}` toward true-state dim S=24.

| variant | d | mppi_latent succ/final | decoded | learned | latent↔true Spearman | free-run 6-step | one-step | decode R²(MLP) | feat_std | recog guild | recog basin | head Spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pure-JEPA idm=1.0 (ref) | 128 | 0%/4.531 | — | — | +0.0845 | 0.0843 | — | — | — | 0.8992 | 0.6896 | — |
| HYBRID mc=0.3 (upper bar) | 128 | 100%/0.804 | — | — | +0.9904 | 0.2834 | — | — | — | 0.9711 | 0.8125 | — |
| d=16 | 16 | 0%/4.823 | 0% | 0% | +0.2536 | 0.0412 | 0.0352 | 0.6597 | 0.01013 | 0.6719 | 0.5750 | 0.4316 |
| d=24 | 24 | 0%/5.092 | 0% | 0% | +0.1972 | 0.0496 | 0.0431 | 0.7211 | 0.00957 | 0.6758 | 0.5333 | 0.4464 |
| d=32 | 32 | 0%/4.235 | 0% | 0% | +0.2373 | 0.0573 | 0.0525 | 0.7800 | 0.00923 | 0.7469 | 0.5813 | 0.5473 |

Raw-latent MPPI 0% at every dim; marginal Spearman nudge (~0.20–0.25), costs recognition (guild 0.90→0.67–0.75, basin 0.69→0.53–0.58) and decodability (R² down to 0.66–0.78). **The metric must be SUPPLIED, not coaxed out by bottlenecking.**

**Cross-experiment synthesis:** Exp1 POSITIVE (closure generalizes, not toy-specific); Exp2 NEGATIVE (self-supervised IDM gives control metric, not state metric); Exp3 NEGATIVE (bottleneck gives marginal nudge, costs recognition). Net: **the privileged true-state metric (isometry auxiliary) is necessary**, robust across gLV systems.

---

## Section 7 — Sequencing-technology invariance

**Branch:** `m2-tech-invariance` (off `sigreg-rep`; not pushed to origin, not folded into `bnz`; gLV/M3/M4 untouched). Consolidated report: `/Users/bnz/DynaAMIcs/examples/microbiome_jepa/REPORT_tech_invariance.md`.

### 7.0 What it tests
After M2 found that the trained set-JEPA embedding is *more* separable by sequencing technology (amplicon-16S vs WGS-shotgun) than raw input, this follow-up tries to *fix* it. Dual-axis tradeoff: drive **technology** (nuisance, binary, chance 0.50 balanced) DOWN while holding **biology** (8-class `biome`) UP. Data **unpaired** (each sample one library strategy) ⇒ adversarial domain-invariance (DANN) + distribution alignment (CORAL/MMD), not paired alignment. Evaluation rule: frozen encoder; both axes scored by a *fresh* post-hoc 5-fold probe (`StratifiedKFold(n_splits=5)`, `LogisticRegression` + `MLPClassifier(hidden=(128,))`) — never the training adversary. Headline metric = balanced accuracy.

### 7.1 Code inventory

| File (`m2-tech-invariance`) | Role |
|---|---|
| `examples/microbiome_jepa/tech_invariance.py` | Eval harness: RunID→Terms join, balanced corpus streaming, fresh 5-fold linear+MLP balanced-acc probe, Susagi imposter-rep baseline |
| `examples/microbiome_jepa/tech_sweep.py` | Sweep orchestrator: `loss.invariance_coeff` × seeds × {vicreg, bcs}, plumbs `--loss.invariance_method {dann,coral,mmd}` |
| `examples/microbiome_jepa/tech_eval_batch.py` | Batched parallel fresh-probe eval (commits `3817926`, `27bcfdf`) |
| `examples/microbiome_jepa/main.py` | Integration owner: `coral_loss` (L78–92), `mmd_rbf_loss` (L95–101), DANN/CORAL/MMD train-loop (L198–304) |
| `eb_jepa/architectures.py` | `_GradientReversalFn` (660), `grad_reverse` (679), `TechAdversaryHead` (683), `dann_lambda` (714) |
| `examples/microbiome_jepa/plot_tech_tradeoff.py` | Dual-axis tradeoff figure |
| `examples/microbiome_jepa/plot_latent_umap.py` | UMAP/PCA latent projections; layouts `{compare, tech_split, tech_within_biome}` |

**Methods.** (1) **DANN** (`7ce355e`): `TechAdversaryHead` behind a gradient-reversal layer (`backward: grad_output.neg() * lambd`); strength `loss.invariance_coeff`; GRL ramp `dann_lambda(p) = 2/(1+e^{−γp}) − 1`, γ=10. `invariance_coeff=0` ⇒ byte-identical baseline. (2) **CORAL** (`a2164ab`): `coral_loss(fa,fb)` aligns per-tech mean+cov (`cov_term = ‖cov_a−cov_b‖²_F/(4d²)` + `mean_term = ‖μ_a−μ_b‖²/d`); deterministic, no min-max. (3) **RBF-MMD** (`a2164ab`): `mmd_rbf_loss` multi-bandwidth (σ ∈ {2,5,10,20}); coded as `invariance_method=mmd` but **NOT swept** (no committed MMD numbers).

### 7.2 Step 0 — Confounding pre-screen (the ceiling) — MEASURED, job 76596
Source `results/tech_confounding.json` (commit `55ae496`). n = 5000 balanced corpus samples (4102 with biome label).

| Quantity | Value (exact) | Reading |
|---|---|---|
| NMI(tech ; biome) | **0.09545563191878492** | mostly independent |
| biome from tech (bal-acc) | **0.1392857142857143** ± 0.0035714285714285696 (chance 0.125) | tech ≈ says nothing about biology |
| biome from tech (raw-acc) | 0.5287665250586733 ± 0.00023016699613778315 | |
| **tech from biome (bal-acc)** | **0.6915087420217638** ± 0.00824982258170542 (chance 0.500) | biology partially predicts tech → floor |
| tech from biome (raw-acc) | 0.7006304031371616 ± 0.007831966645730853 | |

Cross-tab (strat × biome, exact counts):

| | gut | water | soil | plant | feces | skin | oral | stool |
|---|----|------|-----|------|------|-----|-----|------|
| **amplicon** | 696 | 619 | 295 | 148 | 123 | 13 | 10 | 1 |
| **wgs** | 1473 | 292 | 98 | 6 | 190 | 138 | 0 | 0 |

NMI ≈ 0.095, tech→biome ≈ chance ⇒ most tech signal is biome-independent "pure protocol" nuisance. But biome→tech ≈ **0.69 balanced** sets a **floor ≈ 0.69** below which removing tech must cost biology. The 0.69 is an **ESTIMATE** (8-biome one-hot probe, a lower bound on richer-biology leakage).

### 7.3 Before baseline + Susagi imposter-rep baseline — MEASURED
Frozen encoder, fresh 5-fold probe, n = 4960 (tech) / 4062 (biome). Sources: `tech_invariance.json` (VICReg, `6a82f96`), `tech_invariance_sigreg.json` (SIGReg, `18691f8`).

| Representation | TECH bal-acc (↓) | TECH s.e. | BIOME bal-acc (↑) | BIOME s.e. |
|---|---|---|---|---|
| **VICReg JEPA** | **0.9524291199262258** | 0.0022310646168908783 | 0.8050112727543036 | 0.03202941413495098 |
| **SIGReg JEPA** | **0.9669294520900836** | 0.002178830945784939 | 0.8163233192772777 | 0.020582534229092952 |
| raw mean-pool | **0.9380121152544374** | 0.0049277268823316065 | 0.7937392661216569 | 0.023639362681230625 |
| random-init encoder | 0.896899370964426 | 0.0013938798215780911 | 0.741043053792634 | 0.036447164309001204 |
| **Susagi imposter rep** | **0.8913904853854216** | 0.002973818900035744 | 0.7191905283903053 | 0.021550739369108632 |

(Raw-accuracy variants also in JSONs: VICReg tech raw-acc 0.9524193548387097, Susagi tech raw-acc 0.8913306451612903. Chance tech = 0.5038306451612903; chance biome = 0.5324963072378139 raw / 0.125 balanced.)

Trained JEPA is the **most** tech-separable: SIGReg 0.967 > VICReg 0.952 > raw 0.938 > random 0.897 > Susagi 0.891. Susagi imposter-rep (mean-pooled Susagi encoder hidden state) is least tech-separable but also lowest biology (0.719).

### 7.4 Result 1 — DANN tradeoff sweep — NEGATIVE — MEASURED (train 76698/76925/77350; eval 77533; 6 seeds)
Sweep `invariance_coeff ∈ {0,0.3,1.0,3.0,10.0}` × seeds {0..5} × {VICReg, SIGReg}. Source `results/tech_sweep_results.json` (commit `1bfd066`; `d_model=128, per_class_cap=2500, n=4960`). Matched re-baseline raw mean-pool: tech 0.9380121152544374 / biome 0.7927587843232028.

**VICReg**
| coeff | TECH lin (↓) | BIOME lin | TECH MLP | BIOME MLP | n |
|--:|:--:|:--:|:--:|:--:|:--:|
| 0.0 | 0.9603 ± 0.0008 | 0.8295 ± 0.0044 | 0.9791 ± 0.0004 | 0.8331 ± 0.0062 | 6 |
| 0.3 | 0.9513 ± 0.0020 | 0.8197 ± 0.0037 | 0.9764 ± 0.0010 | 0.8338 ± 0.0056 | 6 |
| 1.0 | 0.9539 ± 0.0016 | 0.8030 ± 0.0099 | 0.9741 ± 0.0014 | 0.8211 ± 0.0061 | 6 |
| 3.0 | 0.9498 ± 0.0016 | 0.7914 ± 0.0079 | 0.9722 ± 0.0009 | 0.8124 ± 0.0029 | 6 |
| 10.0 | 0.9223 ± 0.0028 | 0.6896 ± 0.0100 | 0.9566 ± 0.0025 | 0.7497 ± 0.0045 | 6 |

**SIGReg (BCS)**
| coeff | TECH lin (↓) | BIOME lin | TECH MLP | BIOME MLP | n |
|--:|:--:|:--:|:--:|:--:|:--:|
| 0.0 | 0.9339 ± 0.0016 | 0.7670 ± 0.0037 | 0.9698 ± 0.0009 | 0.8065 ± 0.0059 | 6 |
| 0.3 | 0.9354 ± 0.0014 | 0.7554 ± 0.0054 | 0.9655 ± 0.0009 | 0.7959 ± 0.0042 | 6 |
| 1.0 | 0.9236 ± 0.0042 | 0.7473 ± 0.0085 | 0.9607 ± 0.0013 | 0.7757 ± 0.0085 | **5** |
| 3.0 | 0.9063 ± 0.0062 | 0.7119 ± 0.0068 | 0.9489 ± 0.0025 | 0.7599 ± 0.0043 | 6 |
| 10.0 | 0.9011 ± 0.0091 | 0.6740 ± 0.0100 | 0.9390 ± 0.0043 | 0.7391 ± 0.0131 | 6 |

(One `bcs c1.0` seed failed → 59/60 checkpoints.)

**Verdict — NEGATIVE.** Technology stays strongly recoverable across the *entire* sweep (linear tech never <0.90, never near chance 0.50, never reaching the 0.69 floor) while biology is destroyed (biome to/below 0.69 at high coeff). From coeff 0→10: VICReg loses only −0.038 tech but −0.139 biome; SIGReg −0.033 tech / −0.093 biome — invariance pressure removes **~3× more biology than technology**. No sweet spot. MLP probe same trend (tech 0.94–0.98). Diagnosis: tech is the dominant, linearly-accessible axis; the adversary's discriminator accuracy collapsed below chance (encoder fools a one-step discriminator *without* removing info); mild confounding makes cheapest removable directions partly biological.

### 7.5 Result 2 — CORAL/MMD deterministic alignment — PARTIAL POSITIVE — MEASURED (train 77650 / eval 77777; 3 seeds)
Invariance term = CORAL (`invariance_method=coral`), coeffs {0,10,100,1000}. Source `results/tech_sweep_coral_results.json` (commits `c19505b`, `6eaa7c7`; `d_model=128, per_class_cap=2500, n=4960`; 24/24 checkpoints, n=3).

**VICReg + CORAL**
| coeff | TECH lin (↓) | BIOME lin | TECH MLP | BIOME MLP |
|--:|:--:|:--:|:--:|:--:|
| 0 | 0.9596 ± 0.0005 | 0.8368 ± 0.0038 | 0.9794 ± 0.0006 | 0.8300 ± 0.0031 |
| 10 | 0.9122 ± 0.0066 | 0.8191 ± 0.0105 | 0.9769 ± 0.0005 | 0.8376 ± 0.0051 |
| 100 | 0.8697 ± 0.0055 | 0.7882 ± 0.0107 | 0.9694 ± 0.0028 | 0.8308 ± 0.0113 |
| 1000 | **0.8466 ± 0.0028** | 0.7720 ± 0.0069 | 0.9588 ± 0.0010 | 0.7900 ± 0.0097 |

→ −0.113 tech for −0.065 biome (~1.7:1 favourable).

**SIGReg + CORAL**
| coeff | TECH lin (↓) | BIOME lin | TECH MLP | BIOME MLP |
|--:|:--:|:--:|:--:|:--:|
| 0 | 0.9344 ± 0.0020 | 0.7697 ± 0.0027 | 0.9691 ± 0.0009 | 0.8074 ± 0.0126 |
| 10 | 0.9162 ± 0.0027 | 0.7668 ± 0.0098 | 0.9682 ± 0.0031 | 0.8058 ± 0.0094 |
| 100 | 0.8827 ± 0.0020 | 0.7654 ± 0.0064 | 0.9653 ± 0.0022 | 0.8152 ± 0.0085 |
| 1000 | **0.8441 ± 0.0033** | **0.7681 ± 0.0032** | 0.9562 ± 0.0008 | 0.7886 ± 0.0090 |

→ **−0.090 tech for ≈0 biome** (0.9344→0.8441 vs 0.7697→0.7681, within s.e.). The **sweet spot**: ~9 points of protocol signal removed at ≈zero biology cost — what DANN could not do.

**Verdict — PARTIAL POSITIVE.** CORAL removes tech DANN couldn't *and keeps biology*; tradeoff runs the right way. Two honest limits: (1) linear tech bottoms at ~0.84, still above the 0.69 floor and chance 0.50 — reduced, not eliminated; (2) nonlinear MLP probe barely moves (tech-MLP ~0.96 across all coeffs). CORAL aligns only first two moments. Full invariance would need higher-order alignment (RBF-MMD, coded not swept) or conditioning; Harmony/scVI are domain-standard references. **JEPA lesson: for a dominant, linearly-accessible nuisance, alignment >> adversarial.**

### 7.6 UMAP/PCA figures + within-biome correction
`tech_*_coords.json` store raw UMAP/PCA coordinates + labels: `tech_latent_umap_coords.json` 3 panels (`raw input`, `JEPA baseline coeff=0`, `JEPA + DANN`), n_plotted = 2989; `tech_split_umap_coords.json` 2 panels. **Honest correction (commit `1044f20`, MEASURED):** within-biome fresh tech probe (DANN coeff=10 seed0): gut 0.99→0.98, water 0.97→0.96, soil 0.93→0.95 (baseline→DANN) — DANN barely moves within-biome tech recoverability; the 2-D UMAP "mixing" was a **projection artifact**, corroborating the linear-probe negative.

Figures (each `results/<name>.png`): `tech_latent_pca.png` (PCA, `47c7c18`); `tech_latent_umap.png` (UMAP 3 panels, `9af4162`); `tech_split_umap.png` (per-tech biome layout, `1d735e8`/`9d066b0`); `tech_within_biome_umap.png` (within-biome tech bal-acc, `1044f20`); `tech_tradeoff.png` (DANN dual-axis, NEGATIVE, `1bfd066`); `tech_tradeoff_coral.png` (CORAL dual-axis, PARTIAL POSITIVE, `c19505b`).

### 7.7 Overall honest verdict
- **Confounding ceiling:** NMI 0.095; tech→biome ≈ chance; biome→tech ≈ **0.69 balanced** floor (job 76596).
- **Negative being fixed:** trained JEPA most tech-separable (SIGReg 0.967 > VICReg 0.952 > raw 0.938 > random 0.897 > Susagi 0.891).
- **DANN: FAILED** — tech never <0.90, biome lost ~3× faster, no sweet spot (job 77533, 6 seeds).
- **CORAL: PARTIAL FIX** — linear tech-leak removed nearly free (SIGReg+CORAL −0.090 tech / −0.002 biome at c1000), but only linearly (MLP probe still ~0.96, linear bottoms ~0.84 > floor 0.69) (jobs 77650/77777, 3 seeds).
- **RBF-MMD:** implemented, not swept — no committed numbers.

> **(NOTE: sources disagree — verify)** The "before baseline" VICReg JEPA tech value differs by source: `tech_invariance.json` (single corpus draw) = **0.9524**; the DANN sweep coeff=0 baseline (different balanced draw / 6-seed avg) = **0.9603**; SIGReg = **0.9669** (`tech_invariance_sigreg.json`) vs sweep coeff=0 **0.9339**. Both are MEASURED on different corpus draws/seed counts; not a contradiction but report with their provenance.

> **(NOTE: sources disagree — verify)** The consolidated REPORT.md/README.md on `bnz` (written before CORAL) carry the tech-invariance verdict as the **honest negative** (no CORAL). The CORAL "partial fix" lives on the current `m2-tech-invariance` branch (commits `c19505b`, `6eaa7c7`, `tech_sweep_coral_results.json`). Reconcile which verdict the published report uses.

---

## Section 8 — Tahoe perturbation probe

**Branch:** `tahoe-perturbation-probe` (single commit, never folded into `bnz`/`main`). Commit `8113894` (BNZ94, 2026-06-20; 15 files, +2086 lines). Code (`examples/tahoe_probe/`): `model.py`, `data.py`, `eval_probe.py`, `train_probe.py`, `run_ablation_tahoe.py`, `idm_coeff_sweep.py`, `build_centroids.py`, `render_figure.py`, `tune_lr.py`, `_make_synth.py`, `README.md`, `RESULTS.md`. Results: `examples/tahoe_probe/results/ablation/ablation_results.json` (938 lines), `ablation_figure.png`.

### 8.1 What it is
A **real-data generality check** for the *components* of the gLV Layer-B world model (action-conditioned latent predictor + IDM) on single-cell **drug** perturbations, using frozen 2560-d `mosaicfm-3b-prod-cont-MFMv2` embeddings from `tahoebio/Tahoe-x1-embeddings` (no encoder trained). **Not** a new perturbation-response model; does not compete with CPA/GEARS/scGPT. One-step, population/centroid level: control `z_control` = same cell line's `DMSO_TF` vehicle centroid; target `z_treated` = treated centroid. `GRUPredictor` mirrors `RNNPredictor` (action=GRU input, z_control=GRU hidden, update-gate bias high = no-change prior); `IDMClassifier` mirrors `InverseDynamicsModel` but is a classifier (CE over drug identity). Objective: `MSE(z_pred, z_treated) + idm_coeff · CE(idm(z_control, z_pred), drug)`, `idm_coeff ∈ {1.0 (on), 0.0 (off)}`. VICReg dropped (supervised target on frozen latent, no collapse). IDM reads predictor output `z_pred` — explicitly **not an M4 replication**.

### 8.2 Data coverage — MEASURED (`coverage` block, lines 14–41), from `emb0.parquet` (Tahoe-x1 shard-0)

| quantity | value |
|---|---|
| cells | 1,494,131 |
| cell lines (`n_cell_lines`) | 50 |
| drugs, non-control (`n_drugs_total`) | 95 |
| treated pairs (`n_treated_pairs`) | 4,198 |
| control `DMSO_TF` pops | 48 |
| pairs dropped <50 cells | 500 |
| control cells/pop (min/median) | 50 / 638 |
| treated cells/pop (min/median) | 50 / 312 |
| embedding dim | 2560 |
| min cells per pop | 50 |

Config: 5 seeds [0,1,2,3,4], splits ["pairs","celllines"], 300 epochs. Test sizes: `pairs` n_test=840 each seed; `celllines` n_test ∈ {755,844,840,843,755}.

### 8.3 The "absolute-z trap"
Absolute-z metrics high for everything: `no_op.R2_abs ≈ 0.919–0.925`, `no_op.cos_abs ≈ 0.965` vs model `R2_abs ≈ 0.93–0.95`. Drug effect ~29% of state magnitude. All conclusions use the **shift** Δ = z_treated − z_control vs three baselines: **no-op** (Δ̂=0), **global mean-shift**, **PRIMARY per-drug mean-shift** (mean train Δ for that drug over *other* cell lines). `no_op.cos_shift` = NaN by construction.

### 8.4 Result 1 — Held-out PAIRS (primary): predictor WITHOUT IDM beats every baseline (MEASURED, 5 seeds, `summary.pairs` L685–809)

| method | R²(shift) | cos(shift) | drug decodable from predicted shift (top-1) |
|---|---|---|---|
| **model, idm_off** | **+0.20120 ± 0.00359** | **+0.56810 ± 0.00140** | **0.45476 ± 0.00803** |
| model, idm_on | +0.01688 ± 0.00218 | +0.38443 ± 0.00203 | 0.00714 ± 0.00258 |
| per-drug mean-shift (primary) | +0.14853 ± 0.00380 | +0.47949 ± 0.00051 | — |
| global mean-shift | −0.00130 ± 0.00008 | +0.34642 ± 0.00182 | — |
| no-op | −0.12523 ± 0.00176 | NaN | — |

Predictor with no IDM beats per-drug baseline (0.201 vs 0.149, ~13 s.e. apart) on R² and cosine → captures cell-line-specific modulation. Per-seed idm_off R²(shift): 0.2002, 0.2007, 0.1954, 0.1949, 0.2148. **POSITIVE.**

### 8.5 Result 2 — IDM ablation: on real data IDM HURTS (the gLV-M4 cross-check)
Adding IDM collapses the predictor: R²(shift) **0.20120 → 0.01688**, drug-decodability of predicted shift **0.45476 → 0.00714** top-1 (≈ chance). Opposite of gLV M4. Per-seed idm_on R²(shift): 0.0189, 0.0089, 0.0164, 0.0183, 0.0218. **Diagnostic / negative-for-IDM that sharpens the gLV claim:** M4 IDM benefit was collapse-specific; here the frozen encoder has no collapse to fight, so IDM is an unnecessary auxiliary competing with the objective.

`idm_coeff` sensitivity (`RESULTS.md`, 3 seeds, held-out pairs):

| idm_coeff | 0.0 | 0.01 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|
| R²(shift) | +0.199 ± 0.002 | +0.155 ± 0.003 | +0.103 ± 0.003 | +0.071 ± 0.003 | +0.015 ± 0.003 |
| drug decodable (top-1) | 0.450 | 0.141 | 0.002 | 0.001 | 0.010 |

(per-drug baseline for this 3-seed subset = +0.153.) Monotonic degradation.

> **Integrity flag (provenance):** these `idm_coeff`-sweep numbers appear in `RESULTS.md` attributed to `results/coeff_sweep.log`, which is **NOT committed** (`git ls-tree` shows only `ablation_figure.png` + `ablation_results.json`; only the *script* `idm_coeff_sweep.py` is committed). Endpoints (idm_coeff 0.0 → R² 0.201 and 1.0 → R² 0.017) ARE consistent with the committed 5-seed JSON; intermediate points cannot be re-verified from the repo.

### 8.6 Result 3 — Held-out CELL LINES (secondary): does NOT transfer (MEASURED, 5 seeds, `summary.celllines` L811–936)

| method | R²(shift) | cos(shift) |
|---|---|---|
| model, idm_on | −0.14894 ± 0.00481 | +0.08946 ± 0.00386 |
| model, idm_off | −0.20813 ± 0.02012 | +0.10168 ± 0.00298 |
| per-drug mean-shift | **+0.13330 ± 0.01106** | +0.48295 ± 0.00597 |
| global mean-shift | −0.04385 ± 0.00384 | +0.32721 ± 0.00500 |
| no-op | −0.15608 ± 0.00518 | NaN |

For unseen cell lines, both model arms at/below no-op (idm_off −0.208, idm_on −0.149, vs no-op −0.156); per-drug mean-shift stays positive (+0.133). **NEGATIVE / honest limitation:** predictor cannot place a novel baseline state; does not generalize to unseen cell lines (nor unseen drugs — SMILES not attempted).

### 8.7 Result 4 — Action information IS in the real embedding (data property)
Linear probe recovering drug identity from the TRUE latent shift (`decode_true_shift`):
- Held-out pairs: top-1 **0.75262 ± 0.00587**, top-5 **0.95214 ± 0.00278** (chance ≈ 0.020); per-seed top-1 {0.7667, 0.7310, 0.7536, 0.7560, 0.7560}.
- Held-out cell lines: top-1 **0.84132 ± 0.01428**, top-5 **0.97939 ± 0.00119** (chance ≈ 0.011); per-seed {0.8331, 0.8294, 0.8071, 0.8932, 0.8437}.

**Positive premise-check:** the latent transition strongly encodes the intervention on real data; the predictor's failure to carry it under IDM (Result 2) is a training-objective effect, not absence of signal.

### 8.8 Figure
`examples/tahoe_probe/results/ablation/ablation_figure.png` — two-panel grouped bar chart, *"Tahoe drug-perturbation probe — shift-space R² vs baselines (5 seeds, 300 ep)"*. Left **split: pairs**, right **split: celllines**. R²(shift) with s.e. for four groups (model, per-drug, global mean-shift, no-op), two bars each (idm_on green / idm_off red). Pairs: red idm_off model above per-drug, green idm_on near zero. Celllines: both model bars negative (at/below no-op), only per-drug positive.

**Takeaway:** the gLV action-conditioning recipe partially generalizes — predictor (no IDM) beats per-drug baseline on held-out pairs (R²_shift 0.201 vs 0.149) by learning cell-line-specific modulation, drug identity strongly decodable from true shift (75–84% top-1) — but IDM HURTS (0.201→0.017, monotonic), the *opposite* of gLV M4 (sharpens the collapse-specific story), and no transfer to unseen cell lines (both arms ≤ no-op).

---

## Section 9 — Complete results index

Master table of every committed results JSON and figure. Paths under `examples/microbiome_jepa/results/` unless noted.

| File path | What it is | Key measured numbers | Branch |
|---|---|---|---|
| `results/ablation_collapse.json` | M4 IDM ablation, induce-collapse regime | fast_r2_action on 0.7483±0.0512 / off 0.5197±0.0207 (Δ+0.229) | bnz |
| `results/ablation_collapse.png` | M4 collapse-regime bar chart | — | bnz |
| `results/ablation_default.json` | M4 ablation, default VICReg regime | fast_r2_action on 0.3641±0.0199 / off 0.2910±0.0409 (Δ+0.073, seed2 reversed) | bnz |
| `results/ablation_default.png` | M4 default-regime bar chart | — | bnz |
| `results/oracle_K_sweep.json` | Oracle controllability vs K | K24 success 1.000±0.000 final 0.7901±0.0164; K6 final 4.0889 | bnz |
| `results/oracle_K_sweep.png` | Controllability curve | — | bnz |
| `results/planning_results.json` | First latent-MPPI (K=6) | all methods 0%; mppi final 4.8814 | bnz |
| `results/planning_decoded.json` | Stale pre-fix decoded (stray, `6a6fc54`) | mppi_decoded_mlp 0.0278 [0.0833,0,0] (RETRACTED fluke) | bnz |
| `results/planning_decoded_default.json` | Seeded decoded, default-reg K24 | R² 0.7561/0.7784; mppi_decoded_mlp 0% final 4.1218 | bnz |
| `results/planning_decoded_lowreg.json` | Seeded decoded, weak-reg K24 | R² 0.8420/0.8921; mppi_decoded_mlp 0% final 3.0147 best 2.5778 | bnz |
| `results/planning_diagnosis.json` | 3-layer diagnosis K6 default | oracle 0% final 4.0870; latent Pearson −0.1911/Spearman −0.1895; rollout 0.0125 | bnz |
| `results/planning_diagnosis_k24.json` | 3-layer diagnosis K24 default | oracle 1.0; Pearson −0.0023/Spearman +0.0247; rollout 0.0193; learned MPPI 0% | bnz |
| `results/planning_diagnosis_k24_lowreg.json` | K24 weak-reg diagnosis | oracle 1.0; Pearson −0.0551/Spearman −0.0979; rollout 0.0082 | bnz |
| `results/planning_diagnosis_k24_sigreg.json` | K24 SIGReg diagnosis | feat_std 0.5045; Pearson −0.2338/Spearman −0.4068; rollout 0.6103; MPPI 0% | bnz/sigreg-rep |
| `results/planning_diagnosis.png` | Layered diagnosis panel | — | bnz |
| `results/planning_success_rate.png` | Per-method success bars (0%) | — | bnz |
| `results/planning_learned_lowreg.json` | Learned cost head d128 | Spearman 0.7099; mppi_learned 3.0605/2.8083; 0% | m3-learned-cost/bnz |
| `results/planning_learned_d256.json` | Learned cost head d256 | Spearman 0.8077; mppi_learned 3.1460/2.7568; 0% | m3-learned-cost/bnz |
| `results/planning_learned_h2.json` | Horizon sweep h2 | mppi_learned 3.4062; decoded 2.8233; 0% | m3-learned-cost |
| `results/planning_learned_h3.json` | Horizon sweep h3 | mppi_learned 3.2285; decoded 2.8375; 0% | m3-learned-cost |
| `results/planning_learned_h4.json` | Horizon sweep h4 | mppi_learned 3.1532; decoded 2.9992; 0% | m3-learned-cost |
| `results/m3_ensemble_gate.json` | Epistemic gate | planner 0%; corr 0.3770/0.3935; disagree 0.011; err 0.264 | m3-model-fidelity |
| `results/m3_onpolicy.json` | On-policy DAgger/MBPO | 4 rounds err ~0.135 flat; final 0%/4.6985 | m3-multistep-rollout/bnz |
| `results/m3_multistep.json` | Multi-step rollout objective | free-run 0.611→0.086; 1-step 0.237→0.072; plan 0%/4.1255 | m3-multistep-rollout |
| `results/m3_metric_gate.json` | HYBRID gate mc=0.3 | Spearman-to-target 0.0845→0.9904; decode R² 0.8927→0.9788; rollout 0.0843→0.2834 | m3-metric-loss-hybrid/bnz |
| `results/m3_metric_gate_mc03/mc10/mc30.json` | HYBRID gate per coeff (all gate_pass) | Spearman saturates ~0.99 | m3-metric-loss-hybrid/bnz |
| `results/planning_learned_metric_mc03.json` | HYBRID planning mc=0.3 | mppi_latent 100%±0.0 final 0.804; decoded 100%/0.844; learned 80.6%±7.3/1.024; head Spearman 0.9954 | m3-metric-loss-hybrid/bnz |
| `results/planning_learned_metric_mc10.json` | HYBRID planning mc=1.0 | mppi_latent 97.2%±2.8; learned 38.9%±7.3 | m3-metric-loss-hybrid |
| `results/planning_learned_metric_mc30.json` | HYBRID planning mc=3.0 | mppi_latent 91.7%±4.8; learned 50.0%±8.3 | m3-metric-loss-hybrid |
| `results/m3_recognition.json` | Recognition probe mc=0/0.3 | dom-guild 0.8992→0.9711; basin 0.6896→0.8125 | m3-metric-loss-hybrid/bnz |
| `results/m3_recognition_mc10/mc30.json` | Recognition per coeff | dom-guild ~0.967–0.970; basin 0.771/0.779 | m3-metric-loss-hybrid |
| `results/metric_sweep_consolidated.{json,md}` | mc sweep consolidated table | success 100/97.2/91.7%; free-run 0.283/0.365/0.415 | m3-metric-loss-hybrid |
| `results/metric_hybrid.png` | HYBRID loop-closure + tradeoff figure | — | m3-metric-loss-hybrid/bnz |
| `results/exp1_instance_screen.json` | Exp1 gLV instance pre-screen | 4 valid instances; baseline oracle 1.0/0.7785; fastgrow rejected | m3-generalization |
| `results/exp1_generalization.{json,md}` | Exp1 matched eval | 4/4 cross tol; g3_s18 & g5_s30 100%; g4_s24 & strongcomp 88.9%±7.3 | m3-generalization |
| `results/BONUS_SESSION_RESULTS.md` | Bonus session consolidated | — | m3-generalization |
| `results/exp2_idm.{json,md}` | Exp2 IDM self-supervised | 0% all idm; Spearman +0.085→−0.313; head Spearman →0.74 | m3-idm-selfsup |
| `results/exp3_dim.{json,md}` | Exp3 bottleneck | 0% all dim; Spearman ~0.20–0.25; recog drops | m3-bottleneck |
| `results/realdata_infants.json` | M2 VICReg d128 (approx z-score) | linear 0.5079±0.0069 / 0.8960±0.0028 | bnz/sigreg-rep |
| `results/realdata_infants_fair.json` | M2 VICReg d128 fair | linear 0.5088/0.8961; MLP 0.4995/0.8884; Susagi MLP 0.5270/0.8898 | bnz/sigreg-rep |
| `results/realdata_infants_sigreg.json` | M2 SIGReg d256 | linear 0.5138/0.8912; MLP 0.5255/0.8936; finetune 0.5899/0.9180 | bnz/sigreg-rep |
| `results/realdata_infants_sigreg_d384.json` | M2 SIGReg d384 (best) | linear 0.5044/0.8894; MLP 0.5309/0.8986; finetune 0.5855/0.9063 | bnz |
| `results/realdata_infants_sigreg_ema.json` | M2 SIGReg+EMA | linear 0.4474/0.8534; MLP 0.5074/0.8848 | bnz |
| `results/tech_confounding.json` | Tech confounding pre-screen | NMI 0.0955; tech→biome bal-acc 0.139; biome→tech 0.6915 | m2-tech-invariance |
| `results/tech_invariance.json` | VICReg tech/biome probe | tech 0.9524±0.0022; biome 0.8050±0.0320 | m2-tech-invariance/bnz |
| `results/tech_invariance_sigreg.json` | SIGReg tech/biome probe | tech 0.9669±0.0022; biome 0.8163±0.0206 | m2-tech-invariance/sigreg-rep |
| `results/tech_invariance_vicreg_d256.json` | VICReg d256 tech | tech 0.9599±0.0027; biome 0.8432±0.0230 | exp2-genejepa |
| `results/tech_sweep_results.json` | DANN sweep (6 seeds) | tech never <0.90; biome → 0.69 at coeff10; ~3× biology lost | m2-tech-invariance |
| `results/tech_sweep_coral_results.json` | CORAL sweep (3 seeds) | SIGReg+CORAL c1000 tech 0.8441 / biome 0.7681 (−0.090 tech / ≈0 biome) | m2-tech-invariance |
| `results/tech_latent_umap_coords.json` | UMAP coords (3 panels, n=2989) | — | m2-tech-invariance |
| `results/tech_split_umap_coords.json` | UMAP split coords (2 panels) | within-biome tech: gut 0.99→0.98 etc | m2-tech-invariance |
| `results/tech_latent_pca.png` | PCA latent projection | — | m2-tech-invariance |
| `results/tech_latent_umap.png` | UMAP latents 3 panels | — | m2-tech-invariance |
| `results/tech_split_umap.png` | Per-tech biome layout | — | m2-tech-invariance |
| `results/tech_within_biome_umap.png` | Within-biome tech (artifact correction) | — | m2-tech-invariance |
| `results/tech_tradeoff.png` | DANN dual-axis (NEGATIVE) | — | m2-tech-invariance |
| `results/tech_tradeoff_coral.png` | CORAL dual-axis (PARTIAL POSITIVE) | — | m2-tech-invariance |
| `tahoe_probe/results/ablation/ablation_results.json` | Tahoe ablation (5 seeds) | pairs idm_off R²_shift 0.2012; idm_on 0.0169; celllines both ≤no-op; true-shift decode 0.75/0.84 | tahoe-perturbation-probe |
| `tahoe_probe/results/ablation/ablation_figure.png` | Tahoe two-panel R²(shift) bars | — | tahoe-perturbation-probe |

---

## Section 10 — Reproduce commands & file map

**GPU runs (Dalia GB200, `cd $WORK/eb_jepa`; gLV synthetic, no download):**
```bash
sbatch examples/microbiome_jepa/run_glv_final.sh        # headline IDM ablation: 3 seeds × {default, induce-collapse} → figure + JSON
sbatch examples/microbiome_jepa/run_glv_plan_k24.sh     # K=24 planning world model, default reg
sbatch examples/microbiome_jepa/run_glv_k24_lowreg.sh   # K=24 weak reg (best decoded planning)
sbatch examples/microbiome_jepa/run_realdata_big.sh     # real-data Layer A: corpus pretrain + FAIR probe, 100ep/50k/d256 (+finetune)
sbatch examples/microbiome_jepa/run_tech_invariance.sh  # amplicon-vs-WGS: JEPA vs raw / random / Susagi-imposter reps
```
Canonical one-liner (REPORT.md): `cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_glv_final.sh`.

**Fast local CPU runs (`PY=.venv-cpu/bin/python`; planning DIAGNOSIS fully CPU):**
```bash
$PY -m examples.microbiome_jepa.oracle_K_sweep        # controllability gate + figure
$PY -m examples.microbiome_jepa.plan_glv_decoded --checkpoint <wm_ckpt> --tag lowreg --overrides '{"data.n_candidate":24,"model.d_model":128}'
$PY -m examples.microbiome_jepa.diagnose_planning --checkpoint <wm_ckpt> --n_candidate 24
$PY -m examples.microbiome_jepa.make_planning_figure
$PY -m examples.microbiome_jepa.run_ablation --seeds 0 --epochs 6 --n_traj 64 --d_model 64
$PY examples/microbiome_jepa/eval_collapse.py
```
Layer A: `python -m examples.microbiome_jepa.main --fname .../cfgs/layerA_vicreg.yaml`. **Gotcha:** fire override syntax is `--key value`; bare `key=value` binds to positional `cfg` and breaks.

**File map (README.md, exact):**

| file | role |
|---|---|
| `cfgs/layerA_vicreg.yaml` | Layer A (static two-view set-JEPA) config |
| `cfgs/layerB_worldmodel.yaml` | Layer B world model; `model.regularizer.idm_coeff` ablation knob |
| `cfgs/layerA_real.yaml` | real-corpus Layer-A pretraining |
| `main.py` | Layer A trainer (two-view VICReg/BCS; CORAL/MMD/DANN on tech branch) |
| `train_worldmodel.py` | Layer B trainer (set-encoder + GRU predictor + IDM + VC/sim regularizer; `reg_type`, `metric_coeff`) |
| `eval_collapse.py` | collapse metric: frozen-encoder fast/slow probes |
| `run_ablation.py` | IDM-ablation driver |
| `run_glv_final.sh` | headline 3-seed ablation, both regimes (GPU) |
| `plan_glv.py` | latent-MPPI planning + MPC loop + baselines |
| `oracle_K_sweep.py` | controllability gate (oracle MPPI vs K) |
| `plan_glv_decoded.py` | decoded-state MPPI (z→x readout cost) + readout-fidelity sweep |
| `diagnose_planning.py` | controllability / latent-cost alignment / rollout fidelity |
| `make_planning_figure.py` | builds planning figure from JSONs |
| `run_glv_plan_k24.sh`, `run_glv_k24_lowreg.sh`, `run_glv_k24_big.sh`, `run_glv_k24_d256.sh` | K=24 planning world models (GPU) |
| `realdata.py` | real-corpus Layer A probe: frozen linear + MLP vs Susagi MLP |
| `tech_invariance.py` | amplicon-vs-WGS recoverability from JEPA vs raw / random / Susagi reps |
| `tech_sweep.py`, `tech_eval_batch.py`, `plot_tech_tradeoff.py`, `plot_latent_umap.py` | tech-invariance sweep/eval/figures |
| `run_realdata_big.sh`, `run_realdata_eval.sh`, `run_tech_invariance.sh` | real-data pretrain/probe + tech-invariance |
| `plan_glv_learned.py` | M3 learned monotonic cost + horizon sweep |
| `m3_ensemble_gate.py`, `m3_onpolicy.py`, `m3_multistep.py` | M3 model-fidelity levers |
| `m3_metric_gate.py`, `m3_recognition.py` | HYBRID metric-loss gate + recognition probe |
| `run_glv_k24_metric.sh`, `make_metric_figure.py`, `consolidate_metric_sweep.py` | HYBRID metric world model + figure/consolidation |
| `screen_instances.py`, `eval_generalization.py` | Exp1 generalization |
| `eval_pure_variants.py`, `run_glv_idm.sh`, `run_glv_dim.sh` | Exp2/Exp3 self-supervised falsification |
| `run_realdata_sigreg.sh`, `run_realdata_ema.sh` | SIGReg / SIGReg+EMA pretraining |
| `losses.py: SIGReg_IDM_Sim_Regularizer` | SIGReg world-model regularizer |
| `probe_downstream.py`, `baselines_port.py` | earlier Layer A downstream probe scaffolding |
| `eb_jepa/datasets/microbiome/{glv,otu_data,transforms,traj}.py` | gLV simulator + OTU/traj datasets + CLR/z-score |
| `eb_jepa/architectures.py` | `SetTransformerEncoder`; GRL/`TechAdversaryHead`/`dann_lambda` (tech branch) |
| `examples/tahoe_probe/*` | Tahoe drug-perturbation generality probe |
| `results/` | committed figures + raw JSON of measured runs |

**3-minute demo flow (5 beats):** (1) problem — microbiome noisy/sparse/temporal → JEPA; gLV non-monotonic attractors (greedy fails 6/6). (2) headline figure `ablation_collapse.png` — IDM recovers the dropped intervention. (3) planning — diagnosed/partially-closed negative; `oracle_K_sweep.png` (solvable only at K=24); decoded R² 0.78→0.89 ⇒ final 4.12→3.01; `planning_diagnosis.png`. (4) real data + invariance — frozen linear probe ties supervised MLP on AUC (0.896 vs 0.890); tech-invariance honest negative. (5) JEPA lessons — IDM regime-dependence; faithful-1-step ≠ metric-multi-step; probing-good ≠ planning-good.

**Consolidating documents (all on `bnz`):** `REPORT.md` (latest `19387f5`), `README.md` (`a61e437`, integrity-corrected `50323dc`/`00a05a1`), `/PLAN.md` (2026-06-20 ledger appended in `c2a275c`).

---

## Section 11 — Honest verdict summary

**POSITIVE results (with number):**
- **M4 IDM collapse-and-recovery (gLV, induce-collapse regime):** `fast_r2_action` IDM on **0.748 ± 0.051** vs off **0.520 ± 0.021**, Δ **+0.229**, on>off all 3 seeds, non-overlapping error bars (job 74610).
- **M2 frozen-rep win (SIGReg):** SIGReg MLP probe **matches** Susagi MLP baseline at d256 (0.5255/0.8936 vs 0.5270/0.8898) and **beats** it at d384 (**0.5309/0.8986**).
- **M2 supervised ceiling:** finetune upper bound **0.5899 / 0.9180** (SIGReg-d256) beats published Susagi reference (0.549/0.912) — explicitly not the label-free claim.
- **HYBRID metric-loss closes the planning loop:** raw-latent MPPI **0% → 100% ± 0%**, final **0.804 ≈ oracle 0.790**; latent↔true Spearman **0.085 → 0.990** (HYBRID, uses true-state supervision — NOT pure JEPA).
- **Exp1 generalization:** closure holds on **4/4** diverse gLV instances (pure JEPA 0%); 2/4 match oracle 100%, 2/4 reach 88.9% ± 7.3.
- **CORAL partial tech fix:** SIGReg+CORAL removes linear tech-leak **−0.090** (0.934→0.844) at **≈0 biome cost** (0.770→0.768) — where DANN failed.
- **Tahoe predictor (no IDM):** beats per-drug baseline on held-out pairs R²(shift) **0.201 vs 0.149**; drug decodable from true shift 75–84% top-1.

**NEGATIVE results (with number):**
- **Pure-JEPA planning:** **0% success** across every lever (latent MPPI, decoded, learned cost, capacity, horizon, ensemble pessimism, on-policy, multi-step); oracle reaches 0.79.
- **SIGReg M3 geometry:** fixes isotropy (feat_std 0.01→0.50) but distances anti-correlated (Pearson −0.23, Spearman −0.41), rollout degrades (0.01→0.61), MPPI 0%.
- **SIGReg / VICReg tech-invariance:** trained JEPA most tech-separable (SIGReg 0.967 > VICReg 0.952 > raw 0.938 > random 0.897 > Susagi 0.891).
- **DANN adversary:** tech never <0.90, biome lost ~3× faster than tech, no sweet spot.
- **EMA teacher (GeneJepa):** hurts M2 (linear 0.447, MLP 0.507 < no-EMA 0.514/0.526).
- **Exp2 IDM self-supervised:** 0% planning at all idm_coeff; raw latent↔true Spearman driven negative to −0.313.
- **Exp3 bottleneck:** 0% planning at all dims; marginal Spearman ~0.20–0.25; costs recognition (guild 0.90→0.67) and decode R² (→0.66).
- **Tahoe IDM:** HURTS (R²_shift 0.201→0.017, monotonic in idm_coeff) — opposite of gLV M4.
- **Tahoe cross-cell-line transfer:** both model arms at/below no-op (−0.21 / −0.15 vs −0.16).
- **M2 VICReg-d128 (initial, fair):** MLP probe 0.4995/0.888 did NOT beat Susagi MLP baseline.

**DIAGNOSTIC results (with number):**
- **Oracle controllability:** task uncontrollable at K≤21 (0% success, final 4.09→1.73), fully controllable at K=24 (100%, final 0.79).
- **Layered planning diagnosis (K24):** controllable (oracle 1.0) + faithful dynamics (~2% rollout) but uninformative latent cost (Pearson −0.00, Spearman +0.02) → bottleneck = representation geometry.
- **Decoded readout-fidelity trend:** R² 0.78→0.89 ⇒ final distance 4.12→3.01 (beats greedy 3.26), never crosses tol.
- **M3 distribution-shift gate:** predictor 1-step error 0.072 (train) vs 0.264 (planner OOD), 3.7× gap; on-policy closes error to ~0.135 but planning still 0%.
- **M3 multistep:** free-running 6-step error cut 0.611→0.086 (−86%), 1-step 0.237→0.072, planning still 0% → dynamics fidelity exonerated.
- **M2 corpus z-score:** corpus_5000 vs infant-token changes linear from 0.5079→0.5088 (negligible, caveat removed honestly).
- **M2 MLP-vs-linear self-handicap check:** MLP probe (0.4995) < linear (0.5088) → rep already linearly separable.
- **Tahoe true-shift decodability:** drug identity 75–84% top-1 from true latent shift (chance 1–2%) → action signal present in real embedding.
- **Tech confounding ceiling:** NMI(tech;biome) 0.0955; biome→tech floor ≈ 0.6915 balanced (ESTIMATE).
- **Tech within-biome UMAP correction:** DANN coeff=10 within-biome tech bal-acc barely moves (gut 0.99→0.98) — earlier UMAP "mixing" was a projection artifact.

**The narrative spine:** predict-in-latent drifts to slow features (IDM rescues it, regime-dependently); a representation can be faithful-for-one-step-prediction yet not a metric-space for multi-step-planning (only a supplied true-state metric closes it); "good for probing" ≠ "good for planning"; tech-invariance is not free (a JEPA only becomes invariant to a nuisance its losses explicitly span; alignment beats adversarial for a dominant linearly-accessible nuisance).

---

## Open items / things to verify before publishing

1. **MISSING JSON behind the SIGReg-M2 comparison "VICReg-d256" row.** REPORT.md quotes a VICReg (100ep/d256) row at **linear 0.484 / 0.878, MLP 0.484 / 0.873** ("longer/bigger VICReg got worse, 0.484 vs 30ep/d128 0.509"), but **no `realdata_infants_vicreg_d256.json` (or any matched-budget VICReg-d256 realdata file) exists on any branch**. Treat as MEASURED-but-provenance-incomplete; locate on the cluster or downgrade the "+3–4pp attributable win" anchor.

2. **Tech-invariance baseline disagreement (different corpus draws/seed counts).** VICReg JEPA tech: `tech_invariance.json` = **0.9524** vs DANN-sweep coeff=0 = **0.9603**; SIGReg = **0.9669** (`tech_invariance_sigreg.json`) vs sweep coeff=0 = **0.9339**. Both MEASURED on different balanced draws / seed counts (single draw vs 6-seed). Report with provenance; not a contradiction.

3. **Tech verdict disagreement (REPORT vs current branch).** Consolidated `bnz:REPORT.md`/`README.md` carry the tech-invariance verdict as the **honest negative** (no CORAL). The current `m2-tech-invariance` branch has later CORAL commits (`c19505b`, `6eaa7c7`, `tech_sweep_coral_results.json`) claiming a **partial fix**. Decide whether the published report's tech verdict is "honest negative" or "partially fixed by CORAL"; the CORAL JSON is not present on `bnz`.

4. **Tahoe `idm_coeff`-sweep intermediate points unbacked by a committed log.** `RESULTS.md` cites `results/coeff_sweep.log` (idm_coeff 0.01/0.1/0.3 rows), which is **not in the committed tree** (only `idm_coeff_sweep.py` script + the 5-seed `ablation_results.json`). Endpoints (0.0→0.201, 1.0→0.017) are verifiable from the JSON; the intermediate sweep cannot be re-verified from the repo.

5. **`m3-*` related lever family not fully censused.** `bnz` carries `m3_metric_gate.py`/`m3_recognition.py` and `planning_learned_metric_mc{03,10,30}.json` etc. (covered in 6.6), and additional `m3_metric_gate{,_mc03,_mc10,_mc30}.json` / `m3_recognition_mc{10,30}.json` exist only on `m3-metric-loss-hybrid`. If a complete M3 census is required, confirm none are missing from the index.

6. **`tech_invariance.json` REPORT reports accuracy, JSON also has balanced-accuracy.** REPORT uses plain `acc` (JEPA 0.952, raw 0.938, Susagi 0.891, random 0.897); the JSON additionally carries near-identical balanced-acc. No conflict, just note which metric the published report uses.

7. **No "UMAP within-biome projection-artifact" correction appears in the `bnz` consolidating docs** (REPORT.md/README.md/PLAN.md) — it lives only in the `m2-tech-invariance` branch (commit `1044f20`). Ensure the catch is surfaced in the final report if the tech section is included.

8. **Real-data Susagi parsers in `otu_data.py` flagged UNVERIFIED on this Mac** (22 GB corpus is cluster-only). The measured real-data probe numbers come from cluster runs; the local-readable parsing path is not independently verified here.
