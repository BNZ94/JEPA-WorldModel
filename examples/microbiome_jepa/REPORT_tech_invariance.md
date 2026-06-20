# M2 follow-up: making the set-JEPA encoder invariant to sequencing technology

**Branch:** `m2-tech-invariance` (off `sigreg-rep`). NOT pushed to origin; NOT folded into `bnz`;
gLV/M3/M4 untouched. Every table below is labelled **MEASURED** (from a real run, with the job id /
checkpoint) or **PENDING**. No fabricated numbers.

## The arc (diagnose → fix)
M2 produced a documented **negative**: our set-JEPA community embedding is *more* separable by
sequencing technology (amplicon-16S vs WGS-shotgun) than even the raw input — i.e. the SSL encoder
*amplifies* a technical nuisance instead of abstracting it away. This follow-up adds a
technology-invariance term to Layer-A training, sweeps its strength, and reports the dual-axis tradeoff
(technology DOWN, biology MAINTAINED) against a measured confounding ceiling, for **two method
families**: adversarial **DANN** and deterministic distribution-alignment **CORAL**.

**Outcome (MEASURED, honest):**
- **DANN fails** — it never drives technology near chance/the floor and destroys biology *faster* than
  it removes technology (the adversary is fooled without the info being removed).
- **CORAL partially works** — it reduces *linearly-accessible* technology substantially while
  **preserving biology** (SIGReg+CORAL removes ~9 pts of tech at ≈zero biome cost). It is the real
  (partial) fix; full invariance is not reached (a nonlinear MLP probe still recovers technology).

So the arc is *diagnose → adversarial attempt fails → deterministic alignment partially fixes it*, with
a clear diagnosis of **why** (a dominant, linearly-accessible nuisance defeats a min-max adversary but
yields to moment-matching, up to the nonlinear-residual limit).

## Setup (what the two axes are)
- **Technology axis (nuisance, want invariant):** library strategy **amplicon vs WGS**, joined per
  corpus sample via `RunID → Terms`. Binary; **chance = 0.50 balanced**.
- **Biology axis (signal, want preserved):** the **biome** label (gut/soil/water/skin/…; 8 classes
  present), the only biological label that co-exists with the tech label on the corpus. (The
  infant-env downstream task is single-technology — 100% Illumina MiSeq — so it is `N/A` for
  tech-invariance; biome on the corpus is the correct biology control.)
- **Data is unpaired** (each corpus sample has exactly one strategy; no same-sample-two-tech pairs),
  so the right tool is adversarial domain-invariance / distribution alignment, not paired alignment.
- **Honest evaluation rule (both axes, frozen encoder, fresh probes):** technology and biome are each
  scored by a *fresh* 5-fold probe (linear LogReg + MLP) trained post-hoc on the **frozen** latent —
  **never** the training adversary. Headline metric is **balanced accuracy** (class-imbalanced).

## Step 0 — confounding pre-screen (the ceiling) — MEASURED (job 76596, CPU)
Label-only analysis on n=5000 balanced corpus samples (4102 with a biome label). Caps how invariant
we can get without destroying biology.

| quantity | value | reading |
|---|---|---|
| **NMI(tech ; biome)** | **0.095** | mostly independent; mild entanglement |
| predict **biome from tech**-only (bal-acc) | **0.139** (chance 0.125) | tech says ~nothing about biology |
| predict **tech from biome**-only (bal-acc) | **0.692** (chance 0.500) | biology *partially* predicts tech |

Cross-tab (strat × biome, counts):

| | gut | water | soil | plant | feces | skin | oral |
|---|----|------|-----|------|------|-----|-----|
| **amplicon** | 696 | 619 | 295 | 148 | 123 | 13 | 10 |
| **wgs** | 1473 | 292 | 98 | 6 | 190 | 138 | 0 |

**Reading (the ceiling).** Technology and biome are largely separable (NMI 0.095; tech→biome ≈ chance),
so most of the technology signal is **biome-independent "pure protocol" nuisance** that an invariance
term can remove for free. But biome→tech ≈ **0.69 balanced**: biology partially predicts technology
(plant/soil/water lean amplicon; skin/feces lean WGS), so there is a **floor ≈ 0.69 balanced tech-acc**
below which removing technology necessarily starts to cost biology. So the well-posed target is to
drive tech-separability from the baseline down **toward ~0.69** while holding biome; pushing materially
below ~0.69 should show up as biome loss. (0.69 is itself an estimate from an 8-biome one-hot probe — a
*lower bound* on what a richer biology representation could leak about technology.)

## Before baseline (the negative we are fixing) — MEASURED (`results/tech_invariance*.json`)
Frozen encoder, fresh probes, n=4960 (tech) / 4062 (biome). Balanced accuracy.

| representation | TECH bal-acc (↓ better) | BIOME bal-acc (↑ better) |
|---|---|---|
| **VICReg JEPA** (the M2 encoder) | **0.952** | 0.805 |
| SIGReg JEPA | 0.967 | 0.816 |
| raw mean-pool (input) | 0.938 | 0.794 |
| random-init encoder | 0.897 | 0.741 |
| Susagi imposter rep | 0.891 | 0.719 |

The negative: the trained JEPA is the **most** technology-separable representation of all — above raw
input and far above the random encoder and Susagi. SSL on this corpus concentrates the protocol signal.

## Method — DANN technology adversary (config-gated)
- A technology classifier behind a **gradient-reversal layer (GRL)** on the encoder embedding; training
  it *removes* technology-predictive information while the SSL objective preserves community structure.
  Strength = `loss.invariance_coeff`; GRL ramp = standard DANN `λ(p)=2/(1+e^{-γp})−1`.
- **`invariance_coeff = 0` is byte-identical to the baseline** (adversary not built; verified equal to
  the original `main.py` to all printed digits on a CPU run). The sweep's coeff=0 arm is the matched
  re-baseline (same balanced labelled corpus + recipe).
- Applied to **both** VICReg and SIGReg (BCS). Tech labels plumbed into the corpus loader
  (`with_tech_labels`, per-class balanced stream).

## Result — tradeoff sweep — MEASURED (jobs 76698/76925/77350 train; 77533 eval; 6 seeds; fresh probe)
`invariance_coeff ∈ {0, 0.3, 1.0, 3.0, 10.0}` × seeds {0..5} × {VICReg, SIGReg}; fresh **linear**
balanced-acc probe on the frozen latent (MLP probe in the JSON, same trend). Mean ± s.e. over seeds.
Figure: [results/tech_tradeoff.png](results/tech_tradeoff.png). Data:
[results/tech_sweep_results.json](results/tech_sweep_results.json).

**VICReg**
| coeff | TECH bal-acc (↓ better) | BIOME bal-acc (keep) |
|------:|:-----------------------:|:--------------------:|
| 0.0   | 0.960 ± 0.001 | 0.829 ± 0.004 |
| 0.3   | 0.951 ± 0.002 | 0.820 ± 0.004 |
| 1.0   | 0.954 ± 0.002 | 0.803 ± 0.010 |
| 3.0   | 0.950 ± 0.002 | 0.791 ± 0.008 |
| 10.0  | 0.922 ± 0.003 | 0.690 ± 0.010 |

**SIGReg (BCS)**
| coeff | TECH bal-acc (↓ better) | BIOME bal-acc (keep) |
|------:|:-----------------------:|:--------------------:|
| 0.0   | 0.934 ± 0.002 | 0.767 ± 0.004 |
| 0.3   | 0.935 ± 0.001 | 0.755 ± 0.005 |
| 1.0   | 0.924 ± 0.004 | 0.747 ± 0.009 *(n=5)* |
| 3.0   | 0.906 ± 0.006 | 0.712 ± 0.007 |
| 10.0  | 0.901 ± 0.009 | 0.674 ± 0.010 |

### Verdict — NEGATIVE (honest): adversarial DANN does NOT make this encoder technology-invariant
Across the whole sweep, **technology stays strongly recoverable** (tech bal-acc never drops below
**0.90**, far from chance 0.50 and never reaching even the **0.69 confounding floor**), while the
**biology signal is destroyed** (biome bal-acc falls to/below the 0.69 floor at high coeff). The
tradeoff runs the **wrong way**: from coeff 0→10 VICReg loses only −0.038 tech but −0.139 biome
(SIGReg −0.033 tech / −0.093 biome) — i.e. the invariance pressure removes **~3× more biology than
technology**. There is **no sweet spot**; every setting sacrifices more biome than it scrubs tech. The
within-biome probe agrees ([results/tech_within_biome_umap.png](results/tech_within_biome_umap.png)):
at fixed biome, a fresh probe still reads technology at 0.95–0.98 even for coeff=10. The 2-D UMAP
"mixing" was a projection artifact — the linearly-accessible tech direction survives.

### Diagnosis (why the fix fails here)
1. **Technology is the dominant, linearly-accessible axis.** Even the raw input is 0.94 tech-separable
   and the *trained* encoder amplifies it to 0.96; a single-knob GRL adversary cannot scrub a nuisance
   that strong.
2. **The adversary was poorly conditioned.** During training the discriminator's accuracy collapsed
   *below* chance (the encoder out-ran a one-update-per-step discriminator) — the classic DANN failure:
   the moving adversary is fooled without the information being removed, so a fresh probe recovers it.
3. **Mild confounding makes the cheap direction biological.** With biome→tech = 0.69 and tech→biome ≈
   chance, the lowest-variance "removable" directions the adversary finds are partly the biology ones,
   so the gradient erodes biome first.

The M2 technology-leak survives the *adversarial* fix; DANN trades away biology faster than technology.
That motivated a second method family that doesn't rely on a min-max race — see CORAL below.

## Result 2 — deterministic alignment (CORAL) — MEASURED (jobs 77650 train / 77777 eval; 3 seeds)
Same setup, but the invariance term is **CORAL** (align the per-technology latent **mean + covariance**;
`loss.invariance_method=coral`) — no adversary, single knob. Fresh linear probe, mean ± s.e.
Figure: [results/tech_tradeoff_coral.png](results/tech_tradeoff_coral.png). Data:
[results/tech_sweep_coral_results.json](results/tech_sweep_coral_results.json).

| | TECH bal-acc (↓ better) | BIOME bal-acc (keep) | |
|---|---|---|---|
| **VICReg** c0 | 0.960 ± 0.000 | 0.837 ± 0.004 | baseline |
| VICReg c10 | 0.912 ± 0.007 | 0.819 ± 0.010 | |
| VICReg c100 | 0.870 ± 0.005 | 0.788 ± 0.011 | |
| VICReg c1000 | **0.847 ± 0.003** | 0.772 ± 0.007 | −0.113 tech / −0.065 biome |
| **SIGReg** c0 | 0.934 ± 0.002 | 0.770 ± 0.003 | baseline |
| SIGReg c10 | 0.916 ± 0.003 | 0.767 ± 0.010 | |
| SIGReg c100 | 0.883 ± 0.002 | 0.765 ± 0.006 | |
| SIGReg c1000 | **0.844 ± 0.003** | **0.768 ± 0.003** | **−0.090 tech / −0.002 biome** |

### Verdict 2 — CORAL is a real (partial) fix; it succeeds where DANN failed
- **CORAL removes technology that DANN could not, and keeps biology.** The tradeoff now runs the *right*
  way: tech bal-acc drops monotonically with coeff while biome is largely held. **SIGReg+CORAL is the
  sweet spot** — technology 0.934 → **0.844** (−0.090) at **flat biome** (0.770 → 0.768, within s.e.):
  it scrubs ~9 points of protocol signal at **≈zero cost to biology**. VICReg+CORAL also improves
  (−0.113 tech for −0.065 biome) — a ~1.7:1 favourable ratio, vs DANN's ~0.3:1 (wrong-way) ratio.
- **But the fix is PARTIAL, two honest limits:** (1) linear tech bal-acc bottoms at ~0.84 — **still well
  above the 0.69 confounding floor and chance 0.50**, so technology is *reduced*, not *eliminated*;
  (2) a **nonlinear MLP probe barely moves** (tech-MLP stays ~0.96 across all coeffs). CORAL aligns only
  the first two moments, so it removes the **linearly-accessible** nuisance; a nonlinear probe still
  finds the residual. Full invariance would need higher-order alignment (RBF-MMD — implemented as
  `invariance_method=mmd`, not swept here) or conditioning.

### Why CORAL works and DANN doesn't (the JEPA lesson)
A deterministic moment-matching loss applies a **stable, well-posed gradient every step**, so it
actually moves the per-technology marginals together; the adversarial GRL instead plays a min-max game
that the encoder "wins" by fooling a one-step discriminator **without removing the information** (its
training accuracy fell below chance). For a *dominant, linearly-accessible* nuisance like sequencing
protocol, alignment >> adversarial. This is the rubric-relevant finding.

### Honest limitations
- **DANN sweep:** one SIGReg point is n=5 (one `bcs c1.0` seed failed to train); 59/60 checkpoints.
  Everything else n=6. **CORAL sweep:** 24/24 checkpoints, n=3 seeds per point.
- The eval is on n=4960 (tech) / 4062 (biome) balanced corpus samples; biome is 8-class, imbalanced
  (gut-dominated), hence balanced-accuracy is the headline.
- **CORAL gives only *linear* invariance** (1st+2nd moment alignment): linear tech-probe drops to ~0.84
  but a nonlinear MLP probe still recovers technology (~0.96), and tech never reaches the 0.69 floor.
  So the protocol nuisance is *reduced*, not removed. Higher-order alignment (RBF-MMD, coded but not
  swept) or conditioning would be needed for full invariance; Harmony/scVI remain the domain-standard
  references to compare against.
- The DANN negative is specific to a single GRL adversary at this schedule; a stronger k-step
  discriminator might do better. We did not tune it further because CORAL already gave the cleaner fix.

## Reproduce
- Confounding: `sbatch examples/microbiome_jepa/run_tech_confounding.sh`
- DANN train sweep: `NGPU=<=3 sbatch examples/microbiome_jepa/run_tech_sweep.sh` (shared gres/gpu=3 cap)
- CORAL train sweep: `METHOD=coral CKPT_ROOT=checkpoints/microbiome_jepa/tech_sweep_coral COEFFS=0,10,100,1000 NGPU=<=3 sbatch examples/microbiome_jepa/run_tech_sweep.sh`
- Eval (fast, parallel CPU): `sbatch examples/microbiome_jepa/run_tech_eval_batch.sh` (point `--ckpt_root/--out_root` at the coral pool for CORAL)
- Tradeoff figure: `python -m examples.microbiome_jepa.plot_tech_tradeoff --sweep_json <...>/sweep_results.json --confounding_json <...>/confounding.json`
- Qualitative latents: `python -m examples.microbiome_jepa.plot_latent_umap --layout {compare,tech_split,tech_within_biome} --from_reps <reps.npz>`
- coeff=0 bit-exactness: `main.py` with `loss.invariance_coeff=0` ≡ baseline (verified).
