# M2 follow-up: making the set-JEPA encoder invariant to sequencing technology

**Branch:** `m2-tech-invariance` (off `sigreg-rep`). NOT pushed to origin; NOT folded into `bnz`;
gLV/M3/M4 untouched. Every table below is labelled **MEASURED** (from a real run, with the job id /
checkpoint) or **PENDING**. No fabricated numbers.

## The arc (diagnose → fix)
M2 produced a documented **negative**: our set-JEPA community embedding is *more* separable by
sequencing technology (amplicon-16S vs WGS-shotgun) than even the raw input — i.e. the SSL encoder
*amplifies* a technical nuisance instead of abstracting it away. This follow-up turns that into a
diagnose-then-fix arc: add a technology-invariance term (adversarial DANN) to Layer-A training,
sweep its strength, and report the dual-axis tradeoff (technology DOWN, biology MAINTAINED) against
the confounding ceiling.

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

## Result — tradeoff sweep — PENDING (running on Dalia, job(s) TBD)
`invariance_coeff ∈ {0, 0.3, 1.0, 3.0}` × seeds {0,1,2} × {VICReg, SIGReg}; fresh dual-axis probe per
frozen encoder. Table to be filled with MEASURED mean±s.e. (tech bal-acc ↓, biome bal-acc maintained),
read against the ~0.69 confounding floor. Sweet spot = lowest tech-acc that still preserves biome.

## Reproduce
- Confounding: `sbatch examples/microbiome_jepa/run_tech_confounding.sh`
- Sweep: `NGPU=3 sbatch examples/microbiome_jepa/run_tech_sweep.sh` (respects the shared gres/gpu=3 cap)
- coeff=0 bit-exactness: `examples/microbiome_jepa/main.py` with `loss.invariance_coeff=0` ≡ baseline.
