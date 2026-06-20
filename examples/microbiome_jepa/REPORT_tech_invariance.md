# M2 follow-up: making the set-JEPA encoder invariant to sequencing technology

**Branch:** `m2-tech-invariance` (off `sigreg-rep`). NOT pushed to origin; NOT folded into `bnz`;
gLV/M3/M4 untouched. Every table below is labelled **MEASURED** (from a real run, with the job id /
checkpoint) or **PENDING**. No fabricated numbers.

## The arc (diagnose → fix)
M2 produced a documented **negative**: our set-JEPA community embedding is *more* separable by
sequencing technology (amplicon-16S vs WGS-shotgun) than even the raw input — i.e. the SSL encoder
*amplifies* a technical nuisance instead of abstracting it away. This follow-up adds a
technology-invariance term (adversarial DANN) to Layer-A training, sweeps its strength, and reports the
dual-axis tradeoff (technology DOWN, biology MAINTAINED) against a measured confounding ceiling.
**Outcome (MEASURED, honest): the fix FAILS** — DANN never drives technology near chance/the floor and
destroys biology faster than it removes technology. So this is a *diagnose → attempt → diagnosed
negative* arc, not a clean fix; the diagnosis (why a single adversary can't scrub a dominant,
linearly-accessible nuisance) is the contribution.

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

This is a **robust, reportable negative**: the M2 technology-leak survives an adversarial fix; the
fix trades away biology faster than technology. Honest next steps (NOT run here): a **stronger
discriminator** (k updates/step), **distribution alignment** (MMD/CORAL) which doesn't rely on a
min-max race, or **conditional alignment within biome**; and scVI-style *conditioning* (feed technology
to a decoder so the latent need not encode it). Reference batch-correction methods (Harmony/scVI) remain
the domain standard to compare against.

### Honest limitations
- One SIGReg point is n=5 (one `bcs c1.0` seed failed to train); 59/60 checkpoints. Everything else n=6.
- The eval is on n=4960 (tech) / 4062 (biome) balanced corpus samples; biome is 8-class, imbalanced
  (gut-dominated), hence balanced-accuracy is the headline.
- The negative is specific to **this** DANN configuration (single GRL adversary, this schedule); it does
  not prove invariance is impossible — only that the cheap adversarial route fails on a dominant nuisance.

## Reproduce
- Confounding: `sbatch examples/microbiome_jepa/run_tech_confounding.sh`
- Train sweep: `NGPU=<=3 sbatch examples/microbiome_jepa/run_tech_sweep.sh` (shared gres/gpu=3 cap)
- Eval (fast, parallel CPU): `sbatch examples/microbiome_jepa/run_tech_eval_batch.sh`
- Tradeoff figure: `python -m examples.microbiome_jepa.plot_tech_tradeoff --sweep_json <...>/sweep_results.json --confounding_json <...>/confounding.json`
- Qualitative latents: `python -m examples.microbiome_jepa.plot_latent_umap --layout {compare,tech_split,tech_within_biome} --from_reps <reps.npz>`
- coeff=0 bit-exactness: `main.py` with `loss.invariance_coeff=0` ≡ baseline (verified).
