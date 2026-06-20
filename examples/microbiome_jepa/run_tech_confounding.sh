#!/usr/bin/env bash
# M2 tech-invariance STEP 0: label-only confounding pre-screen (CPU, fast).
# Measures NMI(tech;biome) + directional probe accuracies on the real corpus to bound
# how invariant the encoder can become without losing biology. No encoder needed.
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_tech_confounding.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=mb_confound
#SBATCH --output=mb_confound_%j.out
#SBATCH --error=mb_confound_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerA_real.yaml
DATA=$EBJEPA_DSETS/susagi/data
OUT=${OUT:-$WORK/checkpoints/microbiome_jepa/tech_confounding}
$PY -m examples.microbiome_jepa.tech_invariance \
  --fname $CFG --data_dir $DATA --per_class_cap ${CAP:-2500} \
  --confounding_only true --device cpu --out $OUT
echo "MB_CONFOUND_DONE"
