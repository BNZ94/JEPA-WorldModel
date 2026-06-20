#!/usr/bin/env bash
# M2 tech-invariance EVAL (CPU): frozen dual-axis probe (tech DOWN, biome KEEP) on every
# checkpoint produced by run_tech_sweep.sh, + the tradeoff summary. No GPU (frees the shared cap).
# Submit AFTER training: cd $WORK/scratch/m2ti && sbatch examples/microbiome_jepa/run_tech_eval.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=mb_techeval
#SBATCH --output=mb_techeval_%j.out
#SBATCH --error=mb_techeval_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
DATA=$EBJEPA_DSETS/susagi/data

LOSSES=${LOSSES:-vicreg,bcs}
COEFFS=${COEFFS:-0,0.3,1.0,3.0}
SEEDS=${SEEDS:-0,1,2}
DM=${DM:-128}

$PY -m examples.microbiome_jepa.tech_sweep \
  --losses $LOSSES --coeffs $COEFFS --seeds $SEEDS \
  --epochs 0 --ns 0 --d_model $DM \
  --data_dir $DATA --per_class_cap ${CAP:-2500} --skip_train true
echo "MB_TECHEVAL_DONE"
