#!/usr/bin/env bash
# M2 tech-invariance SWEEP: train the DANN invariance_coeff grid (VICReg+SIGReg, multi-seed)
# on the real corpus across the node's 4 GPUs, then frozen-probe each (tech DOWN, biome KEEP).
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_tech_sweep.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --job-name=mb_techsweep
#SBATCH --output=mb_techsweep_%j.out
#SBATCH --error=mb_techsweep_%j.out
# NOTE: the vivatech-dynamics account shares a hard cap of gres/gpu=3 across the team.
# Default to gpu:3; override at submit (sbatch --gres=gpu:N) AND pass NGPU=N so the
# orchestrator's concurrency matches the allocation. Be considerate of teammates.
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
DATA=$EBJEPA_DSETS/susagi/data
$PY -c "import torch; print('torch', torch.__version__, 'n_gpu', torch.cuda.device_count())"

LOSSES=${LOSSES:-vicreg,bcs}
COEFFS=${COEFFS:-0,0.3,1.0,3.0}
SEEDS=${SEEDS:-0,1,2}
EP=${EP:-30}; NS=${NS:-16000}; DM=${DM:-128}; NGPU=${NGPU:-3}

# TRAIN ONLY (GPU): eval runs separately on CPU (run_tech_eval.sh) so GPUs are released
# during the CPU probe phase — friendlier to the shared gres/gpu=3 cap.
$PY -m examples.microbiome_jepa.tech_sweep \
  --losses $LOSSES --coeffs $COEFFS --seeds $SEEDS \
  --n_gpus $NGPU --epochs $EP --ns $NS --d_model $DM \
  --data_dir $DATA --per_class_cap ${CAP:-2500} --skip_eval true
echo "MB_TECHSWEEP_TRAIN_DONE"
