#!/usr/bin/env bash
# M2 tech-invariance EVAL — batched + parallel (CPU). Loads corpus once, fresh-probes every
# checkpoint across cores (reuses any per-tag JSON already on disk). No GPU (frees the shared cap).
# Submit: cd $WORK/scratch/m2ti && sbatch examples/microbiome_jepa/run_tech_eval_batch.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --job-name=mb_evalbatch
#SBATCH --output=mb_evalbatch_%j.out
#SBATCH --error=mb_evalbatch_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
DATA=$EBJEPA_DSETS/susagi/data
# one BLAS thread per worker (avoid oversubscription with the process pool)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

$PY -m examples.microbiome_jepa.tech_eval_batch \
  --losses ${LOSSES:-vicreg,bcs} --coeffs ${COEFFS:-0,0.3,1.0,3.0,10.0} --seeds ${SEEDS:-0,1,2,3,4,5} \
  --d_model ${DM:-128} --per_class_cap ${CAP:-2500} --n_workers ${NW:-24} --data_dir $DATA
echo "MB_EVALBATCH_DONE"
