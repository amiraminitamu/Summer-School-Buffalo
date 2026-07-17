#!/bin/bash
#SBATCH --job-name=pc60_electronic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/electronic_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output_electronic
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PYTHON="${PYSCF_PYTHON:-python}"
replica=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")

"${PYTHON}" src/05_extract_electronic.py \
  --trajectory "output_production/traj_${replica}/aimd.md.xyz" \
  --outdir "output_electronic/traj_${replica}" \
  --start 0 --stop 200 --stride 1 --dt-fs 0.5 \
  --basis 6-31g --xc pbe --grid-level 0 --conv-tol 1e-6 \
  --threads "${SLURM_CPUS_PER_TASK}" --memory-mb 95000
