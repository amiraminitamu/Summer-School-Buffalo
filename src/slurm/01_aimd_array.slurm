#!/bin/bash
#SBATCH --job-name=pc60_aimd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=90G
#SBATCH --time=30:00:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/aimd_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output_production
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PYTHON="${PYSCF_PYTHON:-python}"
replica=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
seed=$((2026071400 + SLURM_ARRAY_TASK_ID))

"${PYTHON}" src/04_run_aimd.py \
  --xyz "data/production_starts/start_${replica}.xyz" \
  --basis 6-31g --xc pbe-d3bj --grid-level 0 \
  --dt-fs 0.5 --tau-fs 100 --steps 200 --temperature 300 \
  --seed "${seed}" --threads "${SLURM_CPUS_PER_TASK}" \
  --memory-mb 85000 --conv-tol 1e-6 \
  --outdir "output_production/traj_${replica}"
