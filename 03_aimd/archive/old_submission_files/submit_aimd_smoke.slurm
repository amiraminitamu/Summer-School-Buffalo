#!/bin/bash
#SBATCH --job-name=pc60_aimd_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=pc60_aimd.%j.out
#SBATCH --error=pc60_aimd.%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pyscf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# First run: plumbing/timing test. Change to def2-svp only after this succeeds.
python 01_pyscf_aimd.py \
  --basis sto-3g \
  --xc pbe-d3bj \
  --grid-level 1 \
  --dt-fs 0.5 \
  --steps 10 \
  --temperature 300 \
  --threads "$SLURM_CPUS_PER_TASK" \
  --memory-mb 56000
