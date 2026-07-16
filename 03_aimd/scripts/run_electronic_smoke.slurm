#!/bin/bash
#SBATCH --job-name=pc60_elec_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=electronic_smoke.%j.log

set -euo pipefail


export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"

/scratch/user/u.aa271894/.conda/envs/pyscf/bin/python scripts/08_extract_electronic_frames.py \
    --trajectory output_production/traj_00/aimd.md.xyz \
    --outdir output_electronic_smoke/traj_00 \
    --start 0 \
    --stop 10 \
    --stride 1 \
    --dt-fs 0.5 \
    --basis 6-31g \
    --xc pbe \
    --grid-level 0 \
    --conv-tol 1e-6 \
    --threads "$SLURM_CPUS_PER_TASK" \
    --memory-mb 55000

