#!/bin/bash
#SBATCH --job-name=aimd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=pc60_static.%j.out
#SBATCH --error=pc60_static.%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pyscf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

time -p python 01_pyscf_aimd.py \
    --xyz ../02_pyscf_static/output_relax/2H2Pc_C60_relaxed.xyz \
    --basis 6-31g \
    --xc pbe-d3bj \
    --grid-level 0 \
    --dt-fs 0.5 \
    --steps 10 \
    --temperature 300 \
    --threads 192 \
    --memory-mb 100000 \
    --conv-tol 1e-6 \
    --outdir output_bench_stable \
    > aimd_bench_stable.log 2>&1
