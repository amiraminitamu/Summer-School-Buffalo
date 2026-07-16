#!/bin/bash
#SBATCH --job-name=aimd_bench80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_80.slurm.log

source ~/.bashrc
conda activate pyscf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /ccstar/scratch/group/p.phy230021.000/CyberTraining/Project/Project_Amir_Amini/03_aimd

time -p python 01_pyscf_aimd.py \
    --xyz benchmark_latest.xyz \
    --basis 6-31g \
    --xc pbe-d3bj \
    --grid-level 0 \
    --dt-fs 0.5 \
    --tau-fs 100 \
    --steps 10 \
    --temperature 300 \
    --seed 80080 \
    --threads 80 \
    --memory-mb 90000 \
    --conv-tol 1e-6 \
    --outdir output_benchmark_80 \
    > benchmark_80.log 2>&1
