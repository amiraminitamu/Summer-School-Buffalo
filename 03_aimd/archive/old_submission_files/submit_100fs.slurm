#!/bin/bash
#SBATCH --job-name=aimd100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=aimd_100fs.slurm.log

source ~/.bashrc
conda activate pyscf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /ccstar/scratch/group/p.phy230021.000/CyberTraining/Project/Project_Amir_Amini/03_aimd

time -p python 01_pyscf_aimd.py \
    --xyz 2H2Pc_C60_after_100fs.xyz \
    --basis 6-31g \
    --xc pbe-d3bj \
    --grid-level 0 \
    --dt-fs 0.5 \
    --tau-fs 10 \
    --steps 100 \
    --temperature 300 \
    --seed 20260714 \
    --threads 192 \
    --memory-mb 100000 \
    --conv-tol 1e-6 \
    --outdir output_eq_50fs \
    > aimd_eq_50fs.log 2>&1
