#!/bin/bash
#SBATCH --job-name=pc60_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=65
#SBATCH --mem=90G
#SBATCH --time=30:00:00
#SBATCH --array=0-9%10
#SBATCH --output=production_%A_%a.slurm.log

set -euo pipefail


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /ccstar/scratch/group/p.phy230021.000/CyberTraining/Project/Project_Amir_Amini/03_aimd

replica=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
seed=$((2026071400 + SLURM_ARRAY_TASK_ID))

xyz="production_starts/start_${replica}.xyz"
outdir="output_production/traj_${replica}"
log="production_traj_${replica}.log"

mkdir -p output_production

echo "Replica:        ${replica}"
echo "Starting XYZ:   ${xyz}"
echo "Random seed:    ${seed}"
echo "Output:         ${outdir}"
echo "CPUs:           ${SLURM_CPUS_PER_TASK}"
echo "Started:        $(date)"

time -p /scratch/user/u.aa271894/.conda/envs/pyscf/bin/python 01_pyscf_aimd.py \
    --xyz "$xyz" \
    --basis 6-31g \
    --xc pbe-d3bj \
    --grid-level 0 \
    --dt-fs 0.5 \
    --tau-fs 100 \
    --steps 200 \
    --temperature 300 \
    --seed "$seed" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --memory-mb 85000 \
    --conv-tol 1e-6 \
    --outdir "$outdir" \
    > "$log" 2>&1

echo "Finished: $(date)"
