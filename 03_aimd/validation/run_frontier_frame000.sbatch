#!/bin/bash
#SBATCH --job-name=frontier000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=frontier_frame000.slurm.log

source ~/.bashrc
conda activate pyscf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /ccstar/scratch/group/p.phy230021.000/CyberTraining/Project/Project_Amir_Amini/03_aimd

time -p python ../02_pyscf_static/00_frontier_analysis.py \
    --xyz test_snapshots/snapshot_199_t099.5fs.xyz \
    --fragments ../01_geometry/fragments.json \
    --basis 6-31g \
    --xc pbe-d3bj \
    --grid-level 0 \
    --threads 96 \
    --memory-mb 100000 \
    --conv-tol 1e-6 \
    --outdir output_frontier_test/frame199 \
    > frontier_frame199.log 2>&1
