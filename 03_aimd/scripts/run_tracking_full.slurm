#!/bin/bash
#SBATCH --job-name=pc60_track
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tracking/tracking_%A_%a.log

set -euo pipefail


export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

replica=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
nframes="${NFRAMES:-200}"
outroot="${OUTROOT:-output_tracked}"
outdir="${outroot}/traj_${replica}"

mkdir -p "$outdir"

echo "Replica:    $replica"
echo "Frames:     $nframes"
echo "Output:     $outdir"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Started:    $(date)"

/scratch/user/u.aa271894/.conda/envs/pyscf/bin/python -u scripts/09_track_and_pack.py \
    --trajectory "output_production/traj_${replica}/aimd.md.xyz" \
    --raw-dir "output_electronic/traj_${replica}/raw_frames" \
    --initial-state "output_electronic/traj_${replica}/initial_state.npz" \
    --outdir "$outdir" \
    --basis 6-31g \
    --nframes "$nframes" \
    --natoms 176 \
    --threads "$SLURM_CPUS_PER_TASK"

echo "Finished: $(date)"
