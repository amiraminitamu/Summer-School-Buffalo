#!/bin/bash
#SBATCH --job-name=pc60_tracking
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/tracking_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output_tracked
PYTHON="${PYSCF_PYTHON:-python}"
replica=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")

"${PYTHON}" src/06_track_states.py \
  --trajectory "output_production/traj_${replica}/aimd.md.xyz" \
  --raw-dir "output_electronic/traj_${replica}/raw_frames" \
  --initial-state "output_electronic/traj_${replica}/initial_state.npz" \
  --outdir "output_tracked/traj_${replica}" \
  --basis 6-31g --nframes 200 --natoms 176 \
  --threads "${SLURM_CPUS_PER_TASK}"
