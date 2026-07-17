#!/bin/bash
#SBATCH --job-name=pc60_libra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/libra_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
PYTHON="${LIBRA_PYTHON:-python}"
replica=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
outroot="${OUTROOT:-output_libra/fssh}"
extra=()
[[ "${BOLTZMANN:-0}" == "1" ]] && extra+=(--boltzmann)

"${PYTHON}" src/07_run_fssh.py \
  --input "output_tracked/traj_${replica}/tracked_data.npz" \
  --outdir "${outroot}/traj_${replica}" \
  --nframes 200 --nsh "${NSH:-20000}" --substeps "${SUBSTEPS:-200}" \
  --seed "$((2026071500 + SLURM_ARRAY_TASK_ID))" --temperature 300 \
  "${extra[@]}"
