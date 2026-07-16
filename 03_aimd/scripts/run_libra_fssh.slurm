#!/bin/bash
#SBATCH --job-name=pc60_libra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --output=logs/libra/libra_%A_%a.log

set -euo pipefail

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

replica=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
nframes="${NFRAMES:-200}"
nsh="${NSH:-20000}"
substeps="${SUBSTEPS:-20}"
outroot="${OUTROOT:-output_libra/fssh}"
boltzmann="${BOLTZMANN:-0}"
seed=$((2026071500 + SLURM_ARRAY_TASK_ID))

extra=()
if [[ "$boltzmann" == "1" ]]; then
    extra+=(--boltzmann)
fi

outdir="${outroot}/traj_${replica}"
mkdir -p "$outdir"

echo "Replica:    $replica"
echo "Frames:     $nframes"
echo "Walkers:    $nsh"
echo "Substeps:   $substeps"
echo "Boltzmann:  $boltzmann"
echo "Output:     $outdir"
echo "Started:    $(date)"

/scratch/user/u.aa271894/.conda/envs/libra/bin/python -u scripts/10_run_libra_fssh.py \
    --input "output_tracked/traj_${replica}/tracked_data.npz" \
    --outdir "$outdir" \
    --nframes "$nframes" \
    --nsh "$nsh" \
    --substeps "$substeps" \
    --seed "$seed" \
    --temperature 300 \
    "${extra[@]}"

echo "Finished: $(date)"
