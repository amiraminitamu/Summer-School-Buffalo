#!/bin/bash
#SBATCH --job-name=pc60_elec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=65
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --output=logs/electronic/electronic_%A_%a.log

set -euo pipefail


export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

replica=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
trajectory="output_production/traj_${replica}/aimd.md.xyz"
outdir="output_electronic/traj_${replica}"

mkdir -p output_electronic

# For trajectory 00, reuse the completed smoke-test frames.
if [[ "$replica" == "00" \
      && -d output_electronic_smoke/traj_00 \
      && ! -e "$outdir" ]]; then
    echo "Reusing trajectory-00 smoke-test data."
    mv output_electronic_smoke/traj_00 "$outdir"
fi

echo "Replica:    $replica"
echo "Trajectory: $trajectory"
echo "Output:     $outdir"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Started:    $(date)"

/scratch/user/u.aa271894/.conda/envs/pyscf/bin/python -u scripts/08_extract_electronic_frames.py \
    --trajectory "$trajectory" \
    --outdir "$outdir" \
    --start 0 \
    --stop 200 \
    --stride 1 \
    --dt-fs 0.5 \
    --basis 6-31g \
    --xc pbe \
    --grid-level 0 \
    --conv-tol 1e-6 \
    --threads "$SLURM_CPUS_PER_TASK" \
    --memory-mb 95000

echo "Finished: $(date)"
