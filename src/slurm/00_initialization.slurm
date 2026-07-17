#!/bin/bash
#SBATCH --job-name=pc60_init
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/initialization_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output_static output_relax
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PYTHON="${PYSCF_PYTHON:-python}"

"${PYTHON}" src/01_frontier_analysis.py \
  --xyz data/2H2Pc_C60.xyz --fragments data/fragments.json \
  --basis 6-31g --xc pbe-d3bj --grid-level 0 \
  --threads "${SLURM_CPUS_PER_TASK}" --memory-mb 95000 \
  --outdir output_static

"${PYTHON}" src/02_relax_geometry.py \
  --xyz data/2H2Pc_C60.xyz --basis 6-31g --xc pbe-d3bj \
  --grid-level 0 --maxsteps "${MAXSTEPS:-20}" \
  --threads "${SLURM_CPUS_PER_TASK}" --memory-mb 95000 \
  --outdir output_relax
