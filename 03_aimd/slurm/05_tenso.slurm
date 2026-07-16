#!/bin/bash
#SBATCH --job-name=pc60_tenso
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tenso_%j.log
#SBATCH --error=logs/tenso_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs "${OUTDIR:-output_tenso/production}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PYTHON="${TENSO_PYTHON:-python}"
renorm=()
[[ "${RENORMALIZE:-1}" == "1" ]] && renorm+=(--renormalize)
"${PYTHON}" scripts/09_run_tenso.py \
  --bath-analysis output_tenso_bath_analysis_hvib/hamiltonian_fluctuation_pca.npz \
  --fits output_tenso_correlation_fits/classical_correlation_fits.npz \
  --outdir "${OUTDIR:-output_tenso/production}" \
  --nmodes "${NMODES:-12}" --temperature 300 --n-ltc "${N_LTC:-1}" \
  --end-time-fs "${END_TIME_FS:-10.0}" --dt-fs "${DT_FS:-0.05}" \
  --dim "${DIM:-4}" --rank "${RANK:-8}" --static-index "${STATIC_INDEX:--1}" \
  --vmf-atol 1e-9 --ps2-atol 1e-9 --ode-atol 1e-9 --ode-rtol 1e-7 \
  --max-auxiliary-rank "${MAX_AUXILIARY_RANK:-64}" "${renorm[@]}"
