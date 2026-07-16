#!/bin/bash
#SBATCH --job-name=tenso_pca
#SBATCH --output=logs/tenso/tenso_pca_%j.log
#SBATCH --error=logs/tenso/tenso_pca_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem=64G

set -euo pipefail


export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

NMODES="${NMODES:-4}"
END_TIME_FS="${END_TIME_FS:-5.0}"
DT_FS="${DT_FS:-0.05}"
DIM="${DIM:-4}"
RANK="${RANK:-8}"
N_LTC="${N_LTC:-1}"
STATIC_INDEX="${STATIC_INDEX:--1}"
VMF_ATOL="${VMF_ATOL:-1e-9}"
PS2_ATOL="${PS2_ATOL:-1e-9}"
ODE_ATOL="${ODE_ATOL:-1e-9}"
ODE_RTOL="${ODE_RTOL:-1e-7}"
MAX_AUXILIARY_RANK="${MAX_AUXILIARY_RANK:-64}"
RENORMALIZE="${RENORMALIZE:-1}"
OUTDIR="${OUTDIR:-output_tenso/pca_diagonal_renorm}"

mkdir -p logs/tenso "${OUTDIR}"

RENORM_ARG=()
if [[ "${RENORMALIZE}" == "1" ]]; then
    RENORM_ARG=(--renormalize)
fi

/scratch/user/u.aa271894/.conda/envs/tenso/bin/python scripts/21_run_tenso_pca_model.py \
    --bath-analysis \
        output_tenso_bath_analysis_hvib/hamiltonian_fluctuation_pca.npz \
    --fits \
        output_tenso_correlation_fits/classical_correlation_fits.npz \
    --outdir "${OUTDIR}" \
    --nmodes "${NMODES}" \
    --temperature 300 \
    --n-ltc "${N_LTC}" \
    --end-time-fs "${END_TIME_FS}" \
    --dt-fs "${DT_FS}" \
    --dim "${DIM}" \
    --rank "${RANK}" \
    --static-index "${STATIC_INDEX}" \
    --vmf-atol "${VMF_ATOL}" \
    --ps2-atol "${PS2_ATOL}" \
    --ode-atol "${ODE_ATOL}" \
    --ode-rtol "${ODE_RTOL}" \
    --max-auxiliary-rank "${MAX_AUXILIARY_RANK}" \
    "${RENORM_ARG[@]}"
