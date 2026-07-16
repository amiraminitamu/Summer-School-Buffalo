#!/bin/bash
set -euo pipefail

RUNNER="${RUNNER:-scripts/run_tenso_pca.slurm}"
CPUS="${CPUS:-32}"
MEM="${MEM:-64G}"
WALLTIME="${WALLTIME:-08:00:00}"
ROOT_DEP="${ROOT_DEP:-}"

mkdir -p logs/tenso
JOB_TABLE="tenso_campaign_jobs.tsv"
printf "label\tjobid\tdependency\toutdir\n" > "${JOB_TABLE}"

if [[ ! -f "${RUNNER}" ]]; then
    echo "ERROR: cannot find ${RUNNER}" >&2
    exit 1
fi

submit_job() {
    local label="$1"
    local dependency="$2"
    local exports="$3"
    local outdir="$4"

    local dep_args=()
    if [[ -n "${dependency}" ]]; then
        dep_args=(--dependency="${dependency}")
    fi

    local jid
    jid=$(sbatch --parsable \
        --job-name="${label}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${WALLTIME}" \
        --output="logs/tenso/${label}_%j.log" \
        --error="logs/tenso/${label}_%j.err" \
        "${dep_args[@]}" \
        --export="ALL,${exports},OUTDIR=${outdir}" \
        "${RUNNER}")

    printf "%s\t%s\t%s\t%s\n" \
        "${label}" "${jid}" "${dependency:-none}" "${outdir}" \
        | tee -a "${JOB_TABLE}" >&2

    printf "%s" "${jid}"
}

ROOT=""
if [[ -n "${ROOT_DEP}" ]]; then
    ROOT="afterany:${ROOT_DEP}"
fi

COMMON="STATIC_INDEX=-1,VMF_ATOL=1e-9,PS2_ATOL=1e-9,ODE_ATOL=1e-9,ODE_RTOL=1e-7,MAX_AUXILIARY_RANK=64,RENORMALIZE=1"

echo "Submitting stage A: numerical and bath-mode preflight jobs..." >&2

J_4_DIM=$(submit_job \
    "t4_dim5_r12_5fs" "${ROOT}" \
    "${COMMON},NMODES=4,END_TIME_FS=5.0,DT_FS=0.05,DIM=5,RANK=12,N_LTC=1" \
    "output_tenso/campaign/4m_dim5_rank12_5fs")

J_4_LTC=$(submit_job \
    "t4_ltc2_5fs" "${ROOT}" \
    "${COMMON},NMODES=4,END_TIME_FS=5.0,DT_FS=0.05,DIM=4,RANK=8,N_LTC=2" \
    "output_tenso/campaign/4m_ltc2_5fs")

J_4_DT=$(submit_job \
    "t4_dt025_5fs" "${ROOT}" \
    "${COMMON},NMODES=4,END_TIME_FS=5.0,DT_FS=0.025,DIM=4,RANK=8,N_LTC=1" \
    "output_tenso/campaign/4m_dt0025_5fs")

J_8_PRE=$(submit_job \
    "t8_pre_1fs" "${ROOT}" \
    "${COMMON},NMODES=8,END_TIME_FS=1.0,DT_FS=0.05,DIM=4,RANK=8,N_LTC=1" \
    "output_tenso/campaign/8m_preflight_1fs")

J_12_PRE=$(submit_job \
    "t12_pre_1fs" "${ROOT}" \
    "${COMMON},NMODES=12,END_TIME_FS=1.0,DT_FS=0.05,DIM=4,RANK=8,N_LTC=1" \
    "output_tenso/campaign/12m_preflight_1fs")

echo "Submitting stage B: common 5 fs bath-mode comparison..." >&2

J_8_5=$(submit_job \
    "t8_5fs" "afterok:${J_8_PRE}" \
    "${COMMON},NMODES=8,END_TIME_FS=5.0,DT_FS=0.05,DIM=4,RANK=8,N_LTC=1" \
    "output_tenso/campaign/8m_5fs")

J_12_5=$(submit_job \
    "t12_5fs" "afterok:${J_12_PRE}" \
    "${COMMON},NMODES=12,END_TIME_FS=5.0,DT_FS=0.05,DIM=4,RANK=8,N_LTC=1" \
    "output_tenso/campaign/12m_5fs")

echo "Submitting stage C: longer curves, gated by successful convergence jobs..." >&2

J_4_20=$(submit_job \
    "t4_20fs" "afterok:${J_4_DIM}:${J_4_LTC}:${J_4_DT}" \
    "${COMMON},NMODES=4,END_TIME_FS=20.0,DT_FS=0.05,DIM=5,RANK=12,N_LTC=2" \
    "output_tenso/campaign/4m_20fs")

J_8_10=$(submit_job \
    "t8_10fs" "afterok:${J_8_5}" \
    "${COMMON},NMODES=8,END_TIME_FS=10.0,DT_FS=0.05,DIM=4,RANK=8,N_LTC=1" \
    "output_tenso/campaign/8m_10fs")

J_12_10=$(submit_job \
    "t12_10fs" "afterok:${J_12_5}" \
    "${COMMON},NMODES=12,END_TIME_FS=10.0,DT_FS=0.05,DIM=4,RANK=8,N_LTC=1" \
    "output_tenso/campaign/12m_10fs")

echo >&2
echo "All jobs are queued." >&2
echo "Saved job IDs to ${JOB_TABLE}" >&2
echo >&2
echo "Monitor with:" >&2
echo "  squeue -u \$USER -o '%.18i %.24j %.2t %.10M %.10l %R'" >&2
echo "  column -t -s \$'\\t' ${JOB_TABLE}" >&2
