#!/bin/bash
set -euo pipefail

JOB_TABLE="${1:-tenso_campaign_jobs.tsv}"

if [[ ! -f "${JOB_TABLE}" ]]; then
    echo "ERROR: ${JOB_TABLE} not found" >&2
    exit 1
fi

echo "Campaign jobs:"
column -t -s $'\t' "${JOB_TABLE}"

echo
echo "Slurm state:"
ids=$(awk 'NR>1 {print $2}' "${JOB_TABLE}" | paste -sd, -)
squeue -j "${ids}" \
    -o "%.18i %.24j %.2t %.10M %.10l %.6C %R" || true

echo
echo "Completed/failed accounting records:"
sacct -j "${ids}" \
    --format=JobID,JobName%24,State,Elapsed,MaxRSS,ExitCode \
    -X || true
