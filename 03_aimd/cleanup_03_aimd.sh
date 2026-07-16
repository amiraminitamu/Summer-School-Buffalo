#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

mkdir -p \
    scripts \
    logs/production \
    logs/equilibration \
    validation \
    archive/pilot_and_benchmarks \
    archive/old_submission_files

move_if_exists() {
    local destination="$1"
    shift

    for item in "$@"; do
        if [[ -e "$item" ]]; then
            echo "Moving $item -> $destination/"
            mv -- "$item" "$destination/"
        fi
    done
}

# ----------------------------------------------------------------------
# Keep: reusable scripts and the successful production submission file
# ----------------------------------------------------------------------

move_if_exists scripts \
    01_pyscf_aimd.py \
    02_extract.py \
    03_track_active_space.py \
    04_find_nacs.py \
    06_propagate_ten_states.py \
    07_structural_qc.py \
    run_10x100fs.slurm

# ----------------------------------------------------------------------
# Keep as validation/proof that the electronic pipeline worked
# ----------------------------------------------------------------------

move_if_exists validation \
    05_propagate_3_state.py \
    test_snapshots \
    tracking_test \
    output_frontier_test \
    output_tracking_test \
    run_frontier_frame000.sbatch \
    run_tracking.sbatch \
    tracking_test.log \
    tracking_test.slurm.log \
    frontier_frame000.log \
    frontier_frame100.log \
    frontier_frame190.log \
    frontier_frame199.log \
    frontier_frame000.slurm.log

# ----------------------------------------------------------------------
# Keep production logs, but compress them
# ----------------------------------------------------------------------

move_if_exists logs/production \
    production_traj_*.log \
    production_*.slurm.log

move_if_exists logs/equilibration \
    aimd_eq_50fs.log

for file in logs/production/*.log logs/equilibration/*.log; do
    echo "Compressing $file"
    gzip -f "$file"
done

# ----------------------------------------------------------------------
# Archive obsolete pilot calculations and benchmarks
# ----------------------------------------------------------------------

move_if_exists archive/pilot_and_benchmarks \
    output_bench_6-31g_96 \
    output_bench_stable \
    output_benchmark_80 \
    output_aimd_100fs \
    aimd_bench_stable.log \
    aimd_100fs.log \
    benchmark_80.log \
    benchmark_latest.xyz \
    2H2Pc_C60_after_100fs.xyz

# ----------------------------------------------------------------------
# Archive old submission scripts and tiny scheduler files
# ----------------------------------------------------------------------

move_if_exists archive/old_submission_files \
    submit_aimd_smoke.slurm \
    submit.slurm \
    submit_100fs.slurm \
    bench.slurm \
    pc60_static.331558.out \
    pc60_static.331558.err \
    benchmark_80.slurm.log \
    aimd_100fs.slurm.log

cat > README_DIRECTORY.txt <<'EOF'
2H2Pc/C60 AIMD directory
========================

output_production/
    Ten accepted 100 fs production trajectories.
    These are the primary nuclear trajectories for electronic analysis.

output_eq_50fs/
    Strong-thermostat equilibration trajectory used to generate the
    production starting geometries.

production_starts/
    Exact starting XYZ geometries and replica manifest.

scripts/
    Reusable AIMD, extraction, state-tracking, NAC, propagation,
    structural-QC, and Slurm scripts.

logs/
    Compressed production and equilibration logs.

validation/
    Successful frontier-orbital, state-tracking, NAC, three-state,
    and ten-state validation calculations.

archive/
    Obsolete pilot AIMD calculations, scaling benchmarks, and older
    submission files. Safe to remove only after the final project is
    backed up.
EOF

echo
echo "Cleanup complete."
echo
du -sh \
    output_production \
    output_eq_50fs \
    production_starts \
    scripts \
    logs \
    validation \
    archive 2>/dev/null || true
