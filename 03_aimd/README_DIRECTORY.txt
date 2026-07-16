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
