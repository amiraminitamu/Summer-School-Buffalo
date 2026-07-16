# Grading-criteria map

| Criterion | Evidence in this repository |
|---|---|
| Scientific motivation and objectives | README scientific question and `docs/SCIENTIFIC_METHOD.md` |
| Analysis and interpretation | `docs/RESULTS.md`; plain/Boltzmann comparison; reduced-model and TENSO convergence metrics |
| Supported conclusions | README limitations and explicit non-claims; quantitative RMSE/endpoint values |
| Workflow correctness | state-tracking diagnostics, norm/Hermiticity checks, projector reconstruction, reduction controls |
| Reproducibility | fixed starts/seeds, portable Slurm files, environments, ordered commands in `docs/REPRODUCIBILITY.md` |
| Technical quality | command-line interfaces, structured outputs, deterministic seeds, CI syntax/geometry checks |
| Documentation | root README, script map, method and reproducibility documents |
| Organization | one numbered production pipeline; test/legacy/duplicate files removed |
| Ease of reuse | no hard-coded usernames; environment-variable Python executables; documented input/output contracts |
| End-to-end integration | PySCF -> orbital tracking -> Libra -> reduced model -> PCA bath -> TENSO -> experiment |
| Originality | ab initio replacement of the reference DFTB3 Hamiltonian and explicit method-sensitivity analysis |
| Visual/presentation value | scripts generate publication-ready ensemble, estimator, and experiment-comparison figures |
