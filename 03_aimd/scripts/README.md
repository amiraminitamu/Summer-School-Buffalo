# Production scripts

These files are numbered in execution order. Each is a command-line program with explicit inputs and outputs; no notebook state is required.

| Step | Script | Required input | Primary output |
|---:|---|---|---|
| 1 | `01_run_aimd.py` | XYZ geometry | `aimd.md.xyz`, energies |
| 2 | `02_extract_electronic.py` | one AIMD trajectory | raw frame NPZ files, initial donor state |
| 3 | `03_track_states.py` | raw frames + trajectory | `tracked_data.npz`, tracking diagnostics |
| 4 | `04_run_fssh.py` | tracked data | `libra_fssh.npz`, populations, summary |
| 5 | `05_analyze_fssh.py` | ten FSSH directories | ensemble CSV/NPZ and confidence intervals |
| 6 | `06_build_reduced_model.py` | tracked ten-state trajectories | validated 4D+3A models |
| 7 | `07_prepare_pca_bath.py` | reduced vibronic Hamiltonians | PCA operators, ACFs, spectra |
| 8 | `08_fit_bath_correlations.py` | PCA bath analysis | compact Drude/Brownian fits |
| 9 | `09_run_tenso.py` | PCA operators + fits | TENSO density-matrix trajectory |
| 10 | `10_plot_libra_experiment.py` | FSSH + tracked projectors + digitized SHG | final comparison figures and metrics |
| 11 | `11_compare_fssh_variants.py` | plain and Boltzmann FSSH | sensitivity figure and metrics |

## Design choices

- The retained files are production stages, not a history of experiments.
- Assertions and diagnostics are kept because they establish state-tracking, propagation, reduction, and density-matrix validity.
- Raw outputs are intentionally excluded from Git; every stage writes structured CSV, JSON, or NPZ artifacts.
- `04_run_fssh.py` uses Kohn-Sham orbital derivative couplings, not TDDFT excited-state couplings.
- `09_run_tenso.py` is exact only for the specified finite reduced Hamiltonian and converged tensor settings.

Run `python SCRIPT --help` before a stage to inspect all options.
