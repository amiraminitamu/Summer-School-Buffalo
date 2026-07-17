# Source code

The numbered programs follow the calculation in execution order.

| Program | Role |
|---|---|
| `00_validate_geometry.py` | Validate atom count, composition, and fragment ordering |
| `01_frontier_analysis.py` | Static frontier orbitals and donor-LUMO projection |
| `02_relax_geometry.py` | Initial PySCF/geomeTRIC relaxation |
| `03_prepare_replicas.py` | Recreate the ten tracked production starts from a thermalization trajectory |
| `04_run_aimd.py` | PySCF Born-Oppenheimer AIMD |
| `05_extract_electronic.py` | Ten-state orbital energies, coefficients, projectors, and initial state |
| `06_track_states.py` | Orbital assignment, phase tracking, and matrix-log derivative couplings |
| `07_run_fssh.py` | Libra CPA-FSSH propagation |
| `08_analyze_fssh.py` | FSSH ensemble aggregation |
| `09_build_reduced_model.py` | Construct and validate the 4D+3A Hamiltonian |
| `10_plot_libra_experiment.py` | Libra estimator and experiment figures |
| `11_compare_fssh_variants.py` | Plain/Boltzmann detailed-balance sensitivity |
| `12_analyze_coherent_mechanism.py` | Coherent dynamics, coherence, and pathway currents |

Portable submission templates are under `src/slurm/`. Comments are limited to formulas and implementation choices that need clarification.
