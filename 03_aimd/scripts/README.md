# Production and analysis scripts

The retained programs are numbered by execution order. Together they reproduce the final coherent, reduced-model, surface-hopping, and pathway-resolved results.

| Step | Script | Required input | Primary output |
|---:|---|---|---|
| 1 | `01_run_aimd.py` | Starting XYZ geometry | AIMD trajectory and energies |
| 2 | `02_extract_electronic.py` | AIMD trajectory | Ten active orbitals, energies, initial state, C60 projectors |
| 3 | `03_track_states.py` | Frame archives + trajectory | Tracked data, overlaps, derivative couplings, `H_vib` |
| 4 | `04_run_fssh.py` | Tracked data | Coherent amplitudes and Libra FSSH histories |
| 5 | `05_analyze_fssh.py` | Ten FSSH directories | Ensemble statistics and confidence intervals |
| 6 | `06_build_reduced_model.py` | Ten tracked trajectories | Validated 4D+3A models |
| 7 | `10_plot_libra_experiment.py` | FSSH + projectors + experiment | Plain FSSH/experiment figures and metrics |
| 8 | `11_compare_fssh_variants.py` | Plain and Boltzmann FSSH | Detailed-balance sensitivity figures and metrics |
| 9 | `12_analyze_coherent_mechanism.py` | Ten 4D+3A model files | Coherent validation, coherence, and channel fluxes |

No notebook state is required. Run `python SCRIPT --help` for each input/output contract.

“Exact coherent propagation” refers to a matrix exponential within the finite active-space Hamiltonian. No TDDFT calculation is performed.
