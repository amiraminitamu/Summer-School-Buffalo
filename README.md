# First-principles charge-transfer dynamics in 2H2Pc/C60

This repository is the submission-ready implementation of a **purely theoretical and computational chemistry capstone** completed for the 2026 CyberTraining Summer School at the University at Buffalo. Despite the program name, this project contains **no biological modeling and no cybersecurity component**.

The scientific objective is to build an end-to-end, auditable workflow for photoinduced electron transfer from a free-base phthalocyanine dimer donor (`2H2Pc`) to a fullerene acceptor (`C60`):

```text
PySCF AIMD
   -> snapshot KS orbitals and fragment projectors
   -> cross-geometry orbital tracking
   -> Libra classical-path FSSH
   -> validated 4-donor + 3-acceptor reduction
   -> PCA bath extraction and correlation fitting
   -> TENSO open-system propagation
   -> comparison with digitized SHG experiment
```

## Scientific question

How sensitive is the predicted early-time C60 charge population to the electronic-structure Hamiltonian, surface-hopping detailed-balance prescription, and reduced open-system treatment?

This is deliberately not a curve-fitting exercise. The calculations use a PySCF-derived Hamiltonian and report both agreement and disagreement with experiment.

## Main findings

- Ten independent 100 fs PySCF AIMD trajectories were generated at 300 K, giving 2,000 electronic snapshots.
- The tracked ten-orbital subspace is numerically smooth: the minimum consecutive-overlap singular value is `0.999580`.
- At 99.5 fs, plain CPA-FSSH gives a paper-style hybrid C60 population of `0.280850`, while the approximate digitized SHG trace is `0.608567`.
- Boltzmann-rescaled upward hops give `0.898903` at 99.5 fs. The experiment therefore lies between the two FSSH treatments; the rescaling is not automatically transferable from the reference DFTB3 Hamiltonian to the present PBE Hamiltonian.
- The selected 4D+3A reduced model reproduces the full ten-state ensemble with RMSE `0.01830`.
- Twelve PCA bath modes retain `90.764%` of the dynamic Hamiltonian-fluctuation variance.
- Four-mode TENSO propagation is converged at 5 fs to approximately `0.016691` C60 population when the auxiliary-rank ceiling is 64.

The supported conclusion is that the transfer kinetics are highly method-sensitive. The project does **not** claim exact reproduction of the experimental trace or numerically exact 100 fs dynamics for the full 176-atom system.

## Repository map

| Path | Purpose |
|---|---|
| `01_geometry/` | Published 176-atom structure, fragment map, and geometry validation |
| `02_pyscf_static/` | Static frontier-orbital and geometry-relaxation calculations |
| `03_aimd/production_starts/` | Exact starting geometries and replica manifest |
| `03_aimd/scripts/` | Eleven ordered production and analysis entry points |
| `03_aimd/slurm/` | Portable Slurm launchers without usernames or absolute cluster paths |
| `docs/SCIENTIFIC_METHOD.md` | Equations, approximations, and observable definitions |
| `docs/REPRODUCIBILITY.md` | Environment setup and complete run order |
| `docs/RUBRIC_MAP.md` | Where each grading criterion is demonstrated |
| `docs/RESULTS.md` | Quantitative interpretation and limitations |
| `results/` | Compact machine-readable headline results |

Testing scripts, obsolete submission files, pilot geometries, duplicate figure scripts, raw trajectories, checkpoints, and large archives were intentionally removed from this branch. The retained scripts are the minimum coherent production pipeline plus the diagnostics needed to defend its numerical correctness.

## Quick start

```bash
git clone https://github.com/amiraminitamu/Summer-School-Buffalo.git
cd Summer-School-Buffalo
git switch submission-ready

conda env create -f environment-pyscf.yml
conda activate pc60-pyscf
python 01_geometry/validate_geometry.py
python 02_pyscf_static/00_frontier_analysis.py --basis sto-3g --threads 4
```

The smoke test validates installation and data flow. Production calculations are HPC workloads and require separate PySCF, Libra, and TENSO environments.

## Ordered workflow

Run computational stages from `03_aimd/` so documented relative paths remain valid.

1. `scripts/01_run_aimd.py` — Born-Oppenheimer PySCF AIMD.
2. `scripts/02_extract_electronic.py` — ten frontier orbitals, energies, initial donor state, and C60 projectors.
3. `scripts/03_track_states.py` — Hungarian assignment, phase correction, polar overlap, matrix-log orbital derivative couplings, and `H_vib`.
4. `scripts/04_run_fssh.py` — exact propagation of the 10x10 piecewise-constant TDSE plus Libra FSSH probabilities.
5. `scripts/05_analyze_fssh.py` — trajectory ensemble statistics and confidence intervals.
6. `scripts/06_build_reduced_model.py` — construct and validate the selected 4D+3A model.
7. `scripts/07_prepare_pca_bath.py` — expand traceless Hermitian fluctuations and perform PCA.
8. `scripts/08_fit_bath_correlations.py` — fit Drude plus damped-Brownian classical correlations.
9. `scripts/09_run_tenso.py` — map the fitted bath to TENSO and propagate the reduced density matrix.
10. `scripts/10_plot_libra_experiment.py` — coherent, active-surface, and paper-style hybrid observables.
11. `scripts/11_compare_fssh_variants.py` — plain versus Boltzmann-rescaled FSSH sensitivity.

See `03_aimd/scripts/README.md` for input/output contracts and `docs/REPRODUCIBILITY.md` for commands.

## Important terminology

No TDDFT calculation was performed. The nonadiabatic quantities are **Kohn-Sham orbital time-derivative couplings** obtained from cross-geometry orbital overlaps,

\[
O_{ij}^{(n)}=\langle\psi_i(R_n)|\psi_j(R_{n+1})\rangle,
\qquad
D^{(n)}=\frac{1}{\Delta t}\log U^{(n)},
\]

where `U` is the closest unitary polar factor after state assignment and phase correction. The propagated single-particle vibronic Hamiltonian is

\[
H_{\mathrm{vib}}=E_{\mathrm{mid}}-iD
\]

in atomic units. These are not many-electron excited-state TDDFT couplings.

## Experimental comparison

`03_aimd/experiment_shg_digitized.csv` is an approximate digitization of the black SHG curve shown in Figure 3A of Yamijala and Huo, *J. Phys. Chem. A* **2021**, 125, 628–635. It is not the original raw experimental dataset. All reported comparisons are restricted to the simulated 0–99.5 fs interval and use the digitized values without arbitrary rescaling.

## Reproducibility boundary

The repository stores source code, exact starting structures, metadata, compact results, and scheduler templates. Multi-gigabyte trajectories, frame-resolved arrays, checkpoints, and tensor-network working files are excluded from Git. The output contracts documented in `03_aimd/scripts/README.md` make each omitted artifact regenerable.

## Citation

Please cite the reference system paper and this repository metadata in `CITATION.cff`. The software is released under the MIT License; the published molecular coordinates and digitized literature curve retain their original source attribution.
