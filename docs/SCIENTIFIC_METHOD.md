# Scientific method and approximations

## Electronic structure and nuclear sampling

The neutral 176-atom complex is propagated with restricted Kohn-Sham PBE-D3BJ/6-31G Born-Oppenheimer AIMD. Dispersion is included for nuclear forces because the donor-acceptor stack is noncovalently bound. Frame-resolved orbital calculations use PBE/6-31G; the D3 correction changes the energy and gradient but not the Kohn-Sham orbitals used here.

Ten independent 100 fs trajectories are sampled at 300 K with 0.5 fs nuclear steps. The first ten unoccupied orbitals define the active single-particle space. The initial state is the isolated donor-dimer LUMO projected into this active space.

## Orbital tracking

For adjacent geometries, cross-basis AO overlaps are evaluated and transformed into the active MO space. States are assigned with the Hungarian algorithm by maximizing total absolute overlap. Real-orbital phases are corrected, and the closest unitary polar factor `U` is used to define

`D = log(U) / dt` and `H_vib = E_mid - i D` in atomic units.

The anti-Hermiticity of `D`, Hermiticity of `H_vib`, overlap singular values, and assignment quality are written for every interval. These are Kohn-Sham orbital derivative couplings, not many-electron TDDFT nonadiabatic couplings.

## Libra dynamics and observables

The electronic amplitudes are advanced by an exact matrix exponential for each piecewise-constant 10x10 Hamiltonian interval. Libra supplies active-state-specific Tully FSSH probabilities. Nuclear paths are prescribed, so this is classical-path/NBRA-style FSSH with no force feedback or momentum rescaling.

Three C60 observables are distinguished:

1. coherent expectation `c^dagger P_C60 c`;
2. active-surface diagonal average;
3. the reference-paper hybrid estimator, which uses FSSH populations on the density-matrix diagonal and coherent amplitudes off diagonal.

The hybrid estimator is used for the closest comparison with Figure 3A of the reference paper. Plain and Boltzmann-rescaled upward-hop variants are both reported as a sensitivity analysis.

## Reduced model

The C60 projector partitions the ten-state space into four donor and six acceptor directions. The retained seven-dimensional model contains the complete four-state donor subspace and the lower three acceptor states. Fragment subspaces are parallel transported before projecting the Hamiltonian and projector. Full and reduced propagations are compared trajectory by trajectory; the ensemble RMSE is the reduction quality criterion.

## Bath model and TENSO

The traceless within-trajectory fluctuations of the 7x7 vibronic Hamiltonian are expanded in an orthonormal generalized Gell-Mann basis. PCA defines noncommuting system-bath operators. Per-mode classical autocorrelations are fitted to a Drude relaxation plus one damped Brownian band. Static between-trajectory offsets are kept separate from the dynamic bath.

TENSO receives the reduced system Hamiltonian, PCA operators, and fitted spectral parameters. Its result is numerically exact only within the finite reduced Hamiltonian, chosen mode truncation, correlation mapping, local dimensions, and tensor-rank convergence—not for the full molecular Hilbert space.
