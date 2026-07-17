# Scientific method and definitions

## Electronic structure and nuclear paths

Ten independent 100 fs ground-state Born-Oppenheimer AIMD trajectories are generated for the neutral 176-atom complex at 300 K. The production electronic calculations use PBE/6-31G Kohn-Sham orbitals along each prescribed nuclear path.

## Active space and fragment observable

The active space consists of the first ten unoccupied Kohn-Sham orbitals. The initial electronic state is the isolated donor-dimer LUMO projected into this space. A symmetrized Mulliken projector defines the C60 charge population,

\[
P_{C_{60}}(t)=\mathbf c^\dagger(t)\mathbf P_{C_{60}}(t)\mathbf c(t).
\]

## State tracking

For adjacent nuclear geometries,

\[
O_{ij}^{(n)}=\langle\psi_i(R_n)|\psi_j(R_{n+1})\rangle.
\]

A Hungarian assignment maximizes the total absolute overlap. Orbital phases are fixed by the matched diagonal. The closest-unitary polar factor `U` is used to define

\[
\mathbf D_{n+1/2}=\frac{1}{\Delta t}\log \mathbf U_n,
\qquad
\mathbf H_{\mathrm{vib},n+1/2}=\mathbf E_{n+1/2}-i\mathbf D_{n+1/2}.
\]

These are Kohn-Sham orbital time-derivative couplings, not many-electron TDDFT nonadiabatic couplings.

## Coherent dynamics

For each piecewise-constant midpoint Hamiltonian,

\[
\mathbf c(t+\Delta t)=
\exp[-i\mathbf H_{\mathrm{vib}}\Delta t]\mathbf c(t).
\]

“Exact” refers to the matrix exponential within the finite ten-state Hamiltonian, not to exact molecular quantum dynamics.

## Surface hopping

Libra evaluates classical-path fewest-switches hopping probabilities for stochastic active-surface histories. The plain calculation uses the standard hopping probabilities. The sensitivity calculation multiplies thermally uphill hops by

\[
\exp[-(E_j-E_i)/(k_BT)], \qquad E_j>E_i.
\]

The coherent amplitudes are identical in the two variants; only the stochastic active-state populations differ.

## Reduced model

The retained seven-dimensional space contains the complete four-state donor subspace and the lowest three-state C60 subspace. The fragment bases are parallel transported, and the reduced Hamiltonian is constructed independently using the same overlap/polar/matrix-log procedure.

## Probability-current analysis

For donor state `d` and acceptor state `a`, the instantaneous forward current is

\[
J_{d\rightarrow a}(t)=
2\,\mathrm{Im}\left[H_{ad}(t)c_d(t)c_a^*(t)\right].
\]

The signed integral measures net transfer. The positive integral measures total forward activity, and the absolute integral quantifies bidirectional exchange. Consequently, a channel can be dynamically dominant while contributing little net population because of recrossing.

## Statistical reporting

All coherent and reduced-model ensemble means use ten independent nuclear trajectories. Reported uncertainty bands are 95% confidence intervals computed as `1.96 s / sqrt(10)`.
