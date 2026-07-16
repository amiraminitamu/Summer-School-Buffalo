# 2H2Pc/C60 photoinduced charge-transfer project

This project studies **system A** from Yamijala and Huo: a neutral phthalocyanine dimer donor (`2H2Pc`) in contact with a `C60` acceptor. It is a theoretical/computational charge-transport project using PySCF, Libra, and TENSO.

## Geometry provenance

`01_geometry/2H2Pc_C60.xyz` was extracted verbatim from pages S4-S9 of the Supporting Information.

Atom ordering:

- atoms 1-58: H2Pc farther from C60
- atoms 59-116: H2Pc in direct contact with C60
- atoms 117-176: C60

The fragment formulas are `C32H18N8 + C32H18N8 + C60`, for 176 atoms and 892 electrons.

Validate the extraction:

```bash
python 01_geometry/validate_geometry.py
```

## Stage 1: static PySCF test

The first script calculates the isolated donor and full complex with the same method, projects the donor LUMO into the complex MO basis, and reports fragment weights for the full-system LUMO through LUMO+9.

Start with a cheap plumbing test:

```bash
cd 02_pyscf_static
python 00_frontier_analysis.py --basis sto-3g --threads 8
```

Then run the first chemically serious calculation:

```bash
python 00_frontier_analysis.py \
  --basis def2-svp \
  --xc pbe-d3bj \
  --grid-level 1 \
  --threads 32
```

Outputs include:

- `frontier_LUMO_to_LUMO9.csv`
- `frontier_LUMO_to_LUMO9.molden`
- `frontier_data.npz`
- donor and complex PySCF checkpoint files

The key diagnostic is `projection_norm_LUMO_to_LUMO9`. If it is small, the ten-orbital active space used by the paper is not adequate at our chosen electronic-structure level.


## Stage 1b: remove the DFTB geometry from the production workflow

The SI coordinate file is only the starting geometry; it was optimized at DFTB3. Before production AIMD, relax it with PySCF:

```bash
cd 02_pyscf_static
python 01_relax_geometry.py --basis sto-3g --maxsteps 3 --threads 8
```

After timing that test, use `def2-svp` and more optimization steps. The relaxed XYZ can then be passed to AIMD with `--xyz output_relax/2H2Pc_C60_relaxed.xyz`.

## Stage 2: PySCF AIMD

The AIMD script uses Born-Oppenheimer PySCF energies and analytic gradients. The default NVT integrator is PySCF's Berendsen thermostat with Maxwell-Boltzmann initial velocities.

Ten-step smoke test:

```bash
cd 03_aimd
python 01_pyscf_aimd.py --basis sto-3g --steps 10 --threads 8
```

After measuring the cost per force evaluation, try a 100 fs pilot:

```bash
python 01_pyscf_aimd.py \
  --basis def2-svp \
  --xc pbe-d3bj \
  --dt-fs 0.5 \
  --steps 200 \
  --temperature 300 \
  --threads 32
```

`PBE-D3BJ` is chosen for the nuclear trajectory because dispersion is essential for this stacked donor-acceptor complex. We should not use a plain semilocal functional without dispersion for production dynamics.

## Why the workflow separates nuclear and electronic levels

A 176-atom, 892-electron AIMD trajectory is expensive. The practical workflow is:

1. PySCF PBE-D3BJ AIMD for nuclear configurations.
2. Higher-quality snapshot electronic calculations and orbital tracking for the charge-transfer subspace.
3. Libra CPA-FSSH on the snapshot Hamiltonians/overlaps.
4. A reduced vibronic Hamiltonian propagated with both Libra and TENSO.
5. Compare C60 population curves with the experimental SHG curve.

TENSO is numerically exact only for the converged reduced vibronic Hamiltonian, not for all 3N-6 modes of the full 176-atom complex.

## Immediate milestone

Do not launch a long trajectory yet. First obtain:

1. a converged donor calculation;
2. a converged complex calculation;
3. donor/C60 character of LUMO-LUMO+9;
4. the donor-LUMO projection captured by that active space;
5. wall time for one PySCF force/gradient evaluation.

Those five numbers determine the viable production method and trajectory length.
