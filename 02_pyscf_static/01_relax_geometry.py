#!/usr/bin/env python3
"""Short PySCF/geomeTRIC relaxation of the SI starting geometry.

The SI geometry was optimized with DFTB3. This script removes DFTB from the
actual workflow by relaxing that geometry on a PySCF KS-DFT potential before
production AIMD. Use a small maxsteps value first to measure cost and inspect
whether the stacked complex remains stable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pyscf import dft, gto, lib
from pyscf.geomopt.geometric_solver import optimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_XYZ = ROOT / '01_geometry' / '2H2Pc_C60.xyz'


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    natm = int(lines[0])
    rows = [line.split() for line in lines[2:2 + natm]]
    if len(rows) != natm:
        raise ValueError(f'{path}: expected {natm} atoms, found {len(rows)}')
    return [(r[0], tuple(float(v) for v in r[1:4])) for r in rows]


def write_xyz(mol, path: Path, comment: str):
    coords = mol.atom_coords(unit='Angstrom')
    with path.open('w') as f:
        f.write(f'{mol.natm}\n{comment}\n')
        for symbol, xyz in zip(mol.elements, coords):
            f.write(f'{symbol:<2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', type=Path, default=DEFAULT_XYZ)
    p.add_argument('--basis', default='sto-3g')
    p.add_argument('--xc', default='pbe-d3bj')
    p.add_argument('--grid-level', type=int, default=1)
    p.add_argument('--maxsteps', type=int, default=3,
                   help='Start with 3-5 steps for a wall-time/force sanity test.')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--memory-mb', type=int, default=56000)
    p.add_argument('--outdir', type=Path, default=HERE / 'output_relax')
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    lib.num_threads(args.threads)

    mol = gto.M(
        atom=read_xyz(args.xyz),
        unit='Angstrom',
        basis=args.basis,
        charge=0,
        spin=0,
        symmetry=False,
        cart=False,
        max_memory=args.memory_mb,
        verbose=4,
    )
    mf = dft.RKS(mol, xc=args.xc)
    mf.grids.level = args.grid_level
    mf.grids.prune = dft.gen_grid.nwchem_prune
    mf.small_rho_cutoff = 1e-7
    mf.conv_tol = 1e-7
    mf.conv_tol_grad = 3e-4
    mf.max_cycle = 100
    mf.chkfile = str(args.outdir / 'relax.chk')
    mf = mf.density_fit()

    mol_eq = optimize(
        mf,
        maxsteps=args.maxsteps,
        callback=None,
        convergence_energy=1e-6,
        convergence_grms=3e-4,
        convergence_gmax=4.5e-4,
        convergence_drms=1.2e-3,
        convergence_dmax=1.8e-3,
    )
    out = args.outdir / '2H2Pc_C60_relaxed.xyz'
    write_xyz(mol_eq, out, f'PySCF relaxation: {args.xc}/{args.basis}; maxsteps={args.maxsteps}')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
