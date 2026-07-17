#!/usr/bin/env python3
"""Relax the published structure with PySCF and geomeTRIC."""
from __future__ import annotations

import argparse
from pathlib import Path

from pyscf import dft, gto, lib
from pyscf.geomopt.geometric_solver import optimize

ROOT = Path(__file__).resolve().parent.parent


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    natoms = int(lines[0])
    rows = [line.split() for line in lines[2 : 2 + natoms]]
    if len(rows) != natoms:
        raise ValueError(f"{path}: expected {natoms} atoms, found {len(rows)}")
    return [(row[0], tuple(float(value) for value in row[1:4])) for row in rows]


def write_xyz(mol, path: Path, comment: str) -> None:
    coordinates = mol.atom_coords(unit="Angstrom")
    with path.open("w") as handle:
        handle.write(f"{mol.natm}\n{comment}\n")
        for symbol, xyz in zip(mol.elements, coordinates):
            handle.write(f"{symbol:<2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, default=ROOT / "data" / "2H2Pc_C60.xyz")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="pbe-d3bj")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--maxsteps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=56000)
    parser.add_argument("--outdir", type=Path, default=ROOT / "output_relax")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    lib.num_threads(args.threads)

    mol = gto.M(
        atom=read_xyz(args.xyz),
        unit="Angstrom",
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
    mf.chkfile = str(args.outdir / "relax.chk")
    mf = mf.density_fit()

    relaxed = optimize(
        mf,
        maxsteps=args.maxsteps,
        convergence_energy=1e-6,
        convergence_grms=3e-4,
        convergence_gmax=4.5e-4,
        convergence_drms=1.2e-3,
        convergence_dmax=1.8e-3,
    )
    output = args.outdir / "2H2Pc_C60_relaxed.xyz"
    write_xyz(relaxed, output, f"PySCF relaxation: {args.xc}/{args.basis}; maxsteps={args.maxsteps}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
