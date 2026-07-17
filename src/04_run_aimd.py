#!/usr/bin/env python3
"""Run Born-Oppenheimer AIMD for the neutral 2H2Pc/C60 complex."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pyscf import dft, gto, lib, md
from pyscf.md.integrators import NVTBerendson

FS_TO_AU = 41.34137333518211
ROOT = Path(__file__).resolve().parent.parent


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    natoms = int(lines[0])
    rows = [line.split() for line in lines[2 : 2 + natoms]]
    if len(rows) != natoms:
        raise ValueError(f"{path}: expected {natoms} atoms, found {len(rows)}")
    return [(row[0], tuple(float(value) for value in row[1:4])) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, default=ROOT / "data" / "2H2Pc_C60.xyz")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="pbe-d3bj")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--tau-fs", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=56000)
    parser.add_argument("--conv-tol", type=float, default=1e-7)
    parser.add_argument("--outdir", type=Path, default=ROOT / "output_aimd")
    args = parser.parse_args()

    if args.dt_fs <= 0 or args.tau_fs <= 0 or args.steps <= 0:
        parser.error("dt, tau, and steps must be positive")

    args.outdir.mkdir(parents=True, exist_ok=True)
    lib.num_threads(args.threads)
    md.set_seed(args.seed)

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
    mf.conv_tol = args.conv_tol
    mf.conv_tol_grad = max(args.conv_tol**0.5, 3e-4)
    mf.max_cycle = 110
    mf.diis_space = 12
    mf.level_shift = 0.10
    mf.chkfile = str(args.outdir / "aimd.chk")
    mf = mf.density_fit()

    print(f"natm={mol.natm}, nelec={mol.nelectron}, nao={mol.nao_nr()}")
    print(f"method={args.xc}/{args.basis}, threads={lib.num_threads()}")
    print(f"dt={args.dt_fs} fs, steps={args.steps}, total={args.dt_fs * args.steps} fs")

    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Initial SCF did not converge")

    velocity = md.distributions.MaxwellBoltzmannVelocity(mol, T=args.temperature)
    scanner = mf.nuc_grad_method().as_scanner()
    integrator = NVTBerendson(
        scanner,
        T=args.temperature,
        taut=args.tau_fs * FS_TO_AU,
        dt=args.dt_fs * FS_TO_AU,
        steps=args.steps,
        veloc=velocity,
        data_output=str(args.outdir / "aimd.md.data"),
        trajectory_output=str(args.outdir / "aimd.md.xyz"),
        incore_anyway=False,
        verbose=4,
    )
    integrator.run()
    integrator.data_output.close()
    integrator.trajectory_output.close()

    print(f"Trajectory: {args.outdir / 'aimd.md.xyz'}")
    print(f"Energies:   {args.outdir / 'aimd.md.data'}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        if "dftd" in str(exc).lower():
            print("Install pyscf-dispersion or use --xc pbe.", file=sys.stderr)
        raise
