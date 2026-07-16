#!/usr/bin/env python3
"""Extract ten-state PySCF electronic data along one AIMD trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from pyscf import dft, gto, lib


NATOMS = 176
DONOR_STOP = 116       # atoms 0..115: two H2Pc molecules
C60_START = 116        # atoms 116..175: C60
NSTATES = 10


def read_xyz_trajectory(path: Path):
    lines = path.read_text().splitlines()
    block = NATOMS + 2
    nframes = len(lines) // block

    symbols = None
    frames = []

    for iframe in range(nframes):
        chunk = lines[iframe * block:(iframe + 1) * block]
        if len(chunk) != block or int(chunk[0]) != NATOMS:
            raise ValueError(f"Incomplete frame {iframe} in {path}")

        current_symbols = []
        coordinates = []

        for line in chunk[2:]:
            fields = line.split()
            current_symbols.append(fields[0])
            coordinates.append([float(x) for x in fields[1:4]])

        if symbols is None:
            symbols = current_symbols
        elif current_symbols != symbols:
            raise ValueError(f"Atom ordering changed at frame {iframe}")

        frames.append(np.asarray(coordinates))

    return symbols, frames


def atoms(symbols, coordinates, stop=None):
    stop = len(symbols) if stop is None else stop
    return [
        (symbols[i], tuple(float(x) for x in coordinates[i]))
        for i in range(stop)
    ]


def build_mol(atom_list, basis, memory_mb):
    return gto.M(
        atom=atom_list,
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        symmetry=False,
        cart=False,
        max_memory=memory_mb,
        verbose=0,
    )


def build_ks(mol, args):
    mf = dft.RKS(mol, xc=args.xc)
    mf.grids.level = args.grid_level
    mf.grids.prune = dft.gen_grid.nwchem_prune
    mf.small_rho_cutoff = 1e-7
    mf.conv_tol = args.conv_tol
    mf.conv_tol_grad = max(args.conv_tol**0.5, 3e-4)
    mf.max_cycle = 110
    mf.diis_space = 12
    mf.level_shift = 0.10
    mf.verbose = 3
    return mf.density_fit()


def run_scf(mf, dm0, label):
    energy = mf.kernel(dm0=dm0)

    if not mf.converged:
        print(f"{label}: retrying SCF with stronger stabilization.")
        retry_dm = mf.make_rdm1() if mf.mo_coeff is not None else dm0
        mf.level_shift = 0.30
        mf.damp = 0.20
        mf.max_cycle = 200
        energy = mf.kernel(dm0=retry_dm)

    if not mf.converged:
        raise RuntimeError(f"{label}: SCF did not converge")

    return float(energy)


def atom_mask(mol, start, stop):
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    slices = mol.aoslice_by_atom()

    for atom_index in range(start, stop):
        p0, p1 = slices[atom_index, 2], slices[atom_index, 3]
        mask[p0:p1] = True

    return mask


def fragment_projector(active_coeff, overlap, mask):
    """Symmetrized Mulliken projector in the active-MO basis."""
    left = active_coeff.conj().T @ (
        mask[:, None] * (overlap @ active_coeff)
    )
    return 0.5 * (left + left.conj().T)


def make_initial_state(
    symbols,
    coordinates,
    full_mol,
    full_overlap,
    active_coeff,
    args,
):
    donor_mol = build_mol(
        atoms(symbols, coordinates, DONOR_STOP),
        args.basis,
        args.memory_mb,
    )
    donor_mf = build_ks(donor_mol, args)

    print("Running isolated-donor SCF for the initial state.")
    donor_energy = run_scf(donor_mf, None, "isolated donor")
    donor_lumo = int(np.flatnonzero(donor_mf.mo_occ < 1e-8)[0])

    embedded_donor_nao = int(
        full_mol.aoslice_by_atom()[DONOR_STOP - 1, 3]
    )
    if donor_mol.nao_nr() != embedded_donor_nao:
        raise RuntimeError("Isolated and embedded donor AO dimensions differ")

    donor_orbital = np.zeros(full_mol.nao_nr(), dtype=complex)
    donor_orbital[:embedded_donor_nao] = (
        donor_mf.mo_coeff[:, donor_lumo]
    )

    donor_norm = np.vdot(
        donor_orbital,
        full_overlap @ donor_orbital,
    ).real
    donor_orbital /= np.sqrt(donor_norm)

    c0 = active_coeff.conj().T @ full_overlap @ donor_orbital
    capture = float(np.vdot(c0, c0).real)

    np.savez(
        args.outdir / "initial_state.npz",
        donor_energy_hartree=donor_energy,
        donor_lumo_0based=donor_lumo,
        active_capture=capture,
        c0=c0,
        c0_normalized=c0 / np.sqrt(capture),
    )

    print(f"Ten-state donor-LUMO capture: {capture:.10f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument(
        "--xc",
        default="pbe",
        help="D3 is omitted because it does not change the KS orbitals.",
    )
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--conv-tol", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-mb", type=int, default=55000)
    args = parser.parse_args()

    if args.start < 0 or args.stride <= 0:
        parser.error("start must be nonnegative and stride must be positive")

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.outdir / "raw_frames"
    raw_dir.mkdir(exist_ok=True)

    lib.num_threads(args.threads)
    symbols, frames = read_xyz_trajectory(args.trajectory)

    stop = len(frames) if args.stop is None else min(args.stop, len(frames))
    selected = list(range(args.start, stop, args.stride))
    if not selected:
        raise ValueError("No frames were selected")

    metadata = {
        "trajectory": str(args.trajectory),
        "frame_indices": selected,
        "dt_fs": args.dt_fs,
        "basis": args.basis,
        "xc": args.xc,
        "grid_level": args.grid_level,
        "conv_tol": args.conv_tol,
        "nstates": NSTATES,
        "donor_atoms_0based": [0, DONOR_STOP - 1],
        "c60_atoms_0based": [C60_START, NATOMS - 1],
    }
    (args.outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"Selected {len(selected)} frames: {selected[0]}..{selected[-1]}")
    print(f"Method: {args.xc}/{args.basis}, grid {args.grid_level}")
    print(f"Threads: {lib.num_threads()}, memory: {args.memory_mb} MB")

    dm_previous = None

    for position, iframe in enumerate(selected):
        output = raw_dir / f"frame_{iframe:04d}.npz"

        if output.exists():
            print(f"Frame {iframe:04d}: already complete; skipping.")
            dm_previous = None
            continue

        started = time.perf_counter()
        mol = build_mol(
            atoms(symbols, frames[iframe]),
            args.basis,
            args.memory_mb,
        )
        mf = build_ks(mol, args)
        total_energy = run_scf(
            mf,
            dm_previous,
            f"frame {iframe:04d}",
        )
        dm_previous = mf.make_rdm1()

        overlap = mol.intor_symmetric("int1e_ovlp")
        lumo = int(np.flatnonzero(mf.mo_occ < 1e-8)[0])
        active_indices = np.arange(lumo, lumo + NSTATES)
        active_coeff = mf.mo_coeff[:, active_indices]
        active_energy = mf.mo_energy[active_indices]

        c60_projector = fragment_projector(
            active_coeff,
            overlap,
            atom_mask(mol, C60_START, NATOMS),
        )

        orth_error = float(
            np.linalg.norm(
                active_coeff.conj().T @ overlap @ active_coeff
                - np.eye(NSTATES)
            )
        )

        np.savez(
            output,
            frame_index=iframe,
            time_fs=iframe * args.dt_fs,
            total_energy_hartree=total_energy,
            lumo_0based=lumo,
            active_indices=active_indices,
            active_energy_hartree=active_energy,
            active_coeff=active_coeff,
            projector_c60=c60_projector,
            active_orthogonality_error=orth_error,
        )

        if position == 0 and not (args.outdir / "initial_state.npz").exists():
            make_initial_state(
                symbols,
                frames[iframe],
                mol,
                overlap,
                active_coeff,
                args,
            )

        elapsed_min = (time.perf_counter() - started) / 60
        print(
            f"Frame {iframe:04d}: E={total_energy:.10f} Eh, "
            f"LUMO={lumo}, orth.err={orth_error:.2e}, "
            f"wall={elapsed_min:.2f} min"
        )

    print(f"Raw electronic data are in {raw_dir}")


if __name__ == "__main__":
    main()
