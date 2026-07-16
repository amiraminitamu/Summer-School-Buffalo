#!/usr/bin/env python3
"""Track LUMO...LUMO+9 through consecutive AIMD geometries."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from pyscf import dft, gto, lib


HARTREE_TO_EV = 27.211386245988


def read_xyz(path: Path):
    """Return PySCF atom specification from one XYZ file."""
    lines = path.read_text().splitlines()
    natoms = int(lines[0])

    atoms = []
    for line in lines[2:2 + natoms]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed XYZ line in {path}: {line}")

        symbol = fields[0]
        xyz = tuple(float(x) for x in fields[1:4])
        atoms.append((symbol, xyz))

    if len(atoms) != natoms:
        raise ValueError(f"Expected {natoms} atoms in {path}, found {len(atoms)}")

    return atoms


def frame_metadata(path: Path):
    """Extract frame index and time from frame_190_t095.0fs.xyz."""
    match = re.search(r"frame_(\d+)_t([0-9.]+)fs", path.stem)
    if not match:
        raise ValueError(f"Cannot parse frame metadata from {path.name}")

    return int(match.group(1)), float(match.group(2))


def build_molecule(atoms, basis: str, memory_mb: int):
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry=False,
        verbose=3,
        max_memory=memory_mb,
    )

    if mol.natm != 176:
        raise ValueError(f"Expected 176 atoms, found {mol.natm}")

    return mol


def run_scf(
    mol,
    xc: str,
    grid_level: int,
    conv_tol: float,
    chkfile: Path,
    dm0=None,
):
    """Run the full-complex PBE calculation for one frame."""
    mf = dft.RKS(mol).density_fit()
    mf.xc = xc
    mf.grids.level = grid_level
    mf.conv_tol = conv_tol
    mf.max_cycle = 110
    mf.diis_space = 12
    mf.level_shift = 0.10
    mf.chkfile = str(chkfile)

    energy = mf.kernel(dm0=dm0)

    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {chkfile.stem}")

    return mf, energy


def ao_mask_for_atoms(mol, atom_start: int, atom_stop: int):
    """Boolean AO mask for atoms [atom_start, atom_stop)."""
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    slices = mol.aoslice_by_atom()

    for atom in range(atom_start, atom_stop):
        ao_start, ao_stop = slices[atom, 2:4]
        mask[ao_start:ao_stop] = True

    return mask


def fragment_weights(mol, coefficients):
    """
    Mulliken-like gross orbital populations.

    Atom order:
      0:58    far H2Pc
      58:116  near H2Pc
      116:176 C60
    """
    overlap = mol.intor_symmetric("int1e_ovlp")
    sc = overlap @ coefficients

    far_mask = ao_mask_for_atoms(mol, 0, 58)
    near_mask = ao_mask_for_atoms(mol, 58, 116)
    c60_mask = ao_mask_for_atoms(mol, 116, 176)

    def gross(mask):
        return np.einsum(
            "pi,pi->i",
            coefficients[mask].conj(),
            sc[mask],
        ).real

    far = gross(far_mask)
    near = gross(near_mask)
    c60 = gross(c60_mask)

    # Remove tiny numerical normalization errors.
    total = far + near + c60
    far /= total
    near /= total
    c60 /= total

    return far, near, c60


def save_matrix(path: Path, matrix: np.ndarray):
    header = ",".join(f"state_{i}" for i in range(matrix.shape[1]))
    np.savetxt(path, matrix, delimiter=",", header=header, comments="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="tracking_test")
    parser.add_argument("--pattern", default="frame_*.xyz")
    parser.add_argument("--outdir", default="output_tracking_test")
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--nactive", type=int, default=10)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--memory-mb", type=int, default=100000)
    parser.add_argument("--conv-tol", type=float, default=1e-6)
    args = parser.parse_args()

    lib.num_threads(args.threads)

    frames_dir = Path(args.frames_dir)
    paths = sorted(frames_dir.glob(args.pattern))

    if len(paths) < 2:
        raise ValueError(f"Found only {len(paths)} matching XYZ files")

    outdir = Path(args.outdir)
    chkdir = outdir / "chk"
    overlap_dir = outdir / "overlaps"
    npzdir = outdir / "frames"

    for directory in (outdir, chkdir, overlap_dir, npzdir):
        directory.mkdir(parents=True, exist_ok=True)

    state_rows = []
    pair_rows = []

    previous = None
    dm_guess = None

    print(f"Found {len(paths)} consecutive frames")
    print(f"Electronic method: {args.xc}/{args.basis}")
    print(f"Active space: {args.nactive} virtual orbitals\n")

    for position, path in enumerate(paths):
        frame, time_fs = frame_metadata(path)
        atoms = read_xyz(path)
        mol = build_molecule(atoms, args.basis, args.memory_mb)

        print("=" * 72)
        print(f"Frame {frame}: t = {time_fs:.1f} fs")
        print("=" * 72)

        mf, total_energy = run_scf(
            mol=mol,
            xc=args.xc,
            grid_level=args.grid_level,
            conv_tol=args.conv_tol,
            chkfile=chkdir / f"frame_{frame:03d}.chk",
            dm0=dm_guess,
        )

        # Use this converged density as the next frame's initial guess.
        dm_guess = mf.make_rdm1()

        occupied = np.flatnonzero(mf.mo_occ > 0)
        homo = int(occupied[-1])
        lumo = homo + 1

        canonical_indices = np.arange(lumo, lumo + args.nactive)
        canonical_coeff = mf.mo_coeff[:, canonical_indices].copy()
        canonical_energy = mf.mo_energy[canonical_indices].copy()

        far, near, c60 = fragment_weights(mol, canonical_coeff)

        if previous is None:
            permutation = np.arange(args.nactive)
            phases = np.ones(args.nactive)
            tracked_coeff = canonical_coeff.copy()

        else:
            # AO overlap between the previous and current geometries.
            cross_ao = gto.intor_cross(
                "int1e_ovlp",
                previous["mol"],
                mol,
            )

            # Previous tracked states versus current canonical states.
            preassignment = (
                previous["coeff"].conj().T
                @ cross_ao
                @ canonical_coeff
            )
            preassignment = np.real_if_close(preassignment).real

            # Match states by maximizing the total absolute overlap.
            rows, columns = linear_sum_assignment(-np.abs(preassignment))

            permutation = np.empty(args.nactive, dtype=int)
            permutation[rows] = columns

            assigned = preassignment[:, permutation]

            # Correct arbitrary MO signs so matched diagonal overlaps are positive.
            phases = np.where(np.diag(assigned) >= 0.0, 1.0, -1.0)
            tracked_coeff = canonical_coeff[:, permutation] * phases
            tracked_overlap = assigned * phases[np.newaxis, :]

            previous_frame = previous["frame"]

            save_matrix(
                overlap_dir
                / f"overlap_{previous_frame:03d}_{frame:03d}_preassignment.csv",
                preassignment,
            )
            save_matrix(
                overlap_dir
                / f"overlap_{previous_frame:03d}_{frame:03d}_tracked.csv",
                tracked_overlap,
            )

            diagonal = np.abs(np.diag(tracked_overlap))

            off_diagonal = np.abs(tracked_overlap.copy())
            np.fill_diagonal(off_diagonal, 0.0)

            orthogonality_error = np.linalg.norm(
                tracked_overlap.T @ tracked_overlap
                - np.eye(args.nactive)
            )

            pair_rows.append(
                {
                    "frame_i": previous_frame,
                    "frame_j": frame,
                    "time_i_fs": previous["time_fs"],
                    "time_j_fs": time_fs,
                    "minimum_diagonal_overlap": diagonal.min(),
                    "mean_diagonal_overlap": diagonal.mean(),
                    "maximum_offdiagonal_overlap": off_diagonal.max(),
                    "subspace_orthogonality_error": orthogonality_error,
                }
            )

            print(
                "Tracked overlap: "
                f"min diagonal = {diagonal.min():.6f}, "
                f"mean diagonal = {diagonal.mean():.6f}, "
                f"max off-diagonal = {off_diagonal.max():.6f}"
            )

        # Put energies and fragment weights into tracked-state order.
        tracked_indices = canonical_indices[permutation]
        tracked_energies = canonical_energy[permutation]
        tracked_far = far[permutation]
        tracked_near = near[permutation]
        tracked_c60 = c60[permutation]

        for state in range(args.nactive):
            canonical_offset = int(permutation[state])

            state_rows.append(
                {
                    "frame": frame,
                    "time_fs": time_fs,
                    "tracked_state": state,
                    "adiabatic_label": f"LUMO+{canonical_offset}",
                    "mo_index_0based": int(tracked_indices[state]),
                    "energy_hartree": tracked_energies[state],
                    "energy_eV": tracked_energies[state] * HARTREE_TO_EV,
                    "far_H2Pc_weight": tracked_far[state],
                    "near_H2Pc_weight": tracked_near[state],
                    "donor_total_weight":
                        tracked_far[state] + tracked_near[state],
                    "C60_weight": tracked_c60[state],
                    "phase": phases[state],
                }
            )

        np.savez_compressed(
            npzdir / f"frame_{frame:03d}.npz",
            frame=frame,
            time_fs=time_fs,
            total_energy=total_energy,
            homo=homo,
            lumo=lumo,
            tracked_mo_indices=tracked_indices,
            tracked_energies_hartree=tracked_energies,
            tracked_coefficients=tracked_coeff,
            far_H2Pc_weight=tracked_far,
            near_H2Pc_weight=tracked_near,
            C60_weight=tracked_c60,
            canonical_permutation=permutation,
            phases=phases,
        )

        donor_total = tracked_far + tracked_near
        strongest = np.argsort(donor_total)[::-1][:2]

        print(f"SCF energy: {total_energy:.12f} Eh")
        for state in strongest:
            print(
                f"  tracked state {state}: "
                f"{tracked_indices[state] - lumo:+d} relative to LUMO, "
                f"E = {tracked_energies[state] * HARTREE_TO_EV:.6f} eV, "
                f"donor = {donor_total[state]:.4f}, "
                f"C60 = {tracked_c60[state]:.4f}"
            )

        previous = {
            "frame": frame,
            "time_fs": time_fs,
            "mol": mol,
            "coeff": tracked_coeff,
        }

    state_csv = outdir / "tracked_states.csv"
    with state_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=state_rows[0].keys())
        writer.writeheader()
        writer.writerows(state_rows)

    pair_csv = outdir / "tracking_summary.csv"
    with pair_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)

    print("\nFinished.")
    print(f"Tracked states:   {state_csv}")
    print(f"Pair diagnostics: {pair_csv}")
    print(f"Overlap matrices: {overlap_dir}")


if __name__ == "__main__":
    main()
