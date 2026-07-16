#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import logm
from scipy.optimize import linear_sum_assignment
from pyscf import gto, lib

FS_TO_AU = 41.3413745758


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track a ten-state active space and pack Libra-ready electronic data."
    )
    p.add_argument("--trajectory", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--initial-state", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--nframes", type=int, default=200)
    p.add_argument("--natoms", type=int, default=176)
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument(
        "--det-tol",
        type=float,
        default=1.0e-8,
        help="Abort if the polar-overlap determinant falls below this value.",
    )
    return p.parse_args()


def read_xyz_frames(path: Path, natoms: int, nframes: int):
    lines = path.read_text().splitlines()
    block = natoms + 2
    complete = len(lines) // block
    if complete < nframes:
        raise RuntimeError(
            f"Requested {nframes} frames but only {complete} complete XYZ frames exist."
        )

    symbols = None
    frames = []
    comments = []

    for iframe in range(nframes):
        chunk = lines[iframe * block : (iframe + 1) * block]
        if len(chunk) != block or int(chunk[0]) != natoms:
            raise RuntimeError(f"Malformed XYZ frame {iframe} in {path}")

        frame_symbols = []
        xyz = np.empty((natoms, 3), dtype=float)
        for iatom, line in enumerate(chunk[2:]):
            fields = line.split()
            frame_symbols.append(fields[0])
            xyz[iatom] = [float(fields[1]), float(fields[2]), float(fields[3])]

        if symbols is None:
            symbols = frame_symbols
        elif frame_symbols != symbols:
            raise RuntimeError(f"Atom ordering changed at frame {iframe}")

        comments.append(chunk[1])
        frames.append(xyz)

    return symbols, frames, comments


def build_mol(symbols, xyz, basis: str, charge: int, spin: int):
    atom = [(symbol, tuple(coord)) for symbol, coord in zip(symbols, xyz)]
    return gto.M(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def load_frame(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def transform_projector(
    projector: np.ndarray, perm: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    transformed = projector[np.ix_(perm, perm)]
    return phase[:, None] * transformed * phase[None, :]


def main() -> None:
    args = parse_args()
    lib.num_threads(args.threads)
    args.outdir.mkdir(parents=True, exist_ok=True)

    symbols, xyz_frames, comments = read_xyz_frames(
        args.trajectory, args.natoms, args.nframes
    )

    first = load_frame(args.raw_dir / "frame_0000.npz")
    nstates = int(first["active_energy_hartree"].shape[0])
    nao = int(first["active_coeff"].shape[0])

    times = np.empty(args.nframes)
    total_energies = np.empty(args.nframes)
    active_energies = np.empty((args.nframes, nstates))
    projectors = np.empty((args.nframes, nstates, nstates), dtype=complex)
    permutations = np.empty((args.nframes, nstates), dtype=int)
    phases = np.empty((args.nframes, nstates), dtype=float)

    overlaps = np.empty((args.nframes - 1, nstates, nstates), dtype=complex)
    polar_overlaps = np.empty_like(overlaps)
    derivative_couplings = np.empty_like(overlaps)
    hvib_mid = np.empty_like(overlaps)

    min_diag = np.empty(args.nframes - 1)
    mean_diag = np.empty(args.nframes - 1)
    max_offdiag = np.empty(args.nframes - 1)
    min_singular = np.empty(args.nframes - 1)
    max_singular = np.empty(args.nframes - 1)
    overlap_orth_error = np.empty(args.nframes - 1)
    polar_det = np.empty(args.nframes - 1)
    d_antiherm_error = np.empty(args.nframes - 1)
    hvib_herm_error = np.empty(args.nframes - 1)
    log_imag_residual = np.empty(args.nframes - 1)

    coeff_prev = np.asarray(first["active_coeff"], dtype=float)
    if coeff_prev.shape != (nao, nstates):
        raise RuntimeError("Unexpected active coefficient shape in frame 0")

    times[0] = float(first["time_fs"])
    total_energies[0] = float(first["total_energy_hartree"])
    active_energies[0] = np.asarray(first["active_energy_hartree"], dtype=float)
    projectors[0] = np.asarray(first["projector_c60"], dtype=complex)
    permutations[0] = np.arange(nstates)
    phases[0] = 1.0

    mol_prev = build_mol(
        symbols, xyz_frames[0], args.basis, args.charge, args.spin
    )
    if mol_prev.nao_nr() != nao:
        raise RuntimeError(
            f"AO mismatch: rebuilt molecule has {mol_prev.nao_nr()} AOs, "
            f"raw file has {nao}."
        )

    print(
        f"Tracking {args.nframes} frames, {nstates} states, {nao} AOs, "
        f"basis={args.basis}, threads={args.threads}",
        flush=True,
    )

    for iframe in range(1, args.nframes):
        raw = load_frame(args.raw_dir / f"frame_{iframe:04d}.npz")
        coeff_raw = np.asarray(raw["active_coeff"], dtype=float)
        energies_raw = np.asarray(raw["active_energy_hartree"], dtype=float)
        projector_raw = np.asarray(raw["projector_c60"], dtype=complex)

        mol_next = build_mol(
            symbols, xyz_frames[iframe], args.basis, args.charge, args.spin
        )
        if mol_next.nao_nr() != nao:
            raise RuntimeError(
                f"AO mismatch at frame {iframe}: {mol_next.nao_nr()} != {nao}"
            )

        # Cross-geometry AO overlap:
        # S^(n,n+1)_mu,nu = <chi_mu(R_n) | chi_nu(R_{n+1})>
        s_cross = gto.intor_cross("int1e_ovlp", mol_prev, mol_next)

        # Active-space overlap before matching:
        raw_overlap = coeff_prev.conj().T @ s_cross @ coeff_raw

        # Match states by maximizing the total absolute overlap.
        rows, cols = linear_sum_assignment(-np.abs(raw_overlap))
        perm = np.empty(nstates, dtype=int)
        perm[rows] = cols

        # Fix the sign of each matched real orbital.
        permuted_overlap = raw_overlap[:, perm]
        phase = np.sign(np.real(np.diag(permuted_overlap)))
        phase[phase == 0.0] = 1.0

        coeff_next = coeff_raw[:, perm] * phase[None, :]
        tracked_overlap = coeff_prev.conj().T @ s_cross @ coeff_next

        # Nearest unitary/orthogonal overlap from the polar factor.
        u, singular, vh = np.linalg.svd(tracked_overlap, full_matrices=False)
        q = u @ vh
        det_q = float(np.real_if_close(np.linalg.det(q)))
        if det_q < args.det_tol:
            raise RuntimeError(
                f"Improper or ill-conditioned polar overlap at pair "
                f"{iframe-1}->{iframe}: det={det_q:.6e}. "
                "Inspect the state assignment before making matrix-log couplings."
            )

        dt_fs = float(raw["time_fs"]) - times[iframe - 1]
        if dt_fs <= 0.0:
            raise RuntimeError(f"Non-positive time step at frame {iframe}: {dt_fs}")
        dt_au = dt_fs * FS_TO_AU

        # Q = exp(D Delta t), so D = log(Q)/Delta t.
        log_q = logm(q)
        log_imag_residual[iframe - 1] = np.max(np.abs(np.imag(log_q)))
        if log_imag_residual[iframe - 1] < 1.0e-9:
            log_q = np.real(log_q)

        dmat = log_q / dt_au
        dmat = 0.5 * (dmat - dmat.conj().T)

        energies_next = energies_raw[perm]
        projector_next = transform_projector(projector_raw, perm, phase)

        # In atomic units, H_vib = E_ad - i D.
        e_mid = 0.5 * (active_energies[iframe - 1] + energies_next)
        h_mid = np.diag(e_mid) - 1j * dmat

        times[iframe] = float(raw["time_fs"])
        total_energies[iframe] = float(raw["total_energy_hartree"])
        active_energies[iframe] = energies_next
        projectors[iframe] = projector_next
        permutations[iframe] = perm
        phases[iframe] = phase

        pair = iframe - 1
        overlaps[pair] = tracked_overlap
        polar_overlaps[pair] = q
        derivative_couplings[pair] = dmat
        hvib_mid[pair] = h_mid

        diagonal = np.abs(np.diag(tracked_overlap))
        offdiag = np.abs(tracked_overlap.copy())
        np.fill_diagonal(offdiag, 0.0)

        min_diag[pair] = diagonal.min()
        mean_diag[pair] = diagonal.mean()
        max_offdiag[pair] = offdiag.max()
        min_singular[pair] = singular.min()
        max_singular[pair] = singular.max()
        overlap_orth_error[pair] = np.linalg.norm(
            tracked_overlap.conj().T @ tracked_overlap - np.eye(nstates)
        )
        polar_det[pair] = det_q
        d_antiherm_error[pair] = np.linalg.norm(dmat + dmat.conj().T)
        hvib_herm_error[pair] = np.linalg.norm(h_mid - h_mid.conj().T)

        print(
            f"{iframe-1:04d}->{iframe:04d}  "
            f"min|diag O|={min_diag[pair]:.6f}  "
            f"max|offdiag O|={max_offdiag[pair]:.6f}  "
            f"sigma=[{min_singular[pair]:.6f},{max_singular[pair]:.6f}]  "
            f"det(Q)={det_q:.6f}",
            flush=True,
        )

        coeff_prev = coeff_next
        mol_prev = mol_next

    with np.load(args.initial_state, allow_pickle=False) as initial:
        c0 = np.asarray(initial["c0_normalized"], dtype=complex)
        capture = float(initial["active_capture"])
        donor_energy = float(initial["donor_energy_hartree"])
        donor_lumo = int(initial["donor_lumo_0based"])

    output = args.outdir / "tracked_data.npz"
    np.savez_compressed(
        output,
        time_fs=times,
        total_energy_hartree=total_energies,
        active_energy_hartree=active_energies,
        projector_c60=projectors,
        overlap=overlaps,
        polar_overlap=polar_overlaps,
        derivative_coupling_au=derivative_couplings,
        hvib_mid_hartree=hvib_mid,
        permutation=permutations,
        phase=phases,
        initial_c0=c0,
        initial_capture=capture,
        donor_energy_hartree=donor_energy,
        donor_lumo_0based=donor_lumo,
        min_matched_diagonal=min_diag,
        mean_matched_diagonal=mean_diag,
        max_offdiagonal=max_offdiag,
        min_singular_value=min_singular,
        max_singular_value=max_singular,
        overlap_orthogonality_error=overlap_orth_error,
        polar_determinant=polar_det,
        matrix_log_imaginary_residual=log_imag_residual,
        derivative_antihermiticity_error=d_antiherm_error,
        hvib_hermiticity_error=hvib_herm_error,
    )

    summary = {
        "trajectory": str(args.trajectory),
        "raw_directory": str(args.raw_dir),
        "nframes": args.nframes,
        "nstates": nstates,
        "nao": nao,
        "basis": args.basis,
        "initial_capture": capture,
        "minimum_matched_diagonal": float(min_diag.min()),
        "maximum_offdiagonal": float(max_offdiag.max()),
        "minimum_singular_value": float(min_singular.min()),
        "maximum_overlap_orthogonality_error": float(overlap_orth_error.max()),
        "maximum_matrix_log_imaginary_residual": float(log_imag_residual.max()),
        "maximum_derivative_antihermiticity_error": float(
            d_antiherm_error.max()
        ),
        "maximum_hvib_hermiticity_error": float(hvib_herm_error.max()),
        "comments": comments,
    }
    (args.outdir / "tracking_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"\nWrote {output}", flush=True)
    print(
        f"Global diagnostics: min matched diagonal={min_diag.min():.6f}, "
        f"max offdiagonal={max_offdiag.max():.6f}, "
        f"min singular={min_singular.min():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
