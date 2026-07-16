#!/usr/bin/env python3
"""
Extract a smooth two-state fragment-diabatic donor/acceptor model from the
tracked ten-state PySCF trajectories.

For each frame:
  1. Diagonalize the C60 population operator.
  2. Define the four lowest-eigenvalue vectors as the donor subspace and the
     six highest-eigenvalue vectors as the acceptor subspace.
  3. Parallel-transport the initially prepared donor state inside the donor
     subspace.
  4. Define the "bright" acceptor as the normalized acceptor component of
     H_el |D>.
  5. Project the electronic Hamiltonian into {|D>, |A>}.

The script also constructs reduced consecutive-frame overlaps, matrix-log
time-derivative couplings, and a reduced vibronic Hamiltonian. Residual
couplings to discarded donor/acceptor directions quantify whether a two-state
model is adequate or whether a larger reduced model is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import logm

HARTREE_TO_EV = 27.211386245988
FS_TO_AU = 41.3413745758


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output_tracked"),
        help="Directory containing traj_XX/tracked_data.npz",
    )
    parser.add_argument(
        "--outroot",
        type=Path,
        default=Path("output_reduced_da"),
        help="Output root directory",
    )
    parser.add_argument("--ntraj", type=int, default=10)
    parser.add_argument(
        "--projector-cutoff",
        type=float,
        default=0.5,
        help="Eigenvalue cutoff separating donor and C60 subspaces",
    )
    parser.add_argument(
        "--coupling-floor",
        type=float,
        default=1.0e-12,
        help="Minimum bright-acceptor coupling norm in Hartree",
    )
    return parser.parse_args()


def hermitize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def normalize(v: np.ndarray, floor: float = 1.0e-14) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(v))
    if norm < floor:
        raise RuntimeError(f"Cannot normalize vector with norm {norm:.3e}")
    return v / norm, norm


def phase_align(
    previous: np.ndarray,
    current: np.ndarray,
    overlap_prev_current: np.ndarray,
) -> np.ndarray:
    """
    Choose the phase of `current` so that
        <previous | current> = previous^dagger O current
    is positive real whenever possible.
    """
    overlap = np.vdot(previous, overlap_prev_current @ current)
    if abs(overlap) > 1.0e-14:
        current = current * np.exp(-1j * np.angle(overlap))
    return current


def polar_factor(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, singular_values, right_h = np.linalg.svd(a)
    return left @ right_h, singular_values


def build_fragment_subspaces(
    projector_c60: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = hermitize(projector_c60)
    eigenvalues, eigenvectors = np.linalg.eigh(p)

    donor_mask = eigenvalues < cutoff
    acceptor_mask = ~donor_mask

    if donor_mask.sum() != 4 or acceptor_mask.sum() != 6:
        raise RuntimeError(
            "Expected a 4D donor and 6D acceptor partition, got "
            f"{donor_mask.sum()} and {acceptor_mask.sum()} with eigenvalues "
            f"{eigenvalues}"
        )

    u_d = eigenvectors[:, donor_mask]
    u_a = eigenvectors[:, acceptor_mask]
    q_d = u_d @ u_d.conj().T
    q_a = u_a @ u_a.conj().T
    return eigenvalues, q_d, q_a


def process_trajectory(
    input_path: Path,
    output_dir: Path,
    cutoff: float,
    coupling_floor: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=False) as data:
        time_fs = np.asarray(data["time_fs"], dtype=float)
        energies = np.asarray(data["active_energy_hartree"], dtype=float)
        projectors = np.asarray(data["projector_c60"], dtype=complex)
        overlaps = np.asarray(data["polar_overlap"], dtype=complex)
        c0 = np.asarray(data["initial_c0"], dtype=complex)
        initial_capture = float(data["initial_capture"])

    nframes, nstates = energies.shape
    if nstates != 10:
        raise RuntimeError(f"Expected 10 active states, found {nstates}")
    if projectors.shape != (nframes, nstates, nstates):
        raise RuntimeError(f"Unexpected projector shape {projectors.shape}")
    if overlaps.shape != (nframes - 1, nstates, nstates):
        raise RuntimeError(f"Unexpected overlap shape {overlaps.shape}")

    basis = np.zeros((nframes, nstates, 2), dtype=complex)
    hel = np.zeros((nframes, 2, 2), dtype=complex)

    projector_eigenvalues = np.zeros((nframes, nstates), dtype=float)
    donor_c60_population = np.zeros(nframes, dtype=float)
    acceptor_c60_population = np.zeros(nframes, dtype=float)
    donor_transport_overlap = np.ones(nframes, dtype=float)
    acceptor_transport_overlap = np.ones(nframes, dtype=float)
    bright_coupling_norm = np.zeros(nframes, dtype=float)

    donor_residual = np.zeros(nframes, dtype=float)
    acceptor_residual = np.zeros(nframes, dtype=float)
    donor_back_residual = np.zeros(nframes, dtype=float)

    donor_state = None
    acceptor_state = None
    initial_donor_projection_norm = None

    for frame in range(nframes):
        eigenvalues, q_d, q_a = build_fragment_subspaces(
            projectors[frame], cutoff
        )
        projector_eigenvalues[frame] = eigenvalues
        h_el = np.diag(energies[frame].astype(complex))

        if frame == 0:
            donor_state, donor_norm = normalize(q_d @ c0)
            initial_donor_projection_norm = donor_norm
        else:
            mapped_previous_donor = overlaps[frame - 1].conj().T @ donor_state
            donor_state, _ = normalize(q_d @ mapped_previous_donor)
            donor_state = phase_align(
                basis[frame - 1, :, 0],
                donor_state,
                overlaps[frame - 1],
            )
            donor_transport_overlap[frame] = abs(
                np.vdot(
                    basis[frame - 1, :, 0],
                    overlaps[frame - 1] @ donor_state,
                )
            )

        # The acceptor combination directly coupled to the transported donor.
        bright_vector = q_a @ (h_el @ donor_state)
        bright_norm = float(np.linalg.norm(bright_vector))
        bright_coupling_norm[frame] = bright_norm

        if bright_norm > coupling_floor:
            acceptor_state = bright_vector / bright_norm
        elif frame > 0:
            mapped_previous_acceptor = (
                overlaps[frame - 1].conj().T @ acceptor_state
            )
            acceptor_state, _ = normalize(q_a @ mapped_previous_acceptor)
        else:
            raise RuntimeError(
                f"Bright acceptor coupling is too small at frame 0: "
                f"{bright_norm:.3e} Hartree"
            )

        if frame > 0:
            acceptor_state = phase_align(
                basis[frame - 1, :, 1],
                acceptor_state,
                overlaps[frame - 1],
            )
            acceptor_transport_overlap[frame] = abs(
                np.vdot(
                    basis[frame - 1, :, 1],
                    overlaps[frame - 1] @ acceptor_state,
                )
            )

        # Numerical orthogonalization is harmless because the fragment
        # eigenspaces are orthogonal already.
        acceptor_state -= donor_state * np.vdot(donor_state, acceptor_state)
        acceptor_state, _ = normalize(acceptor_state)

        b = np.column_stack((donor_state, acceptor_state))
        h_red = hermitize(b.conj().T @ h_el @ b)

        basis[frame] = b
        hel[frame] = h_red

        donor_c60_population[frame] = float(
            np.vdot(donor_state, projectors[frame] @ donor_state).real
        )
        acceptor_c60_population[frame] = float(
            np.vdot(acceptor_state, projectors[frame] @ acceptor_state).real
        )

        # Residual coupling norms to discarded states.
        h_d = h_el @ donor_state
        h_a = h_el @ acceptor_state
        eps_d = h_red[0, 0]
        eps_a = h_red[1, 1]
        v_da = h_red[0, 1]

        donor_residual[frame] = np.linalg.norm(q_d @ h_d - eps_d * donor_state)
        acceptor_residual[frame] = np.linalg.norm(
            q_a @ h_a - eps_a * acceptor_state
        )
        donor_back_residual[frame] = np.linalg.norm(
            q_d @ h_a - np.conj(v_da) * donor_state
        )

    reduced_overlap = np.zeros((nframes - 1, 2, 2), dtype=complex)
    reduced_polar = np.zeros_like(reduced_overlap)
    reduced_singular_values = np.zeros((nframes - 1, 2), dtype=float)
    reduced_derivative_coupling = np.zeros_like(reduced_overlap)
    reduced_hvib_mid = np.zeros_like(reduced_overlap)
    reduced_overlap_orthogonality_error = np.zeros(nframes - 1, dtype=float)
    reduced_hvib_hermiticity_error = np.zeros(nframes - 1, dtype=float)

    for interval in range(nframes - 1):
        s_red = (
            basis[interval].conj().T
            @ overlaps[interval]
            @ basis[interval + 1]
        )
        u_red, singular_values = polar_factor(s_red)

        dt_au = (time_fs[interval + 1] - time_fs[interval]) * FS_TO_AU
        d_red = logm(u_red) / dt_au
        # Enforce the anti-Hermitian part at roundoff level.
        d_red = 0.5 * (d_red - d_red.conj().T)

        h_mid = 0.5 * (hel[interval] + hel[interval + 1]) - 1j * d_red
        h_mid = hermitize(h_mid)

        reduced_overlap[interval] = s_red
        reduced_polar[interval] = u_red
        reduced_singular_values[interval] = singular_values
        reduced_derivative_coupling[interval] = d_red
        reduced_hvib_mid[interval] = h_mid
        reduced_overlap_orthogonality_error[interval] = np.linalg.norm(
            s_red.conj().T @ s_red - np.eye(2)
        )
        reduced_hvib_hermiticity_error[interval] = np.linalg.norm(
            h_mid - h_mid.conj().T
        )

    epsilon_d = hel[:, 0, 0].real
    epsilon_a = hel[:, 1, 1].real
    coupling = hel[:, 0, 1]
    gap = epsilon_a - epsilon_d

    np.savez_compressed(
        output_dir / "reduced_da.npz",
        time_fs=time_fs,
        basis_vectors=basis,
        electronic_hamiltonian_hartree=hel,
        epsilon_d_hartree=epsilon_d,
        epsilon_a_hartree=epsilon_a,
        gap_hartree=gap,
        coupling_hartree=coupling,
        projector_eigenvalues=projector_eigenvalues,
        donor_c60_population=donor_c60_population,
        acceptor_c60_population=acceptor_c60_population,
        donor_transport_overlap=donor_transport_overlap,
        acceptor_transport_overlap=acceptor_transport_overlap,
        bright_coupling_norm_hartree=bright_coupling_norm,
        donor_residual_hartree=donor_residual,
        acceptor_residual_hartree=acceptor_residual,
        donor_back_residual_hartree=donor_back_residual,
        reduced_overlap=reduced_overlap,
        reduced_polar_overlap=reduced_polar,
        reduced_singular_values=reduced_singular_values,
        reduced_derivative_coupling_au=reduced_derivative_coupling,
        reduced_hvib_mid_hartree=reduced_hvib_mid,
        reduced_overlap_orthogonality_error=(
            reduced_overlap_orthogonality_error
        ),
        reduced_hvib_hermiticity_error=reduced_hvib_hermiticity_error,
        initial_c0_reduced=np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        initial_full_active_capture=initial_capture,
        initial_donor_projection_norm=initial_donor_projection_norm,
    )

    csv_path = output_dir / "reduced_da.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "frame",
            "time_fs",
            "epsilon_d_eV",
            "epsilon_a_eV",
            "gap_eV",
            "coupling_real_eV",
            "coupling_imag_eV",
            "coupling_abs_eV",
            "donor_c60_population",
            "acceptor_c60_population",
            "donor_transport_overlap",
            "acceptor_transport_overlap",
            "donor_residual_eV",
            "acceptor_residual_eV",
            "donor_back_residual_eV",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for frame in range(nframes):
            writer.writerow(
                {
                    "frame": frame,
                    "time_fs": time_fs[frame],
                    "epsilon_d_eV": epsilon_d[frame] * HARTREE_TO_EV,
                    "epsilon_a_eV": epsilon_a[frame] * HARTREE_TO_EV,
                    "gap_eV": gap[frame] * HARTREE_TO_EV,
                    "coupling_real_eV": coupling[frame].real * HARTREE_TO_EV,
                    "coupling_imag_eV": coupling[frame].imag * HARTREE_TO_EV,
                    "coupling_abs_eV": abs(coupling[frame]) * HARTREE_TO_EV,
                    "donor_c60_population": donor_c60_population[frame],
                    "acceptor_c60_population": acceptor_c60_population[frame],
                    "donor_transport_overlap": donor_transport_overlap[frame],
                    "acceptor_transport_overlap": acceptor_transport_overlap[frame],
                    "donor_residual_eV": donor_residual[frame] * HARTREE_TO_EV,
                    "acceptor_residual_eV": (
                        acceptor_residual[frame] * HARTREE_TO_EV
                    ),
                    "donor_back_residual_eV": (
                        donor_back_residual[frame] * HARTREE_TO_EV
                    ),
                }
            )

    summary = {
        "input": str(input_path),
        "nframes": nframes,
        "projector_cutoff": cutoff,
        "initial_full_active_capture": initial_capture,
        "initial_donor_projection_norm": initial_donor_projection_norm,
        "projector_eigenvalue_max_donor": float(
            np.max(projector_eigenvalues[:, :4])
        ),
        "projector_eigenvalue_min_acceptor": float(
            np.min(projector_eigenvalues[:, 4:])
        ),
        "minimum_donor_transport_overlap": float(
            np.min(donor_transport_overlap[1:])
        ),
        "minimum_acceptor_transport_overlap": float(
            np.min(acceptor_transport_overlap[1:])
        ),
        "minimum_reduced_singular_value": float(
            np.min(reduced_singular_values)
        ),
        "maximum_reduced_overlap_orthogonality_error": float(
            np.max(reduced_overlap_orthogonality_error)
        ),
        "mean_gap_eV": float(np.mean(gap) * HARTREE_TO_EV),
        "std_gap_eV": float(np.std(gap, ddof=1) * HARTREE_TO_EV),
        "mean_abs_coupling_eV": float(
            np.mean(np.abs(coupling)) * HARTREE_TO_EV
        ),
        "max_abs_coupling_eV": float(
            np.max(np.abs(coupling)) * HARTREE_TO_EV
        ),
        "mean_donor_residual_eV": float(
            np.mean(donor_residual) * HARTREE_TO_EV
        ),
        "mean_acceptor_residual_eV": float(
            np.mean(acceptor_residual) * HARTREE_TO_EV
        ),
        "mean_donor_back_residual_eV": float(
            np.mean(donor_back_residual) * HARTREE_TO_EV
        ),
        "maximum_reduced_hvib_hermiticity_error": float(
            np.max(reduced_hvib_hermiticity_error)
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    args = parse_args()
    args.outroot.mkdir(parents=True, exist_ok=True)

    summaries = []
    for trajectory in range(args.ntraj):
        input_path = args.root / f"traj_{trajectory:02d}" / "tracked_data.npz"
        output_dir = args.outroot / f"traj_{trajectory:02d}"
        summary = process_trajectory(
            input_path,
            output_dir,
            args.projector_cutoff,
            args.coupling_floor,
        )
        summary["trajectory"] = trajectory
        summaries.append(summary)

    aggregate_path = args.outroot / "aggregate_summary.csv"
    fields = [
        "trajectory",
        "projector_eigenvalue_max_donor",
        "projector_eigenvalue_min_acceptor",
        "minimum_donor_transport_overlap",
        "minimum_acceptor_transport_overlap",
        "minimum_reduced_singular_value",
        "mean_gap_eV",
        "std_gap_eV",
        "mean_abs_coupling_eV",
        "max_abs_coupling_eV",
        "mean_donor_residual_eV",
        "mean_acceptor_residual_eV",
        "mean_donor_back_residual_eV",
    ]
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})

    print(
        f"{'traj':>5} {'D max eig':>10} {'A min eig':>10} "
        f"{'min <D|D+>':>11} {'min <A|A+>':>11} {'min sv':>9} "
        f"{'<gap>/eV':>10} {'sd gap':>9} {'<|V|>/eV':>10} "
        f"{'max |V|':>9} {'D resid':>9} {'A resid':>9}"
    )
    print("-" * 142)

    for summary in summaries:
        print(
            f"{summary['trajectory']:5d} "
            f"{summary['projector_eigenvalue_max_donor']:10.5f} "
            f"{summary['projector_eigenvalue_min_acceptor']:10.5f} "
            f"{summary['minimum_donor_transport_overlap']:11.6f} "
            f"{summary['minimum_acceptor_transport_overlap']:11.6f} "
            f"{summary['minimum_reduced_singular_value']:9.6f} "
            f"{summary['mean_gap_eV']:10.4f} "
            f"{summary['std_gap_eV']:9.4f} "
            f"{summary['mean_abs_coupling_eV']:10.4f} "
            f"{summary['max_abs_coupling_eV']:9.4f} "
            f"{summary['mean_donor_residual_eV']:9.4f} "
            f"{summary['mean_acceptor_residual_eV']:9.4f}"
        )

    print(f"\nWrote reduced models to {args.outroot}")


if __name__ == "__main__":
    main()
