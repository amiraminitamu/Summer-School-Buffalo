#!/usr/bin/env python3
"""
Build and validate a 1-donor + 3-acceptor fragment-diabatic model.

The model is derived from each tracked ten-state PySCF trajectory:

  * donor state:
      one fragment-localized donor eigenstate, selected at frame 0 by maximum
      overlap with the prepared donor wavefunction and tracked thereafter by
      maximum cross-frame overlap;

  * acceptor states:
      the isolated lower C60 triplet.  The three-dimensional subspace is
      parallel-transported between frames to avoid artificial state swapping
      inside the nearly degenerate triplet.

The script constructs the reduced electronic and vibronic Hamiltonians,
propagates both the full ten-state and reduced four-state models, evaluates
the full projected C60 population, and reports the reduction error.

No fitting is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, logm

FS_TO_AU = 41.3413745758
HARTREE_TO_EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("output_tracked"))
    p.add_argument(
        "--outroot",
        type=Path,
        default=Path("output_reduced_1d3a"),
    )
    p.add_argument("--ntraj", type=int, default=10)
    p.add_argument("--projector-cutoff", type=float, default=0.5)
    return p.parse_args()


def hermitize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def normalize(v: np.ndarray, floor: float = 1.0e-14) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < floor:
        raise RuntimeError(f"Cannot normalize vector with norm {norm:.3e}")
    return v / norm


def polar_factor(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, singular_values, right_h = np.linalg.svd(a)
    return left @ right_h, singular_values


def phase_align(
    previous: np.ndarray,
    current: np.ndarray,
    overlap_prev_current: np.ndarray,
) -> np.ndarray:
    z = np.vdot(previous, overlap_prev_current @ current)
    if abs(z) > 1.0e-14:
        current = current * np.exp(-1j * np.angle(z))
    return current


def fragment_eigenstates(
    energies_h: np.ndarray,
    projector_c60: np.ndarray,
    cutoff: float,
):
    p = hermitize(projector_c60)
    pvals, pvecs = np.linalg.eigh(p)

    donor_mask = pvals < cutoff
    acceptor_mask = ~donor_mask

    if donor_mask.sum() != 4 or acceptor_mask.sum() != 6:
        raise RuntimeError(
            f"Expected 4 donor and 6 acceptor directions; got "
            f"{donor_mask.sum()} and {acceptor_mask.sum()}. "
            f"Projector eigenvalues: {pvals}"
        )

    u_d = pvecs[:, donor_mask]
    u_a = pvecs[:, acceptor_mask]
    h = np.diag(np.asarray(energies_h, dtype=complex))

    h_dd = hermitize(u_d.conj().T @ h @ u_d)
    h_aa = hermitize(u_a.conj().T @ h @ u_a)

    donor_energies, x_d = np.linalg.eigh(h_dd)
    acceptor_energies, x_a = np.linalg.eigh(h_aa)

    donor_states = u_d @ x_d
    acceptor_states = u_a @ x_a

    return {
        "projector_eigenvalues": pvals,
        "donor_energies": donor_energies,
        "acceptor_energies": acceptor_energies,
        "donor_states": donor_states,
        "acceptor_states": acceptor_states,
    }


def parallel_transport_subspace(
    previous_basis: np.ndarray,
    current_raw_basis: np.ndarray,
    overlap_prev_current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate the current subspace basis to maximize continuity with the
    previous basis.

    If X = B_prev^dagger O B_raw = L Sigma R^dagger, use
    B_new = B_raw R L^dagger, so B_prev^dagger O B_new is Hermitian positive.
    """
    x = previous_basis.conj().T @ overlap_prev_current @ current_raw_basis
    left, singular_values, right_h = np.linalg.svd(x)
    rotation = right_h.conj().T @ left.conj().T
    new_basis = current_raw_basis @ rotation
    return new_basis, singular_values


def propagate_piecewise(
    c0: np.ndarray,
    hvib_mid: np.ndarray,
    time_fs: np.ndarray,
) -> np.ndarray:
    nframes = len(time_fs)
    coeff = np.zeros((nframes, len(c0)), dtype=complex)
    coeff[0] = normalize(c0)

    for interval in range(nframes - 1):
        dt_au = (time_fs[interval + 1] - time_fs[interval]) * FS_TO_AU
        h = hermitize(hvib_mid[interval])
        coeff[interval + 1] = expm(-1j * h * dt_au) @ coeff[interval]

    return coeff


def fragment_population(
    coefficients: np.ndarray,
    projectors: np.ndarray,
) -> np.ndarray:
    result = np.empty(coefficients.shape[0], dtype=float)
    for frame, c in enumerate(coefficients):
        value = np.vdot(c, projectors[frame] @ c)
        if abs(value.imag) > 1.0e-8:
            raise RuntimeError(
                f"Large imaginary fragment population at frame {frame}: {value}"
            )
        result[frame] = value.real
    return result


def process_trajectory(
    input_path: Path,
    output_dir: Path,
    cutoff: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=False) as d:
        time_fs = np.asarray(d["time_fs"], dtype=float)
        energies = np.asarray(d["active_energy_hartree"], dtype=float)
        projectors = np.asarray(d["projector_c60"], dtype=complex)
        polar_overlaps = np.asarray(d["polar_overlap"], dtype=complex)
        full_hvib_mid = np.asarray(d["hvib_mid_hartree"], dtype=complex)
        c0_full = np.asarray(d["initial_c0"], dtype=complex)

    nframes, nstates = energies.shape
    if nstates != 10:
        raise RuntimeError(f"Expected 10 states, found {nstates}")

    basis = np.zeros((nframes, nstates, 4), dtype=complex)
    hel_reduced = np.zeros((nframes, 4, 4), dtype=complex)
    projector_reduced = np.zeros((nframes, 4, 4), dtype=complex)

    donor_branch_index = np.zeros(nframes, dtype=int)
    donor_transport_overlap = np.ones(nframes, dtype=float)
    acceptor_subspace_min_sv = np.ones(nframes, dtype=float)
    reduced_basis_orthogonality_error = np.zeros(nframes, dtype=float)

    donor_energy = np.zeros(nframes, dtype=float)
    low_acceptor_energies = np.zeros((nframes, 3), dtype=float)
    upper_acceptor_gap = np.zeros(nframes, dtype=float)

    previous_donor = None
    previous_acceptors = None

    for frame in range(nframes):
        frag = fragment_eigenstates(
            energies[frame],
            projectors[frame],
            cutoff,
        )

        donor_candidates = frag["donor_states"]
        acceptor_candidates = frag["acceptor_states"]
        raw_low_acceptors = acceptor_candidates[:, :3]

        if frame == 0:
            donor_weights = np.abs(donor_candidates.conj().T @ c0_full) ** 2
            donor_index = int(np.argmax(donor_weights))
            donor = donor_candidates[:, donor_index]
            low_acceptors = raw_low_acceptors
        else:
            overlaps_to_candidates = (
                previous_donor.conj().T
                @ polar_overlaps[frame - 1]
                @ donor_candidates
            )
            donor_index = int(np.argmax(np.abs(overlaps_to_candidates)))
            donor = donor_candidates[:, donor_index]
            donor = phase_align(
                previous_donor,
                donor,
                polar_overlaps[frame - 1],
            )
            donor_transport_overlap[frame] = abs(
                np.vdot(
                    previous_donor,
                    polar_overlaps[frame - 1] @ donor,
                )
            )

            low_acceptors, singular_values = parallel_transport_subspace(
                previous_acceptors,
                raw_low_acceptors,
                polar_overlaps[frame - 1],
            )
            acceptor_subspace_min_sv[frame] = float(
                np.min(singular_values)
            )

        # Fragment donor and acceptor spaces are already orthogonal.
        b = np.column_stack((donor, low_acceptors))
        overlap_check = b.conj().T @ b
        reduced_basis_orthogonality_error[frame] = np.linalg.norm(
            overlap_check - np.eye(4)
        )

        # Clean only roundoff-level drift.
        q, _ = np.linalg.qr(b)
        donor = phase_align(donor, q[:, 0], np.eye(nstates))
        # Preserve the selected donor exactly and orthogonalize acceptors to it.
        acceptor_block = low_acceptors - donor[:, None] * (
            donor.conj().T @ low_acceptors
        )[None, :]
        acceptor_block, _ = np.linalg.qr(acceptor_block)
        b = np.column_stack((normalize(donor), acceptor_block[:, :3]))

        h_el = np.diag(energies[frame].astype(complex))
        basis[frame] = b
        hel_reduced[frame] = hermitize(b.conj().T @ h_el @ b)
        projector_reduced[frame] = hermitize(
            b.conj().T @ projectors[frame] @ b
        )

        donor_branch_index[frame] = donor_index
        donor_energy[frame] = frag["donor_energies"][donor_index]
        low_acceptor_energies[frame] = frag["acceptor_energies"][:3]
        upper_acceptor_gap[frame] = (
            frag["acceptor_energies"][3]
            - frag["acceptor_energies"][2]
        )

        previous_donor = b[:, 0]
        previous_acceptors = b[:, 1:4]

    reduced_overlap = np.zeros((nframes - 1, 4, 4), dtype=complex)
    reduced_polar = np.zeros_like(reduced_overlap)
    reduced_singular_values = np.zeros((nframes - 1, 4), dtype=float)
    reduced_derivative_coupling = np.zeros_like(reduced_overlap)
    reduced_hvib_mid = np.zeros_like(reduced_overlap)
    reduced_overlap_orthogonality_error = np.zeros(
        nframes - 1, dtype=float
    )

    for interval in range(nframes - 1):
        s_red = (
            basis[interval].conj().T
            @ polar_overlaps[interval]
            @ basis[interval + 1]
        )
        u_red, singular_values = polar_factor(s_red)

        dt_au = (
            time_fs[interval + 1] - time_fs[interval]
        ) * FS_TO_AU
        d_red = logm(u_red) / dt_au
        d_red = 0.5 * (d_red - d_red.conj().T)

        h_mid = (
            0.5 * (hel_reduced[interval] + hel_reduced[interval + 1])
            - 1j * d_red
        )
        h_mid = hermitize(h_mid)

        reduced_overlap[interval] = s_red
        reduced_polar[interval] = u_red
        reduced_singular_values[interval] = singular_values
        reduced_derivative_coupling[interval] = d_red
        reduced_hvib_mid[interval] = h_mid
        reduced_overlap_orthogonality_error[interval] = np.linalg.norm(
            s_red.conj().T @ s_red - np.eye(4)
        )

    c0_reduced_raw = basis[0].conj().T @ c0_full
    initial_reduced_capture = float(
        np.vdot(c0_reduced_raw, c0_reduced_raw).real
    )
    c0_reduced = normalize(c0_reduced_raw)

    full_coefficients = propagate_piecewise(
        c0_full,
        full_hvib_mid,
        time_fs,
    )
    reduced_coefficients = propagate_piecewise(
        c0_reduced,
        reduced_hvib_mid,
        time_fs,
    )

    full_c60 = fragment_population(full_coefficients, projectors)
    reduced_c60 = fragment_population(
        reduced_coefficients,
        projector_reduced,
    )

    error = reduced_c60 - full_c60
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    max_abs_error = float(np.max(np.abs(error)))
    final_error = float(error[-1])

    np.savez_compressed(
        output_dir / "model_1d3a.npz",
        time_fs=time_fs,
        basis_vectors=basis,
        electronic_hamiltonian_hartree=hel_reduced,
        projector_c60_reduced=projector_reduced,
        reduced_overlap=reduced_overlap,
        reduced_polar_overlap=reduced_polar,
        reduced_singular_values=reduced_singular_values,
        reduced_derivative_coupling_au=reduced_derivative_coupling,
        reduced_hvib_mid_hartree=reduced_hvib_mid,
        initial_c0_reduced=c0_reduced,
        initial_reduced_capture=initial_reduced_capture,
        donor_branch_index=donor_branch_index,
        donor_transport_overlap=donor_transport_overlap,
        acceptor_subspace_min_singular_value=acceptor_subspace_min_sv,
        donor_energy_hartree=donor_energy,
        low_acceptor_energies_hartree=low_acceptor_energies,
        upper_acceptor_gap_hartree=upper_acceptor_gap,
        full_coefficients=full_coefficients,
        reduced_coefficients=reduced_coefficients,
        full_c60_population=full_c60,
        reduced_c60_population=reduced_c60,
        population_error=error,
    )

    csv_path = output_dir / "validation_timeseries.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "frame",
            "time_fs",
            "full_c60_population",
            "reduced_c60_population",
            "error",
            "donor_branch_index",
            "donor_transport_overlap",
            "acceptor_subspace_min_singular_value",
            "donor_energy_eV",
            "acceptor_1_energy_eV",
            "acceptor_2_energy_eV",
            "acceptor_3_energy_eV",
            "upper_acceptor_gap_eV",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for frame in range(nframes):
            writer.writerow(
                {
                    "frame": frame,
                    "time_fs": time_fs[frame],
                    "full_c60_population": full_c60[frame],
                    "reduced_c60_population": reduced_c60[frame],
                    "error": error[frame],
                    "donor_branch_index": donor_branch_index[frame],
                    "donor_transport_overlap": donor_transport_overlap[frame],
                    "acceptor_subspace_min_singular_value": (
                        acceptor_subspace_min_sv[frame]
                    ),
                    "donor_energy_eV": donor_energy[frame] * HARTREE_TO_EV,
                    "acceptor_1_energy_eV": (
                        low_acceptor_energies[frame, 0] * HARTREE_TO_EV
                    ),
                    "acceptor_2_energy_eV": (
                        low_acceptor_energies[frame, 1] * HARTREE_TO_EV
                    ),
                    "acceptor_3_energy_eV": (
                        low_acceptor_energies[frame, 2] * HARTREE_TO_EV
                    ),
                    "upper_acceptor_gap_eV": (
                        upper_acceptor_gap[frame] * HARTREE_TO_EV
                    ),
                }
            )

    summary = {
        "input": str(input_path),
        "model": "1 donor state + lower 3-state C60 manifold",
        "nframes": nframes,
        "initial_reduced_capture": initial_reduced_capture,
        "minimum_donor_transport_overlap": float(
            np.min(donor_transport_overlap[1:])
        ),
        "minimum_acceptor_subspace_singular_value": float(
            np.min(acceptor_subspace_min_sv[1:])
        ),
        "minimum_reduced_overlap_singular_value": float(
            np.min(reduced_singular_values)
        ),
        "maximum_reduced_overlap_orthogonality_error": float(
            np.max(reduced_overlap_orthogonality_error)
        ),
        "mean_upper_acceptor_gap_eV": float(
            np.mean(upper_acceptor_gap) * HARTREE_TO_EV
        ),
        "full_final_c60_population": float(full_c60[-1]),
        "reduced_final_c60_population": float(reduced_c60[-1]),
        "final_population_error": final_error,
        "population_rmse": rmse,
        "population_mae": mae,
        "population_max_abs_error": max_abs_error,
        "maximum_full_norm_error": float(
            np.max(
                np.abs(
                    np.sum(np.abs(full_coefficients) ** 2, axis=1) - 1.0
                )
            )
        ),
        "maximum_reduced_norm_error": float(
            np.max(
                np.abs(
                    np.sum(np.abs(reduced_coefficients) ** 2, axis=1) - 1.0
                )
            )
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    return summary, time_fs, full_c60, reduced_c60


def main() -> None:
    args = parse_args()
    args.outroot.mkdir(parents=True, exist_ok=True)

    summaries = []
    full_all = []
    reduced_all = []
    reference_time = None

    for traj in range(args.ntraj):
        input_path = args.root / f"traj_{traj:02d}" / "tracked_data.npz"
        output_dir = args.outroot / f"traj_{traj:02d}"

        summary, time_fs, full_c60, reduced_c60 = process_trajectory(
            input_path,
            output_dir,
            args.projector_cutoff,
        )
        summary["trajectory"] = traj
        summaries.append(summary)
        full_all.append(full_c60)
        reduced_all.append(reduced_c60)

        if reference_time is None:
            reference_time = time_fs
        elif not np.array_equal(reference_time, time_fs):
            raise RuntimeError("Trajectory time grids do not match")

    full_all = np.asarray(full_all)
    reduced_all = np.asarray(reduced_all)
    full_mean = np.mean(full_all, axis=0)
    reduced_mean = np.mean(reduced_all, axis=0)
    ensemble_error = reduced_mean - full_mean
    ensemble_rmse = float(np.sqrt(np.mean(ensemble_error**2)))

    aggregate_path = args.outroot / "aggregate_validation.csv"
    fields = [
        "trajectory",
        "initial_reduced_capture",
        "minimum_donor_transport_overlap",
        "minimum_acceptor_subspace_singular_value",
        "minimum_reduced_overlap_singular_value",
        "mean_upper_acceptor_gap_eV",
        "full_final_c60_population",
        "reduced_final_c60_population",
        "final_population_error",
        "population_rmse",
        "population_mae",
        "population_max_abs_error",
        "maximum_full_norm_error",
        "maximum_reduced_norm_error",
    ]
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})

    np.savez_compressed(
        args.outroot / "ensemble_validation.npz",
        time_fs=reference_time,
        full_c60_all=full_all,
        reduced_c60_all=reduced_all,
        full_c60_mean=full_mean,
        reduced_c60_mean=reduced_mean,
        ensemble_error=ensemble_error,
        ensemble_rmse=ensemble_rmse,
    )

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(reference_time, full_mean, linewidth=2.0, label="Full 10-state")
    plt.plot(
        reference_time,
        reduced_mean,
        linewidth=2.0,
        linestyle="--",
        label="Reduced 1D+3A",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel(r"$\langle P_{\mathrm{C60}}(t)\rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        args.outroot / "ensemble_full_vs_1d3a.png",
        dpi=300,
    )
    plt.close()

    print(
        f"{'traj':>5} {'capture':>9} {'min D ov':>9} "
        f"{'min A sv':>9} {'min red sv':>10} "
        f"{'Pfull final':>12} {'Pred final':>12} "
        f"{'RMSE':>9} {'max err':>9}"
    )
    print("-" * 104)

    for summary in summaries:
        print(
            f"{summary['trajectory']:5d} "
            f"{summary['initial_reduced_capture']:9.6f} "
            f"{summary['minimum_donor_transport_overlap']:9.6f} "
            f"{summary['minimum_acceptor_subspace_singular_value']:9.6f} "
            f"{summary['minimum_reduced_overlap_singular_value']:10.6f} "
            f"{summary['full_final_c60_population']:12.6f} "
            f"{summary['reduced_final_c60_population']:12.6f} "
            f"{summary['population_rmse']:9.5f} "
            f"{summary['population_max_abs_error']:9.5f}"
        )

    print("\nEnsemble comparison:")
    print(f"  Full final mean:    {full_mean[-1]:.8f}")
    print(f"  Reduced final mean: {reduced_mean[-1]:.8f}")
    print(f"  Final mean error:   {ensemble_error[-1]:+.8f}")
    print(f"  Ensemble RMSE:      {ensemble_rmse:.8f}")
    print(f"\nWrote validation to {args.outroot}")


if __name__ == "__main__":
    main()
