#!/usr/bin/env python3
"""
Build and validate the selected 4-donor + 3-acceptor reduced model.

The retained seven-dimensional space consists of:
    * the complete four-state donor fragment subspace;
    * the isolated lower three-state C60 subspace.

Both fragment subspaces are parallel-transported between consecutive PySCF
frames.  Two reduced propagations are compared with the full ten-state result:

1. Constructed Hamiltonian:
       H_red(t) = B(t)^dagger H_el(t) B(t)
   with the matrix-log derivative coupling obtained from consecutive reduced
   overlaps.

2. Projected-map control:
       M_n = B_(n+1)^dagger exp[-i H_vib,n dt] B_n
   followed by the closest unitary polar factor.  This is the best closed
   seven-state interval propagation available from the full dynamics and is
   an upper-bound control for the reduction.

The frame-resolved electronic Hamiltonian produced here is the input for the
subsequent fluctuation/correlation analysis used to parameterize TENSO.
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
        default=Path("output_reduced_4d3a"),
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


def parallel_transport_subspace(
    previous_basis: np.ndarray,
    current_raw_basis: np.ndarray,
    overlap_prev_current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate the current basis inside its own subspace so its overlap with the
    previous basis is Hermitian positive.
    """
    cross = (
        previous_basis.conj().T
        @ overlap_prev_current
        @ current_raw_basis
    )
    left, singular_values, right_h = np.linalg.svd(cross)
    rotation = right_h.conj().T @ left.conj().T
    return current_raw_basis @ rotation, singular_values


def fragment_bases(
    energies_h: np.ndarray,
    projector_c60: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the four donor fragment eigenstates and the lower three acceptor
    fragment eigenstates in the tracked ten-state basis.
    """
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

    _, x_d = np.linalg.eigh(h_dd)
    acceptor_energies, x_a = np.linalg.eigh(h_aa)

    donor_basis = u_d @ x_d
    acceptor_basis_low = u_a @ x_a[:, :3]
    upper_acceptor_gap = acceptor_energies[3] - acceptor_energies[2]

    return donor_basis, acceptor_basis_low, upper_acceptor_gap


def propagate_hamiltonian(
    c0: np.ndarray,
    hvib_mid: np.ndarray,
    time_fs: np.ndarray,
) -> np.ndarray:
    nframes = len(time_fs)
    coefficients = np.zeros((nframes, len(c0)), dtype=complex)
    coefficients[0] = normalize(c0)

    for interval in range(nframes - 1):
        dt_au = (
            time_fs[interval + 1] - time_fs[interval]
        ) * FS_TO_AU
        coefficients[interval + 1] = (
            expm(-1j * hermitize(hvib_mid[interval]) * dt_au)
            @ coefficients[interval]
        )

    return coefficients


def propagate_maps(
    c0: np.ndarray,
    maps: np.ndarray,
) -> np.ndarray:
    nframes = maps.shape[0] + 1
    coefficients = np.zeros((nframes, len(c0)), dtype=complex)
    coefficients[0] = normalize(c0)

    for interval, mapping in enumerate(maps):
        coefficients[interval + 1] = mapping @ coefficients[interval]

    return coefficients


def fragment_population(
    coefficients: np.ndarray,
    projectors: np.ndarray,
) -> np.ndarray:
    result = np.empty(coefficients.shape[0], dtype=float)

    for frame, coefficient in enumerate(coefficients):
        value = np.vdot(
            coefficient,
            projectors[frame] @ coefficient,
        )
        if abs(value.imag) > 1.0e-8:
            raise RuntimeError(
                f"Large imaginary fragment population at frame "
                f"{frame}: {value}"
            )
        result[frame] = value.real

    return result


def error_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - reference
    return {
        "error": error,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "final_error": float(error[-1]),
    }


def process_trajectory(
    input_path: Path,
    output_dir: Path,
    cutoff: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=False) as d:
        time_fs = np.asarray(d["time_fs"], dtype=float)
        energies = np.asarray(d["active_energy_hartree"], dtype=float)
        projectors = np.asarray(d["projector_c60"], dtype=complex)
        frame_overlaps = np.asarray(d["polar_overlap"], dtype=complex)
        full_hvib_mid = np.asarray(d["hvib_mid_hartree"], dtype=complex)
        c0_full = np.asarray(d["initial_c0"], dtype=complex)

    nframes, nstates = energies.shape
    nred = 7

    if nstates != 10:
        raise RuntimeError(f"Expected 10 tracked states, found {nstates}")

    basis = np.zeros((nframes, nstates, nred), dtype=complex)
    hel_reduced = np.zeros((nframes, nred, nred), dtype=complex)
    projector_reduced = np.zeros((nframes, nred, nred), dtype=complex)

    donor_subspace_min_sv = np.ones(nframes, dtype=float)
    acceptor_subspace_min_sv = np.ones(nframes, dtype=float)
    basis_orthogonality_error = np.zeros(nframes, dtype=float)
    upper_acceptor_gap_h = np.zeros(nframes, dtype=float)

    previous_donor = None
    previous_acceptor = None

    for frame in range(nframes):
        raw_donor, raw_acceptor, upper_gap = fragment_bases(
            energies[frame],
            projectors[frame],
            cutoff,
        )

        if frame == 0:
            donor_basis = raw_donor
            acceptor_basis = raw_acceptor
        else:
            donor_basis, donor_sv = parallel_transport_subspace(
                previous_donor,
                raw_donor,
                frame_overlaps[frame - 1],
            )
            acceptor_basis, acceptor_sv = parallel_transport_subspace(
                previous_acceptor,
                raw_acceptor,
                frame_overlaps[frame - 1],
            )
            donor_subspace_min_sv[frame] = float(np.min(donor_sv))
            acceptor_subspace_min_sv[frame] = float(
                np.min(acceptor_sv)
            )

        current_basis = np.column_stack(
            (donor_basis, acceptor_basis)
        )
        orth_error = np.linalg.norm(
            current_basis.conj().T @ current_basis
            - np.eye(nred)
        )
        if orth_error > 1.0e-9:
            raise RuntimeError(
                f"Reduced basis orthogonality error at frame "
                f"{frame}: {orth_error:.3e}"
            )

        h_el = np.diag(energies[frame].astype(complex))
        basis[frame] = current_basis
        hel_reduced[frame] = hermitize(
            current_basis.conj().T @ h_el @ current_basis
        )
        projector_reduced[frame] = hermitize(
            current_basis.conj().T
            @ projectors[frame]
            @ current_basis
        )

        basis_orthogonality_error[frame] = orth_error
        upper_acceptor_gap_h[frame] = upper_gap
        previous_donor = donor_basis
        previous_acceptor = acceptor_basis

    reduced_overlap = np.zeros(
        (nframes - 1, nred, nred), dtype=complex
    )
    reduced_polar_overlap = np.zeros_like(reduced_overlap)
    reduced_overlap_sv = np.zeros((nframes - 1, nred), dtype=float)
    reduced_derivative_coupling = np.zeros_like(reduced_overlap)
    reduced_hvib_mid = np.zeros_like(reduced_overlap)

    projected_maps = np.zeros_like(reduced_overlap)
    projected_unitary_maps = np.zeros_like(reduced_overlap)
    projected_map_sv = np.zeros((nframes - 1, nred), dtype=float)
    projected_effective_hamiltonian = np.zeros_like(reduced_overlap)

    full_interval_propagators = np.zeros(
        (nframes - 1, nstates, nstates), dtype=complex
    )

    for interval in range(nframes - 1):
        dt_au = (
            time_fs[interval + 1] - time_fs[interval]
        ) * FS_TO_AU

        s_red = (
            basis[interval].conj().T
            @ frame_overlaps[interval]
            @ basis[interval + 1]
        )
        q_overlap, overlap_sv = polar_factor(s_red)
        d_red = logm(q_overlap) / dt_au
        d_red = 0.5 * (d_red - d_red.conj().T)

        h_red_mid = (
            0.5
            * (
                hel_reduced[interval]
                + hel_reduced[interval + 1]
            )
            - 1j * d_red
        )
        h_red_mid = hermitize(h_red_mid)

        u_full = expm(
            -1j * hermitize(full_hvib_mid[interval]) * dt_au
        )
        projected = (
            basis[interval + 1].conj().T
            @ u_full
            @ basis[interval]
        )
        q_projected, map_sv = polar_factor(projected)

        # exp(-i H dt) = Q  =>  H = i log(Q) / dt
        h_projected = hermitize(1j * logm(q_projected) / dt_au)

        reduced_overlap[interval] = s_red
        reduced_polar_overlap[interval] = q_overlap
        reduced_overlap_sv[interval] = overlap_sv
        reduced_derivative_coupling[interval] = d_red
        reduced_hvib_mid[interval] = h_red_mid

        projected_maps[interval] = projected
        projected_unitary_maps[interval] = q_projected
        projected_map_sv[interval] = map_sv
        projected_effective_hamiltonian[interval] = h_projected
        full_interval_propagators[interval] = u_full

    c0_reduced_raw = basis[0].conj().T @ c0_full
    initial_capture = float(
        np.vdot(c0_reduced_raw, c0_reduced_raw).real
    )
    c0_reduced = normalize(c0_reduced_raw)

    full_coefficients = propagate_hamiltonian(
        c0_full,
        full_hvib_mid,
        time_fs,
    )
    constructed_coefficients = propagate_hamiltonian(
        c0_reduced,
        reduced_hvib_mid,
        time_fs,
    )
    projected_coefficients = propagate_maps(
        c0_reduced,
        projected_unitary_maps,
    )

    full_c60 = fragment_population(
        full_coefficients,
        projectors,
    )
    constructed_c60 = fragment_population(
        constructed_coefficients,
        projector_reduced,
    )
    projected_c60 = fragment_population(
        projected_coefficients,
        projector_reduced,
    )

    exact_capture = np.zeros(nframes, dtype=float)
    for frame in range(nframes):
        projected_exact = basis[frame].conj().T @ full_coefficients[frame]
        exact_capture[frame] = np.vdot(
            projected_exact, projected_exact
        ).real

    constructed_metrics = error_metrics(
        full_c60, constructed_c60
    )
    projected_metrics = error_metrics(
        full_c60, projected_c60
    )

    np.savez_compressed(
        output_dir / "model_4d3a.npz",
        time_fs=time_fs,
        basis_vectors=basis,
        electronic_hamiltonian_hartree=hel_reduced,
        projector_c60_reduced=projector_reduced,
        upper_acceptor_gap_hartree=upper_acceptor_gap_h,
        donor_subspace_min_singular_value=donor_subspace_min_sv,
        acceptor_subspace_min_singular_value=acceptor_subspace_min_sv,
        basis_orthogonality_error=basis_orthogonality_error,
        reduced_overlap=reduced_overlap,
        reduced_polar_overlap=reduced_polar_overlap,
        reduced_overlap_singular_values=reduced_overlap_sv,
        reduced_derivative_coupling_au=reduced_derivative_coupling,
        reduced_hvib_mid_hartree=reduced_hvib_mid,
        projected_interval_maps=projected_maps,
        projected_unitary_maps=projected_unitary_maps,
        projected_map_singular_values=projected_map_sv,
        projected_effective_hamiltonian_mid_hartree=(
            projected_effective_hamiltonian
        ),
        initial_c0_reduced=c0_reduced,
        initial_reduced_capture=initial_capture,
        exact_wavefunction_capture=exact_capture,
        full_coefficients=full_coefficients,
        constructed_coefficients=constructed_coefficients,
        projected_map_coefficients=projected_coefficients,
        full_c60_population=full_c60,
        constructed_c60_population=constructed_c60,
        projected_map_c60_population=projected_c60,
        constructed_population_error=constructed_metrics["error"],
        projected_map_population_error=projected_metrics["error"],
    )

    with (output_dir / "validation_timeseries.csv").open(
        "w", newline=""
    ) as handle:
        fieldnames = [
            "frame",
            "time_fs",
            "full_c60_population",
            "constructed_c60_population",
            "projected_map_c60_population",
            "constructed_error",
            "projected_map_error",
            "exact_wavefunction_capture",
            "donor_subspace_min_singular_value",
            "acceptor_subspace_min_singular_value",
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
                    "constructed_c60_population": constructed_c60[frame],
                    "projected_map_c60_population": projected_c60[frame],
                    "constructed_error": (
                        constructed_metrics["error"][frame]
                    ),
                    "projected_map_error": (
                        projected_metrics["error"][frame]
                    ),
                    "exact_wavefunction_capture": exact_capture[frame],
                    "donor_subspace_min_singular_value": (
                        donor_subspace_min_sv[frame]
                    ),
                    "acceptor_subspace_min_singular_value": (
                        acceptor_subspace_min_sv[frame]
                    ),
                    "upper_acceptor_gap_eV": (
                        upper_acceptor_gap_h[frame]
                        * HARTREE_TO_EV
                    ),
                }
            )

    summary = {
        "input": str(input_path),
        "model": "4 donor states + lower 3-state C60 manifold",
        "nframes": nframes,
        "initial_reduced_capture": initial_capture,
        "minimum_exact_wavefunction_capture": float(
            np.min(exact_capture)
        ),
        "minimum_donor_subspace_transport_singular_value": float(
            np.min(donor_subspace_min_sv[1:])
        ),
        "minimum_acceptor_subspace_transport_singular_value": float(
            np.min(acceptor_subspace_min_sv[1:])
        ),
        "minimum_reduced_overlap_singular_value": float(
            np.min(reduced_overlap_sv)
        ),
        "minimum_projected_map_singular_value": float(
            np.min(projected_map_sv)
        ),
        "mean_upper_acceptor_gap_eV": float(
            np.mean(upper_acceptor_gap_h) * HARTREE_TO_EV
        ),
        "full_final_c60_population": float(full_c60[-1]),
        "constructed_final_c60_population": float(
            constructed_c60[-1]
        ),
        "projected_map_final_c60_population": float(
            projected_c60[-1]
        ),
        "constructed_rmse": constructed_metrics["rmse"],
        "constructed_mae": constructed_metrics["mae"],
        "constructed_max_abs_error": (
            constructed_metrics["max_abs_error"]
        ),
        "constructed_final_error": (
            constructed_metrics["final_error"]
        ),
        "projected_map_rmse": projected_metrics["rmse"],
        "projected_map_mae": projected_metrics["mae"],
        "projected_map_max_abs_error": (
            projected_metrics["max_abs_error"]
        ),
        "projected_map_final_error": (
            projected_metrics["final_error"]
        ),
        "maximum_full_norm_error": float(
            np.max(
                np.abs(
                    np.sum(np.abs(full_coefficients) ** 2, axis=1)
                    - 1.0
                )
            )
        ),
        "maximum_constructed_norm_error": float(
            np.max(
                np.abs(
                    np.sum(
                        np.abs(constructed_coefficients) ** 2,
                        axis=1,
                    )
                    - 1.0
                )
            )
        ),
        "maximum_projected_map_norm_error": float(
            np.max(
                np.abs(
                    np.sum(
                        np.abs(projected_coefficients) ** 2,
                        axis=1,
                    )
                    - 1.0
                )
            )
        ),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    return (
        summary,
        time_fs,
        full_c60,
        constructed_c60,
        projected_c60,
    )


def main() -> None:
    args = parse_args()
    args.outroot.mkdir(parents=True, exist_ok=True)

    summaries = []
    full_all = []
    constructed_all = []
    projected_all = []
    reference_time = None

    for trajectory in range(args.ntraj):
        input_path = (
            args.root
            / f"traj_{trajectory:02d}"
            / "tracked_data.npz"
        )
        output_dir = args.outroot / f"traj_{trajectory:02d}"

        (
            summary,
            time_fs,
            full_c60,
            constructed_c60,
            projected_c60,
        ) = process_trajectory(
            input_path,
            output_dir,
            args.projector_cutoff,
        )

        summary["trajectory"] = trajectory
        summaries.append(summary)
        full_all.append(full_c60)
        constructed_all.append(constructed_c60)
        projected_all.append(projected_c60)

        if reference_time is None:
            reference_time = time_fs
        elif not np.array_equal(reference_time, time_fs):
            raise RuntimeError("Trajectory time grids differ")

    full_all = np.asarray(full_all)
    constructed_all = np.asarray(constructed_all)
    projected_all = np.asarray(projected_all)

    full_mean = np.mean(full_all, axis=0)
    constructed_mean = np.mean(constructed_all, axis=0)
    projected_mean = np.mean(projected_all, axis=0)

    constructed_ensemble = error_metrics(
        full_mean, constructed_mean
    )
    projected_ensemble = error_metrics(
        full_mean, projected_mean
    )

    fields = [
        "trajectory",
        "initial_reduced_capture",
        "minimum_exact_wavefunction_capture",
        "minimum_donor_subspace_transport_singular_value",
        "minimum_acceptor_subspace_transport_singular_value",
        "minimum_reduced_overlap_singular_value",
        "minimum_projected_map_singular_value",
        "mean_upper_acceptor_gap_eV",
        "full_final_c60_population",
        "constructed_final_c60_population",
        "projected_map_final_c60_population",
        "constructed_rmse",
        "projected_map_rmse",
        "constructed_max_abs_error",
        "projected_map_max_abs_error",
    ]

    with (args.outroot / "aggregate_validation.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {field: summary[field] for field in fields}
            )

    np.savez_compressed(
        args.outroot / "ensemble_validation.npz",
        time_fs=reference_time,
        full_c60_all=full_all,
        constructed_c60_all=constructed_all,
        projected_map_c60_all=projected_all,
        full_c60_mean=full_mean,
        constructed_c60_mean=constructed_mean,
        projected_map_c60_mean=projected_mean,
        constructed_ensemble_error=constructed_ensemble["error"],
        projected_map_ensemble_error=projected_ensemble["error"],
    )

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        reference_time,
        full_mean,
        linewidth=2.4,
        label="Full 10-state",
    )
    plt.plot(
        reference_time,
        constructed_mean,
        linewidth=2.0,
        linestyle="--",
        label="Constructed 4D+3A Hamiltonian",
    )
    plt.plot(
        reference_time,
        projected_mean,
        linewidth=1.8,
        linestyle=":",
        label="Projected-map control",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel(r"$\langle P_{\mathrm{C60}}(t)\rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        args.outroot / "ensemble_full_vs_4d3a.png",
        dpi=300,
    )
    plt.close()

    print(
        f"{'traj':>5} {'init cap':>9} {'min cap':>9} "
        f"{'min O sv':>9} {'min M sv':>9} "
        f"{'Pfull f':>9} {'PH f':>9} {'PM f':>9} "
        f"{'H RMSE':>9} {'M RMSE':>9}"
    )
    print("-" * 111)

    for summary in summaries:
        print(
            f"{summary['trajectory']:5d} "
            f"{summary['initial_reduced_capture']:9.6f} "
            f"{summary['minimum_exact_wavefunction_capture']:9.6f} "
            f"{summary['minimum_reduced_overlap_singular_value']:9.6f} "
            f"{summary['minimum_projected_map_singular_value']:9.6f} "
            f"{summary['full_final_c60_population']:9.5f} "
            f"{summary['constructed_final_c60_population']:9.5f} "
            f"{summary['projected_map_final_c60_population']:9.5f} "
            f"{summary['constructed_rmse']:9.5f} "
            f"{summary['projected_map_rmse']:9.5f}"
        )

    print("\nEnsemble comparison:")
    print(f"  Full final mean:        {full_mean[-1]:.8f}")
    print(f"  Constructed final mean: {constructed_mean[-1]:.8f}")
    print(f"  Projected final mean:   {projected_mean[-1]:.8f}")
    print(
        f"  Constructed RMSE:       "
        f"{constructed_ensemble['rmse']:.8f}"
    )
    print(
        f"  Projected-map RMSE:     "
        f"{projected_ensemble['rmse']:.8f}"
    )
    print(f"\nWrote validated 4D+3A models to {args.outroot}")


if __name__ == "__main__":
    main()
