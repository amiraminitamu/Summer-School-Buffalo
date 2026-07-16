#!/usr/bin/env python3
"""
Diagnose a hierarchy of fragment-reduced models against the full ten-state
electronic dynamics.

Models tested:
    1D+3A, 2D+3A, 2D+6A, 4D+3A, 4D+6A

The validation uses the exact interval propagator of the stored full
ten-state vibronic Hamiltonian,

    U_n = exp[-i H_vib,n dt],

and projects it between consecutive reduced bases,

    M_n = B_{n+1}^\dagger U_n B_n.

For an incomplete reduced space, singular values of M_n below one measure
true dynamical leakage into discarded states.  The polar unitary part of M_n
defines the closest closed reduced propagation.  The 4D+6A model spans the
entire ten-state space and therefore serves as an exact code check.

This is a diagnostic/validation step, not a bath fit.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, polar

FS_TO_AU = 41.3413745758

MODELS = {
    "1D3A": (1, 3),
    "2D3A": (2, 3),
    "2D6A": (2, 6),
    "4D3A": (4, 3),
    "4D6A": (4, 6),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("output_tracked"))
    p.add_argument(
        "--outroot",
        type=Path,
        default=Path("output_reduced_hierarchy"),
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


def fragment_eigenstates(
    energies_h: np.ndarray,
    projector_c60: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = hermitize(projector_c60)
    pvals, pvecs = np.linalg.eigh(p)

    donor_mask = pvals < cutoff
    acceptor_mask = ~donor_mask
    if donor_mask.sum() != 4 or acceptor_mask.sum() != 6:
        raise RuntimeError(
            f"Expected a 4D donor and 6D acceptor split; got "
            f"{donor_mask.sum()} and {acceptor_mask.sum()}. "
            f"Eigenvalues: {pvals}"
        )

    u_d = pvecs[:, donor_mask]
    u_a = pvecs[:, acceptor_mask]
    h = np.diag(np.asarray(energies_h, dtype=complex))

    h_dd = hermitize(u_d.conj().T @ h @ u_d)
    h_aa = hermitize(u_a.conj().T @ h @ u_a)

    _, x_d = np.linalg.eigh(h_dd)
    _, x_a = np.linalg.eigh(h_aa)

    return u_d @ x_d, u_a @ x_a


def choose_initial_donors(
    donor_candidates: np.ndarray,
    c0: np.ndarray,
    ndonor: int,
) -> np.ndarray:
    weights = np.abs(donor_candidates.conj().T @ c0) ** 2
    chosen = np.argsort(weights)[-ndonor:]
    # Put the most strongly occupied state first for readability.
    chosen = chosen[np.argsort(weights[chosen])[::-1]]
    return donor_candidates[:, chosen]


def choose_continuous_subset(
    previous_basis: np.ndarray,
    candidates: np.ndarray,
    overlap_prev_current: np.ndarray,
    nchoose: int,
) -> np.ndarray:
    """
    Select the candidate subset whose subspace has the greatest overlap with
    the previously selected subspace, then parallel-transport its internal
    gauge.
    """
    ncandidates = candidates.shape[1]
    if nchoose == ncandidates:
        selected = candidates
    else:
        best_score = -np.inf
        selected = None
        for combo in itertools.combinations(range(ncandidates), nchoose):
            trial = candidates[:, combo]
            cross = (
                previous_basis.conj().T
                @ overlap_prev_current
                @ trial
            )
            singular_values = np.linalg.svd(
                cross, compute_uv=False
            )
            score = float(np.sum(singular_values**2))
            if score > best_score:
                best_score = score
                selected = trial

        if selected is None:
            raise RuntimeError("No candidate subset was selected")

    cross = previous_basis.conj().T @ overlap_prev_current @ selected
    left, _, right_h = np.linalg.svd(cross)
    rotation = right_h.conj().T @ left.conj().T
    return selected @ rotation


def build_model_bases(
    energies: np.ndarray,
    projectors: np.ndarray,
    frame_overlaps: np.ndarray,
    c0: np.ndarray,
    ndonor: int,
    nacceptor: int,
    cutoff: float,
) -> np.ndarray:
    nframes, nstates = energies.shape
    nred = ndonor + nacceptor
    bases = np.zeros((nframes, nstates, nred), dtype=complex)

    previous_d = None
    previous_a = None

    for frame in range(nframes):
        donor_candidates, acceptor_candidates = fragment_eigenstates(
            energies[frame],
            projectors[frame],
            cutoff,
        )

        if nacceptor == 3:
            acceptor_candidates = acceptor_candidates[:, :3]

        if frame == 0:
            donor_basis = choose_initial_donors(
                donor_candidates, c0, ndonor
            )
            acceptor_basis = acceptor_candidates
        else:
            donor_basis = choose_continuous_subset(
                previous_d,
                donor_candidates,
                frame_overlaps[frame - 1],
                ndonor,
            )
            acceptor_basis = choose_continuous_subset(
                previous_a,
                acceptor_candidates,
                frame_overlaps[frame - 1],
                nacceptor,
            )

        basis = np.column_stack((donor_basis, acceptor_basis))
        orth_error = np.linalg.norm(
            basis.conj().T @ basis - np.eye(nred)
        )
        if orth_error > 1.0e-9:
            raise RuntimeError(
                f"Reduced basis lost orthogonality at frame {frame}: "
                f"{orth_error:.3e}"
            )

        bases[frame] = basis
        previous_d = donor_basis
        previous_a = acceptor_basis

    return bases


def propagate_full(
    c0: np.ndarray,
    hvib_mid: np.ndarray,
    time_fs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nframes = len(time_fs)
    nstates = len(c0)
    coeff = np.zeros((nframes, nstates), dtype=complex)
    propagators = np.zeros((nframes - 1, nstates, nstates), dtype=complex)

    coeff[0] = normalize(c0)
    for interval in range(nframes - 1):
        dt_au = (
            time_fs[interval + 1] - time_fs[interval]
        ) * FS_TO_AU
        u = expm(-1j * hermitize(hvib_mid[interval]) * dt_au)
        propagators[interval] = u
        coeff[interval + 1] = u @ coeff[interval]

    return coeff, propagators


def fragment_population(
    coefficients: np.ndarray,
    projectors: np.ndarray,
) -> np.ndarray:
    pop = np.empty(coefficients.shape[0], dtype=float)
    for frame, c in enumerate(coefficients):
        pop[frame] = np.vdot(
            c, projectors[frame] @ c
        ).real
    return pop


def validate_model(
    bases: np.ndarray,
    full_coeff: np.ndarray,
    full_propagators: np.ndarray,
    projectors: np.ndarray,
    time_fs: np.ndarray,
) -> dict:
    nframes = len(time_fs)
    nred = bases.shape[2]

    reduced_projectors = np.zeros(
        (nframes, nred, nred), dtype=complex
    )
    for frame in range(nframes):
        reduced_projectors[frame] = hermitize(
            bases[frame].conj().T
            @ projectors[frame]
            @ bases[frame]
        )

    # Instantaneous capture of the exact full wavefunction.
    capture = np.zeros(nframes, dtype=float)
    projected_full_population = np.zeros(nframes, dtype=float)

    for frame in range(nframes):
        projected = bases[frame].conj().T @ full_coeff[frame]
        capture[frame] = np.vdot(projected, projected).real
        projected_full_population[frame] = np.vdot(
            projected,
            reduced_projectors[frame] @ projected,
        ).real

    # Exact projected interval maps and their closest unitary factors.
    maps = np.zeros((nframes - 1, nred, nred), dtype=complex)
    unitary_maps = np.zeros_like(maps)
    singular_values = np.zeros((nframes - 1, nred), dtype=float)

    for interval in range(nframes - 1):
        m = (
            bases[interval + 1].conj().T
            @ full_propagators[interval]
            @ bases[interval]
        )
        q, _ = polar(m)
        maps[interval] = m
        unitary_maps[interval] = q
        singular_values[interval] = np.linalg.svd(
            m, compute_uv=False
        )

    # Closed reduced propagation: closest unitary projected dynamics.
    a0_raw = bases[0].conj().T @ full_coeff[0]
    initial_capture = float(np.vdot(a0_raw, a0_raw).real)
    a_closed = np.zeros((nframes, nred), dtype=complex)
    a_closed[0] = normalize(a0_raw)

    # Open projected propagation: retain leakage as norm loss.
    a_lossy = np.zeros((nframes, nred), dtype=complex)
    a_lossy[0] = a0_raw

    for interval in range(nframes - 1):
        a_closed[interval + 1] = (
            unitary_maps[interval] @ a_closed[interval]
        )
        a_lossy[interval + 1] = maps[interval] @ a_lossy[interval]

    closed_population = fragment_population(
        a_closed, reduced_projectors
    )
    lossy_population = fragment_population(
        a_lossy, reduced_projectors
    )
    lossy_norm = np.sum(np.abs(a_lossy) ** 2, axis=1)

    full_population = fragment_population(
        full_coeff, projectors
    )
    error = closed_population - full_population

    return {
        "initial_capture": initial_capture,
        "instantaneous_capture": capture,
        "projected_full_population": projected_full_population,
        "maps": maps,
        "unitary_maps": unitary_maps,
        "dynamic_singular_values": singular_values,
        "closed_coefficients": a_closed,
        "lossy_coefficients": a_lossy,
        "closed_population": closed_population,
        "lossy_population": lossy_population,
        "lossy_norm": lossy_norm,
        "full_population": full_population,
        "population_error": error,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "final_error": float(error[-1]),
    }


def main() -> None:
    args = parse_args()
    args.outroot.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {
        name: [] for name in MODELS
    }
    reference_time = None
    full_populations = []

    for traj in range(args.ntraj):
        path = args.root / f"traj_{traj:02d}" / "tracked_data.npz"
        with np.load(path, allow_pickle=False) as d:
            time_fs = np.asarray(d["time_fs"], dtype=float)
            energies = np.asarray(
                d["active_energy_hartree"], dtype=float
            )
            projectors = np.asarray(
                d["projector_c60"], dtype=complex
            )
            frame_overlaps = np.asarray(
                d["polar_overlap"], dtype=complex
            )
            hvib_mid = np.asarray(
                d["hvib_mid_hartree"], dtype=complex
            )
            c0 = np.asarray(d["initial_c0"], dtype=complex)

        full_coeff, full_propagators = propagate_full(
            c0, hvib_mid, time_fs
        )
        full_population = fragment_population(
            full_coeff, projectors
        )
        full_populations.append(full_population)

        if reference_time is None:
            reference_time = time_fs
        elif not np.array_equal(reference_time, time_fs):
            raise RuntimeError("Trajectory time grids differ")

        traj_dir = args.outroot / f"traj_{traj:02d}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        for model_name, (ndonor, nacceptor) in MODELS.items():
            bases = build_model_bases(
                energies,
                projectors,
                frame_overlaps,
                c0,
                ndonor,
                nacceptor,
                args.projector_cutoff,
            )
            result = validate_model(
                bases,
                full_coeff,
                full_propagators,
                projectors,
                time_fs,
            )
            all_results[model_name].append(result)

            np.savez_compressed(
                traj_dir / f"{model_name}.npz",
                time_fs=time_fs,
                basis_vectors=bases,
                instantaneous_capture=result[
                    "instantaneous_capture"
                ],
                projected_full_population=result[
                    "projected_full_population"
                ],
                projected_interval_maps=result["maps"],
                projected_unitary_maps=result["unitary_maps"],
                dynamic_singular_values=result[
                    "dynamic_singular_values"
                ],
                closed_population=result["closed_population"],
                lossy_population=result["lossy_population"],
                lossy_norm=result["lossy_norm"],
                full_population=result["full_population"],
                population_error=result["population_error"],
            )

    full_populations = np.asarray(full_populations)
    full_mean = np.mean(full_populations, axis=0)

    rows = []
    ensemble_curves = {"Full10": full_mean}

    print(
        f"{'model':>7} {'init cap':>9} {'min cap':>9} "
        f"{'min dyn sv':>10} {'lossy norm f':>12} "
        f"{'Pfull f':>9} {'Pred f':>9} "
        f"{'ens RMSE':>10} {'mean path RMSE':>15}"
    )
    print("-" * 104)

    for model_name in MODELS:
        results = all_results[model_name]
        closed_all = np.stack(
            [r["closed_population"] for r in results]
        )
        closed_mean = np.mean(closed_all, axis=0)
        ensemble_curves[model_name] = closed_mean

        ensemble_error = closed_mean - full_mean
        ensemble_rmse = float(
            np.sqrt(np.mean(ensemble_error**2))
        )

        row = {
            "model": model_name,
            "mean_initial_capture": float(
                np.mean([r["initial_capture"] for r in results])
            ),
            "minimum_instantaneous_capture": float(
                min(
                    np.min(r["instantaneous_capture"])
                    for r in results
                )
            ),
            "mean_instantaneous_capture": float(
                np.mean(
                    np.concatenate(
                        [
                            r["instantaneous_capture"]
                            for r in results
                        ]
                    )
                )
            ),
            "minimum_dynamic_singular_value": float(
                min(
                    np.min(r["dynamic_singular_values"])
                    for r in results
                )
            ),
            "mean_final_lossy_norm": float(
                np.mean([r["lossy_norm"][-1] for r in results])
            ),
            "full_final_mean": float(full_mean[-1]),
            "closed_final_mean": float(closed_mean[-1]),
            "final_mean_error": float(
                closed_mean[-1] - full_mean[-1]
            ),
            "ensemble_rmse": ensemble_rmse,
            "mean_path_rmse": float(
                np.mean([r["rmse"] for r in results])
            ),
            "maximum_path_error": float(
                max(r["max_abs_error"] for r in results)
            ),
        }
        rows.append(row)

        print(
            f"{model_name:>7} "
            f"{row['mean_initial_capture']:9.6f} "
            f"{row['minimum_instantaneous_capture']:9.6f} "
            f"{row['minimum_dynamic_singular_value']:10.6f} "
            f"{row['mean_final_lossy_norm']:12.6f} "
            f"{row['full_final_mean']:9.5f} "
            f"{row['closed_final_mean']:9.5f} "
            f"{row['ensemble_rmse']:10.5f} "
            f"{row['mean_path_rmse']:15.5f}"
        )

    csv_path = args.outroot / "hierarchy_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.outroot / "hierarchy_ensemble.npz",
        time_fs=reference_time,
        full_mean=full_mean,
        **{
            f"{name}_mean": curve
            for name, curve in ensemble_curves.items()
            if name != "Full10"
        },
    )

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        reference_time,
        full_mean,
        linewidth=2.4,
        label="Full 10-state",
    )
    for model_name in MODELS:
        plt.plot(
            reference_time,
            ensemble_curves[model_name],
            linewidth=1.6,
            label=model_name,
        )
    plt.xlabel("Time (fs)")
    plt.ylabel(r"$\langle P_{\mathrm{C60}}(t)\rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        args.outroot / "hierarchy_ensemble_comparison.png",
        dpi=300,
    )
    plt.close()

    (args.outroot / "README.txt").write_text(
        "The 4D6A model spans the complete ten-state active space and "
        "must reproduce the full result to numerical precision. Models "
        "with low instantaneous capture or projected-map singular values "
        "substantially below one are not dynamically closed.\n"
    )

    print(f"\nWrote hierarchy diagnostics to {args.outroot}")


if __name__ == "__main__":
    main()
