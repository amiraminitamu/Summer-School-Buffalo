#!/usr/bin/env python3
"""Analyze coherent charge-transfer dynamics and transfer pathways.

This script uses the already generated 4D+3A reduced-model files. It does not
run PySCF or Libra. For each nuclear trajectory it evaluates

* numerically exact unitary propagation in the full ten-state Hamiltonian;
* the independently constructed seven-state propagation;
* donor-acceptor coherence and participation measures; and
* pair-resolved probability currents between four donor and three acceptor
  states.

The term "exact" refers only to unitary propagation within the finite,
time-dependent Hamiltonian supplied in each model_4d3a.npz file.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

FS_TO_AU = 41.3413745758
NDONOR = 4
NACCEPTOR = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output_reduced_4d3a"),
        help="Directory containing traj_XX/model_4d3a.npz files.",
    )
    parser.add_argument("--ntraj", type=int, default=10)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("experiment_shg_digitized.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_coherent_mechanism"),
    )
    return parser.parse_args()


def ci95(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros(values.shape[1], dtype=float)
    return 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def block_coherence(coefficients: np.ndarray) -> np.ndarray:
    """Frobenius norm of the donor-acceptor density-matrix block."""
    donor = coefficients[:, :NDONOR]
    acceptor = coefficients[:, NDONOR:]
    rho_da = np.einsum("td,ta->tda", donor, acceptor.conj())
    return np.linalg.norm(rho_da, axis=(1, 2))


def participation_ratio(coefficients: np.ndarray) -> np.ndarray:
    probabilities = np.abs(coefficients) ** 2
    return 1.0 / np.sum(probabilities**2, axis=1)


def interval_currents(
    time_fs: np.ndarray,
    coefficients: np.ndarray,
    hvib_mid_hartree: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pair currents J[d,a] and midpoint times in probability/fs."""
    ninterval = len(time_fs) - 1
    current = np.zeros((ninterval, NDONOR, NACCEPTOR), dtype=float)
    midpoint_time = 0.5 * (time_fs[:-1] + time_fs[1:])

    for interval in range(ninterval):
        dt_fs = time_fs[interval + 1] - time_fs[interval]
        dt_au = dt_fs * FS_TO_AU
        h = 0.5 * (
            hvib_mid_hartree[interval]
            + hvib_mid_hartree[interval].conj().T
        )

        # The Hamiltonian is piecewise constant on each nuclear interval.
        c_mid = expm(-1j * h * (0.5 * dt_au)) @ coefficients[interval]

        for donor in range(NDONOR):
            for acceptor_local in range(NACCEPTOR):
                acceptor = NDONOR + acceptor_local
                z = h[acceptor, donor] * c_mid[donor] * c_mid[acceptor].conj()
                current[interval, donor, acceptor_local] = (
                    2.0 * FS_TO_AU * np.imag(z)
                )

    return current, midpoint_time


def load_experiment(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.array([float(row["time_fs"]) for row in rows]),
        np.array([float(row["c60_charge_population"]) for row in rows]),
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    full_population = []
    reduced_population = []
    partition_population = []
    coherence = []
    participation = []
    currents = []
    trajectory_rmse = []
    flux_identity_rmse = []
    projector_partition_difference = []
    time_fs = None
    midpoint_time = None

    for trajectory in range(args.ntraj):
        path = args.root / f"traj_{trajectory:02d}" / "model_4d3a.npz"
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            current_time = np.asarray(data["time_fs"], dtype=float)
            full = np.asarray(data["full_c60_population"], dtype=float)
            reduced = np.asarray(
                data["constructed_c60_population"], dtype=float
            )
            coefficients = np.asarray(
                data["constructed_coefficients"], dtype=complex
            )
            hvib = np.asarray(
                data["reduced_hvib_mid_hartree"], dtype=complex
            )

        if time_fs is None:
            time_fs = current_time
        elif not np.allclose(current_time, time_fs, atol=1.0e-12, rtol=0.0):
            raise RuntimeError(f"Time grid differs in {path}")

        partition = np.sum(
            np.abs(coefficients[:, NDONOR:]) ** 2,
            axis=1,
        )
        pair_current, current_midpoint = interval_currents(
            current_time,
            coefficients,
            hvib,
        )
        net_current = np.sum(pair_current, axis=(1, 2))
        finite_difference = np.diff(partition) / np.diff(current_time)

        if midpoint_time is None:
            midpoint_time = current_midpoint

        full_population.append(full)
        reduced_population.append(reduced)
        partition_population.append(partition)
        coherence.append(block_coherence(coefficients))
        participation.append(participation_ratio(coefficients))
        currents.append(pair_current)
        trajectory_rmse.append(
            float(np.sqrt(np.mean((reduced - full) ** 2)))
        )
        flux_identity_rmse.append(
            float(np.sqrt(np.mean((net_current - finite_difference) ** 2)))
        )
        projector_partition_difference.append(
            float(np.max(np.abs(reduced - partition)))
        )

    assert time_fs is not None
    assert midpoint_time is not None

    full_population = np.asarray(full_population)
    reduced_population = np.asarray(reduced_population)
    partition_population = np.asarray(partition_population)
    coherence = np.asarray(coherence)
    participation = np.asarray(participation)
    currents = np.asarray(currents)

    full_mean = np.mean(full_population, axis=0)
    reduced_mean = np.mean(reduced_population, axis=0)
    partition_mean = np.mean(partition_population, axis=0)
    coherence_mean = np.mean(coherence, axis=0)
    participation_mean = np.mean(participation, axis=0)
    net_current = np.sum(currents, axis=(2, 3))
    net_current_mean = np.mean(net_current, axis=0)

    dt_fs = np.diff(time_fs)
    signed_transfer = np.sum(currents * dt_fs[None, :, None, None], axis=1)
    positive_transfer = np.sum(
        np.maximum(currents, 0.0) * dt_fs[None, :, None, None], axis=1
    )
    absolute_transfer = np.sum(
        np.abs(currents) * dt_fs[None, :, None, None], axis=1
    )

    ensemble_csv = args.outdir / "coherent_ensemble.csv"
    with ensemble_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_fs",
                "full_10state_c60_mean",
                "full_10state_c60_ci95",
                "reduced_4d3a_c60_mean",
                "reduced_4d3a_c60_ci95",
                "partition_acceptor_mean",
                "donor_acceptor_coherence_mean",
                "donor_acceptor_coherence_ci95",
                "participation_ratio_mean",
            ]
        )
        for row in zip(
            time_fs,
            full_mean,
            ci95(full_population),
            reduced_mean,
            ci95(reduced_population),
            partition_mean,
            coherence_mean,
            ci95(coherence),
            participation_mean,
        ):
            writer.writerow(row)

    flux_csv = args.outdir / "donor_acceptor_channel_flux.csv"
    with flux_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "donor_state",
                "acceptor_state",
                "mean_signed_transfer",
                "std_signed_transfer",
                "mean_positive_transfer",
                "mean_absolute_transfer",
            ]
        )
        for donor in range(NDONOR):
            for acceptor in range(NACCEPTOR):
                values = signed_transfer[:, donor, acceptor]
                writer.writerow(
                    [
                        donor + 1,
                        acceptor + 1,
                        np.mean(values),
                        np.std(values, ddof=1),
                        np.mean(positive_transfer[:, donor, acceptor]),
                        np.mean(absolute_transfer[:, donor, acceptor]),
                    ]
                )

    experiment = load_experiment(args.experiment)

    plt.figure(figsize=(7.2, 4.7))
    plt.plot(time_fs, full_mean, label="Full 10-state coherent")
    plt.fill_between(
        time_fs,
        full_mean - ci95(full_population),
        full_mean + ci95(full_population),
        alpha=0.2,
    )
    plt.plot(time_fs, reduced_mean, linestyle="--", label="Reduced 4D+3A")
    if experiment is not None:
        exp_time, exp_population = experiment
        plt.plot(
            exp_time,
            exp_population,
            marker="o",
            markersize=4,
            label="Digitized SHG experiment",
        )
    plt.xlabel("Time (fs)")
    plt.ylabel(r"$P_{\mathrm{C}_{60}}$")
    plt.xlim(time_fs[0], time_fs[-1])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.outdir / "coherent_full_vs_reduced.pdf")
    plt.savefig(args.outdir / "coherent_full_vs_reduced.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 4.7))
    plt.plot(midpoint_time, net_current_mean)
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Time (fs)")
    plt.ylabel(r"Net donor $\rightarrow$ acceptor current (fs$^{-1}$)")
    plt.xlim(time_fs[0], time_fs[-1])
    plt.tight_layout()
    plt.savefig(args.outdir / "coherent_net_current.pdf")
    plt.savefig(args.outdir / "coherent_net_current.png", dpi=300)
    plt.close()

    labels = []
    channel_values = []
    for donor in range(NDONOR):
        for acceptor in range(NACCEPTOR):
            labels.append(f"D{donor + 1}->A{acceptor + 1}")
            channel_values.append(
                np.mean(positive_transfer[:, donor, acceptor])
            )

    order = np.argsort(channel_values)[::-1]
    plt.figure(figsize=(8.0, 4.8))
    plt.bar(
        np.arange(len(order)),
        np.asarray(channel_values)[order],
    )
    plt.xticks(
        np.arange(len(order)),
        np.asarray(labels)[order],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Integrated positive probability flux")
    plt.tight_layout()
    plt.savefig(args.outdir / "dominant_transfer_channels.pdf")
    plt.savefig(args.outdir / "dominant_transfer_channels.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 4.7))
    plt.plot(time_fs, coherence_mean, label="Donor-acceptor coherence")
    plt.fill_between(
        time_fs,
        coherence_mean - ci95(coherence),
        coherence_mean + ci95(coherence),
        alpha=0.2,
    )
    plt.xlabel("Time (fs)")
    plt.ylabel("Donor-acceptor coherence (Frobenius norm)")
    plt.xlim(time_fs[0], time_fs[-1])
    plt.tight_layout()
    plt.savefig(args.outdir / "donor_acceptor_coherence.pdf")
    plt.savefig(args.outdir / "donor_acceptor_coherence.png", dpi=300)
    plt.close()

    dominant_flat = int(np.argmax(np.mean(positive_transfer, axis=0)))
    dominant_donor, dominant_acceptor = np.unravel_index(
        dominant_flat,
        (NDONOR, NACCEPTOR),
    )

    summary = {
        "n_trajectories": args.ntraj,
        "duration_fs": float(time_fs[-1]),
        "full_10state_final_c60_mean": float(full_mean[-1]),
        "full_10state_final_c60_ci95": float(ci95(full_population)[-1]),
        "reduced_4d3a_final_c60_mean": float(reduced_mean[-1]),
        "reduced_4d3a_final_c60_ci95": float(
            ci95(reduced_population)[-1]
        ),
        "mean_trajectory_reduction_rmse": float(np.mean(trajectory_rmse)),
        "ensemble_curve_reduction_rmse": float(
            np.sqrt(np.mean((reduced_mean - full_mean) ** 2))
        ),
        "maximum_projector_partition_population_difference": float(
            np.max(projector_partition_difference)
        ),
        "mean_flux_continuity_rmse_per_fs": float(
            np.mean(flux_identity_rmse)
        ),
        "maximum_mean_donor_acceptor_coherence": float(
            np.max(coherence_mean)
        ),
        "time_of_maximum_mean_coherence_fs": float(
            time_fs[np.argmax(coherence_mean)]
        ),
        "dominant_positive_flux_channel": {
            "donor_state": int(dominant_donor + 1),
            "acceptor_state": int(dominant_acceptor + 1),
            "mean_integrated_positive_flux": float(
                np.mean(
                    positive_transfer[
                        :, dominant_donor, dominant_acceptor
                    ]
                )
            ),
        },
        "interpretation": (
            "Numerically exact unitary propagation within the trajectory-specific "
            "ten-state Hamiltonians, with an independently constructed 4D+3A "
            "reduced-model control and pair-resolved donor-acceptor probability flux."
        ),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {ensemble_csv}")
    print(f"Wrote {flux_csv}")
    print(f"Wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()
