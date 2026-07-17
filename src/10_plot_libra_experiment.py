#!/usr/bin/env python3
r"""
Build deadline-ready Libra/experiment comparison figures.

The script computes three C60 estimators for each classical nuclear path:

1. coherent estimator
       P_coh = c^\dagger P_C60 c

2. active-surface diagonal diagnostic
       P_diag = sum_i p_i^SH (P_C60)_ii

3. paper-style hybrid CPA-FSSH estimator
       P_hybrid = P_diag
                  + sum_{i != j} c_i^* c_j (P_C60)_ij

The third estimator follows the population convention used in eqs. 12-14
of Yamijala and Huo, J. Phys. Chem. A 2021, 125, 628-635.  It uses
surface-hopping populations on the density-matrix diagonal and coherent
electronic amplitudes off the diagonal.

Expected inputs:
    output_libra/fssh/traj_XX/libra_fssh.npz
    output_tracked/traj_XX/tracked_data.npz
    experiment_shg_digitized.csv

Outputs:
    fig_libra_hybrid_ensemble.png
    fig_libra_estimators.png
    fig_libra_vs_experiment.png
    libra_experiment_comparison.csv
    comparison_metrics.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libra-root",
        type=Path,
        default=Path("output_libra/fssh"),
    )
    parser.add_argument(
        "--tracked-root",
        type=Path,
        default=Path("output_tracked"),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("experiment_shg_digitized.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_libra/final_figures"),
    )
    parser.add_argument("--ntraj", type=int, default=10)
    parser.add_argument(
        "--comparison-max-fs",
        type=float,
        default=99.5,
    )
    return parser.parse_args()


def mean_sem_ci95(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = values.shape[0]
    mean = np.mean(values, axis=0)

    if n < 2:
        zeros = np.zeros_like(mean)
        return mean, zeros, zeros

    sem = np.std(values, axis=0, ddof=1) / np.sqrt(n)
    ci = student_t.ppf(0.975, n - 1) * sem
    return mean, sem, ci


def first_crossing(time_fs: np.ndarray, values: np.ndarray, level: float) -> float:
    indices = np.flatnonzero(values >= level)
    return float(time_fs[indices[0]]) if indices.size else float("nan")


def load_experiment(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times = []
    populations = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time_fs", "c60_charge_population"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: "
                f"time_fs,c60_charge_population"
            )

        for row in reader:
            times.append(float(row["time_fs"]))
            populations.append(float(row["c60_charge_population"]))

    if not times:
        raise ValueError(f"No experimental data found in {path}")

    return np.asarray(times, dtype=float), np.asarray(populations, dtype=float)


def load_trajectory(
    libra_root: Path,
    tracked_root: Path,
    trajectory: int,
) -> dict[str, np.ndarray]:
    libra_path = (
        libra_root
        / f"traj_{trajectory:02d}"
        / "libra_fssh.npz"
    )
    tracked_path = (
        tracked_root
        / f"traj_{trajectory:02d}"
        / "tracked_data.npz"
    )

    if not libra_path.is_file():
        raise FileNotFoundError(libra_path)
    if not tracked_path.is_file():
        raise FileNotFoundError(tracked_path)

    with np.load(libra_path, allow_pickle=False) as data:
        time_fs = np.asarray(data["time_fs"], dtype=float)
        coefficients = np.asarray(data["coefficients"], dtype=complex)
        se_state = np.asarray(data["se_state_population"], dtype=float)
        sh_state = np.asarray(data["sh_state_population"], dtype=float)
        coherent = np.asarray(data["se_c60_population"], dtype=float)
        active_diagonal = np.asarray(
            data["active_surface_c60_diagonal"],
            dtype=float,
        )

    with np.load(tracked_path, allow_pickle=False) as data:
        tracked_time = np.asarray(data["time_fs"], dtype=float)
        projector = np.asarray(data["projector_c60"], dtype=complex)

    nframes = len(time_fs)
    if not np.array_equal(time_fs, tracked_time[:nframes]):
        raise RuntimeError(
            f"Time-grid mismatch for trajectory {trajectory:02d}"
        )

    projector = projector[:nframes]
    if projector.shape[:2] != (nframes, coefficients.shape[1]):
        raise RuntimeError(
            f"Projector shape mismatch for trajectory {trajectory:02d}: "
            f"{projector.shape}"
        )

    projector_diagonal = np.real(
        np.diagonal(projector, axis1=1, axis2=2)
    )

    coherent_diagonal = np.sum(se_state * projector_diagonal, axis=1)
    coherent_offdiagonal = coherent - coherent_diagonal

    # Eq. 12-14 style estimator:
    # active-surface diagonal + coefficient-based off-diagonal coherence.
    hybrid = active_diagonal + coherent_offdiagonal

    # Direct reconstruction for consistency checking.
    density = np.einsum(
        "ti,tj->tij",
        coefficients.conj(),
        coefficients,
    )
    indices = np.arange(coefficients.shape[1])
    density[:, indices, indices] = sh_state
    hybrid_direct = np.real(
        np.einsum("tij,tji->t", density, projector)
    )

    discrepancy = float(np.max(np.abs(hybrid - hybrid_direct)))
    if discrepancy > 1.0e-10:
        raise RuntimeError(
            f"Hybrid reconstruction mismatch for trajectory "
            f"{trajectory:02d}: {discrepancy:.3e}"
        )

    return {
        "time_fs": time_fs,
        "coherent": coherent,
        "active_diagonal": active_diagonal,
        "coherent_diagonal": coherent_diagonal,
        "coherent_offdiagonal": coherent_offdiagonal,
        "hybrid": hybrid,
    }


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    trajectories = [
        load_trajectory(
            args.libra_root,
            args.tracked_root,
            trajectory,
        )
        for trajectory in range(args.ntraj)
    ]

    time_fs = trajectories[0]["time_fs"]
    for trajectory, data in enumerate(trajectories[1:], start=1):
        if not np.array_equal(time_fs, data["time_fs"]):
            raise RuntimeError(
                f"Time-grid mismatch in trajectory {trajectory:02d}"
            )

    coherent_all = np.stack([data["coherent"] for data in trajectories])
    active_all = np.stack(
        [data["active_diagonal"] for data in trajectories]
    )
    offdiagonal_all = np.stack(
        [data["coherent_offdiagonal"] for data in trajectories]
    )
    hybrid_all = np.stack([data["hybrid"] for data in trajectories])

    coherent_mean, coherent_sem, coherent_ci = mean_sem_ci95(coherent_all)
    active_mean, active_sem, active_ci = mean_sem_ci95(active_all)
    offdiag_mean, offdiag_sem, offdiag_ci = mean_sem_ci95(offdiagonal_all)
    hybrid_mean, hybrid_sem, hybrid_ci = mean_sem_ci95(hybrid_all)

    exp_time, exp_population = load_experiment(args.experiment)
    comparison_mask = time_fs <= args.comparison_max_fs
    comparison_time = time_fs[comparison_mask]
    experiment_on_libra_grid = np.interp(
        comparison_time,
        exp_time,
        exp_population,
    )

    hybrid_error = hybrid_mean[comparison_mask] - experiment_on_libra_grid
    coherent_error = (
        coherent_mean[comparison_mask] - experiment_on_libra_grid
    )

    hybrid_rmse = float(np.sqrt(np.mean(hybrid_error**2)))
    coherent_rmse = float(np.sqrt(np.mean(coherent_error**2)))
    hybrid_mae = float(np.mean(np.abs(hybrid_error)))
    coherent_mae = float(np.mean(np.abs(coherent_error)))

    # Figure 1: individual hybrid trajectories plus ensemble uncertainty.
    plt.figure(figsize=(7.2, 4.8))
    for curve in hybrid_all:
        plt.plot(time_fs, curve, linewidth=0.8, alpha=0.30)
    plt.plot(
        time_fs,
        hybrid_mean,
        linewidth=2.2,
        label="CPA-FSSH hybrid mean",
    )
    plt.fill_between(
        time_fs,
        hybrid_mean - hybrid_ci,
        hybrid_mean + hybrid_ci,
        alpha=0.20,
        label="95% t interval across nuclear paths",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel("C60 charge population")
    plt.xlim(0.0, args.comparison_max_fs)
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=False)
    save_figure(args.outdir / "fig_libra_hybrid_ensemble.png")

    # Figure 2: estimator decomposition.
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        time_fs,
        hybrid_mean,
        linewidth=2.2,
        label="Hybrid CPA-FSSH estimator",
    )
    plt.plot(
        time_fs,
        coherent_mean,
        linewidth=1.8,
        linestyle="--",
        label=r"Coherent $c^\dagger P_{\mathrm{C60}}c$",
    )
    plt.plot(
        time_fs,
        active_mean,
        linewidth=1.6,
        linestyle=":",
        label="Active-surface diagonal term",
    )
    plt.plot(
        time_fs,
        offdiag_mean,
        linewidth=1.3,
        linestyle="-.",
        label="Coherent off-diagonal term",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel("C60 population contribution")
    plt.xlim(0.0, args.comparison_max_fs)
    plt.legend(frameon=False)
    save_figure(args.outdir / "fig_libra_estimators.png")

    # Figure 3: direct raw-population comparison with digitized experiment.
    plt.figure(figsize=(7.2, 4.8))
    experiment_mask = exp_time <= args.comparison_max_fs
    plt.plot(
        exp_time[experiment_mask],
        exp_population[experiment_mask],
        linewidth=2.4,
        label="Experiment (digitized SHG)",
    )
    plt.plot(
        time_fs,
        hybrid_mean,
        linewidth=2.2,
        label="This work: CPA-FSSH hybrid mean",
    )
    plt.fill_between(
        time_fs,
        hybrid_mean - hybrid_ci,
        hybrid_mean + hybrid_ci,
        alpha=0.20,
        label="95% t interval",
    )
    plt.plot(
        time_fs,
        coherent_mean,
        linewidth=1.5,
        linestyle="--",
        label=r"Coherent $c^\dagger P_{\mathrm{C60}}c$ diagnostic",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel("C60 charge population")
    plt.xlim(0.0, args.comparison_max_fs)
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=False)
    save_figure(args.outdir / "fig_libra_vs_experiment.png")

    # Save a complete analysis table.
    csv_path = args.outdir / "libra_experiment_comparison.csv"
    fields = [
        "time_fs",
        "experiment_digitized",
        "hybrid_mean",
        "hybrid_sem",
        "hybrid_ci95_low",
        "hybrid_ci95_high",
        "coherent_mean",
        "coherent_sem",
        "active_surface_diagonal_mean",
        "coherent_offdiagonal_mean",
    ]

    experiment_full_grid = np.interp(time_fs, exp_time, exp_population)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame, time in enumerate(time_fs):
            writer.writerow(
                {
                    "time_fs": float(time),
                    "experiment_digitized": float(
                        experiment_full_grid[frame]
                    ),
                    "hybrid_mean": float(hybrid_mean[frame]),
                    "hybrid_sem": float(hybrid_sem[frame]),
                    "hybrid_ci95_low": float(
                        hybrid_mean[frame] - hybrid_ci[frame]
                    ),
                    "hybrid_ci95_high": float(
                        hybrid_mean[frame] + hybrid_ci[frame]
                    ),
                    "coherent_mean": float(coherent_mean[frame]),
                    "coherent_sem": float(coherent_sem[frame]),
                    "active_surface_diagonal_mean": float(
                        active_mean[frame]
                    ),
                    "coherent_offdiagonal_mean": float(
                        offdiag_mean[frame]
                    ),
                }
            )

    endpoint = int(np.flatnonzero(comparison_mask)[-1])
    exp_endpoint = float(experiment_on_libra_grid[-1])

    metrics = f"""Libra versus digitized experiment
===================================

Comparison interval: 0.0-{comparison_time[-1]:.1f} fs
Number of nuclear trajectories: {args.ntraj}

Paper-style hybrid CPA-FSSH estimator
  initial population: {hybrid_mean[0]:.8f}
  final population:   {hybrid_mean[endpoint]:.8f}
  experiment final:   {exp_endpoint:.8f}
  RMSE:               {hybrid_rmse:.8f}
  MAE:                {hybrid_mae:.8f}
  first P >= 0.25:    {first_crossing(time_fs, hybrid_mean, 0.25):.2f} fs
  first P >= 0.50:    {first_crossing(time_fs, hybrid_mean, 0.50):.2f} fs

Pure coherent estimator
  initial population: {coherent_mean[0]:.8f}
  final population:   {coherent_mean[endpoint]:.8f}
  RMSE:               {coherent_rmse:.8f}
  MAE:                {coherent_mae:.8f}
  first P >= 0.25:    {first_crossing(time_fs, coherent_mean, 0.25):.2f} fs
  first P >= 0.50:    {first_crossing(time_fs, coherent_mean, 0.50):.2f} fs

Important:
  The experimental series is an approximate digitization of Figure 3A,
  not original raw SHG data. The hybrid estimator is the apples-to-apples
  comparison with the population convention used in the reference paper.
"""
    (args.outdir / "comparison_metrics.txt").write_text(metrics)
    print(metrics)
    print(f"Wrote figures and tables to {args.outdir}")


if __name__ == "__main__":
    main()
