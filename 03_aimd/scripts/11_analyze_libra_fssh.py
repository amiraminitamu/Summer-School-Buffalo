#!/usr/bin/env python3
"""
Aggregate and analyze ten completed classical-path Libra FSSH calculations.

Inputs expected under ROOT/traj_XX/:
    libra_fssh.npz
    summary.json

Outputs:
    per_trajectory_summary.csv
    ensemble_timeseries.csv
    ensemble_results.npz
    physical_c60_ensemble.png
    active_surface_c60_ensemble.png
    mean_sh_state_populations.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("output_libra/fssh"))
    p.add_argument("--outdir", type=Path, default=Path("output_libra/analysis"))
    p.add_argument("--ntraj", type=int, default=10)
    return p.parse_args()


def load_all(root: Path, ntraj: int):
    datasets = []
    summaries = []

    for i in range(ntraj):
        folder = root / f"traj_{i:02d}"
        npz_path = folder / "libra_fssh.npz"
        json_path = folder / "summary.json"

        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        if not json_path.is_file():
            raise FileNotFoundError(json_path)

        with np.load(npz_path, allow_pickle=False) as d:
            datasets.append({key: np.array(d[key]) for key in d.files})

        summaries.append(json.loads(json_path.read_text()))

    return datasets, summaries


def confidence_interval_95(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean, SEM, and two-sided 95% Student-t half-width across trajectories.
    Axis 0 is the trajectory axis.
    """
    n = values.shape[0]
    mean = np.mean(values, axis=0)

    if n < 2:
        sem = np.zeros_like(mean)
        half = np.zeros_like(mean)
        return mean, sem, half

    std = np.std(values, axis=0, ddof=1)
    sem = std / np.sqrt(n)
    half = student_t.ppf(0.975, df=n - 1) * sem
    return mean, sem, half


def first_crossing_time(time_fs: np.ndarray, y: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero(y >= threshold)
    return float(time_fs[indices[0]]) if indices.size else float("nan")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    datasets, summaries = load_all(args.root, args.ntraj)

    reference_time = datasets[0]["time_fs"]
    nframes = len(reference_time)
    nstates = datasets[0]["sh_state_population"].shape[1]

    for i, data in enumerate(datasets):
        if not np.array_equal(data["time_fs"], reference_time):
            raise RuntimeError(f"Time grid mismatch in trajectory {i:02d}")
        if data["sh_state_population"].shape != (nframes, nstates):
            raise RuntimeError(
                f"Unexpected SH population shape in trajectory {i:02d}: "
                f"{data['sh_state_population'].shape}"
            )

    physical_c60 = np.stack([d["se_c60_population"] for d in datasets])
    surface_c60 = np.stack([d["active_surface_c60_diagonal"] for d in datasets])
    se_state = np.stack([d["se_state_population"] for d in datasets])
    sh_state = np.stack([d["sh_state_population"] for d in datasets])
    norm = np.stack([d["norm"] for d in datasets])
    interval_hops = np.stack([d["interval_hops"] for d in datasets])

    physical_mean, physical_sem, physical_ci = confidence_interval_95(physical_c60)
    surface_mean, surface_sem, surface_ci = confidence_interval_95(surface_c60)
    se_state_mean, se_state_sem, se_state_ci = confidence_interval_95(se_state)
    sh_state_mean, sh_state_sem, sh_state_ci = confidence_interval_95(sh_state)

    # Per-trajectory summary
    per_rows = []
    for i, (data, summary) in enumerate(zip(datasets, summaries)):
        p = data["se_c60_population"]
        s = data["active_surface_c60_diagonal"]
        hops = data["interval_hops"]

        row = {
            "trajectory": i,
            "initial_physical_c60": float(p[0]),
            "final_physical_c60": float(p[-1]),
            "maximum_physical_c60": float(np.max(p)),
            "time_of_maximum_physical_c60_fs": float(reference_time[np.argmax(p)]),
            "first_time_physical_c60_ge_0.10_fs": first_crossing_time(
                reference_time, p, 0.10
            ),
            "first_time_physical_c60_ge_0.25_fs": first_crossing_time(
                reference_time, p, 0.25
            ),
            "initial_surface_c60": float(s[0]),
            "final_surface_c60": float(s[-1]),
            "maximum_surface_c60": float(np.max(s)),
            "maximum_norm_error": float(np.max(np.abs(data["norm"] - 1.0))),
            "total_surface_changes": int(np.sum(hops)),
            "mean_surface_changes_per_interval": float(np.mean(hops)),
            "max_surface_changes_in_interval": int(np.max(hops)),
            "renormalized_probability_vector_count": int(
                summary.get("renormalized_probability_vector_count", -1)
            ),
            "maximum_raw_probability_vector_sum": float(
                summary.get("maximum_raw_probability_vector_sum", np.nan)
            ),
        }
        per_rows.append(row)

    per_path = args.outdir / "per_trajectory_summary.csv"
    with per_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_rows[0]))
        writer.writeheader()
        writer.writerows(per_rows)

    # Ensemble time series
    time_fields = [
        "time_fs",
        "physical_c60_mean",
        "physical_c60_sem",
        "physical_c60_ci95_low",
        "physical_c60_ci95_high",
        "surface_c60_mean",
        "surface_c60_sem",
        "surface_c60_ci95_low",
        "surface_c60_ci95_high",
    ]
    time_fields += [f"sh_state_{i}_mean" for i in range(nstates)]
    time_fields += [f"sh_state_{i}_sem" for i in range(nstates)]

    ts_path = args.outdir / "ensemble_timeseries.csv"
    with ts_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=time_fields)
        writer.writeheader()

        for k, time in enumerate(reference_time):
            row = {
                "time_fs": float(time),
                "physical_c60_mean": float(physical_mean[k]),
                "physical_c60_sem": float(physical_sem[k]),
                "physical_c60_ci95_low": float(physical_mean[k] - physical_ci[k]),
                "physical_c60_ci95_high": float(physical_mean[k] + physical_ci[k]),
                "surface_c60_mean": float(surface_mean[k]),
                "surface_c60_sem": float(surface_sem[k]),
                "surface_c60_ci95_low": float(surface_mean[k] - surface_ci[k]),
                "surface_c60_ci95_high": float(surface_mean[k] + surface_ci[k]),
            }
            row.update(
                {f"sh_state_{i}_mean": float(sh_state_mean[k, i]) for i in range(nstates)}
            )
            row.update(
                {f"sh_state_{i}_sem": float(sh_state_sem[k, i]) for i in range(nstates)}
            )
            writer.writerow(row)

    np.savez_compressed(
        args.outdir / "ensemble_results.npz",
        time_fs=reference_time,
        physical_c60_all=physical_c60,
        physical_c60_mean=physical_mean,
        physical_c60_sem=physical_sem,
        physical_c60_ci95_halfwidth=physical_ci,
        surface_c60_all=surface_c60,
        surface_c60_mean=surface_mean,
        surface_c60_sem=surface_sem,
        surface_c60_ci95_halfwidth=surface_ci,
        se_state_population_all=se_state,
        se_state_population_mean=se_state_mean,
        se_state_population_sem=se_state_sem,
        se_state_population_ci95_halfwidth=se_state_ci,
        sh_state_population_all=sh_state,
        sh_state_population_mean=sh_state_mean,
        sh_state_population_sem=sh_state_sem,
        sh_state_population_ci95_halfwidth=sh_state_ci,
        norm_all=norm,
        interval_hops_all=interval_hops,
    )

    # Plot 1: physical C60 population.
    plt.figure(figsize=(7.2, 4.8))
    for curve in physical_c60:
        plt.plot(reference_time, curve, linewidth=0.8, alpha=0.35)
    plt.plot(reference_time, physical_mean, linewidth=2.0, label="Ensemble mean")
    plt.fill_between(
        reference_time,
        physical_mean - physical_ci,
        physical_mean + physical_ci,
        alpha=0.2,
        label="95% t interval",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel(r"$\langle c^\dagger P_{\mathrm{C60}} c\rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir / "physical_c60_ensemble.png", dpi=300)
    plt.close()

    # Plot 2: active-surface diagonal fragment diagnostic.
    plt.figure(figsize=(7.2, 4.8))
    for curve in surface_c60:
        plt.plot(reference_time, curve, linewidth=0.8, alpha=0.35)
    plt.plot(reference_time, surface_mean, linewidth=2.0, label="Ensemble mean")
    plt.fill_between(
        reference_time,
        surface_mean - surface_ci,
        surface_mean + surface_ci,
        alpha=0.2,
        label="95% t interval",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel("Active-surface C60 diagonal diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir / "active_surface_c60_ensemble.png", dpi=300)
    plt.close()

    # Plot 3: mean SH state populations.
    plt.figure(figsize=(7.2, 4.8))
    for state in range(nstates):
        plt.plot(
            reference_time,
            sh_state_mean[:, state],
            linewidth=1.3,
            label=f"State {state}",
        )
    plt.xlabel("Time (fs)")
    plt.ylabel("Mean FSSH active-state population")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.outdir / "mean_sh_state_populations.png", dpi=300)
    plt.close()

    # Compact terminal report.
    print(
        f"{'traj':>5} {'P0 physical':>13} {'Pfinal physical':>16} "
        f"{'Pmax physical':>14} {'Pfinal surface':>15} "
        f"{'norm err':>11} {'hops':>9}"
    )
    print("-" * 96)
    for row in per_rows:
        print(
            f"{row['trajectory']:5d} "
            f"{row['initial_physical_c60']:13.6f} "
            f"{row['final_physical_c60']:16.6f} "
            f"{row['maximum_physical_c60']:14.6f} "
            f"{row['final_surface_c60']:15.6f} "
            f"{row['maximum_norm_error']:11.2e} "
            f"{row['total_surface_changes']:9d}"
        )

    print("\nEnsemble endpoint:")
    print(
        f"  Physical C60: {physical_mean[-1]:.8f} "
        f"+/- {physical_sem[-1]:.8f} SEM "
        f"(95% CI half-width {physical_ci[-1]:.8f})"
    )
    print(
        f"  Surface diagnostic: {surface_mean[-1]:.8f} "
        f"+/- {surface_sem[-1]:.8f} SEM "
        f"(95% CI half-width {surface_ci[-1]:.8f})"
    )
    print(
        f"  Max norm error over all paths: "
        f"{np.max(np.abs(norm - 1.0)):.3e}"
    )
    print(
        f"  Total surface changes over all paths: "
        f"{int(np.sum(interval_hops))}"
    )
    print(f"\nWrote analysis to: {args.outdir}")


if __name__ == "__main__":
    main()
