#!/usr/bin/env python3
"""
Compare plain and Boltzmann-rescaled CPA-FSSH C60 populations with the
digitized experimental SHG curve.

For each trajectory, the paper-style hybrid estimator is

    P_hybrid = P_active-surface-diagonal
               + P_coherent
               - P_coherent-diagonal.

The same tracked C60 projector is used for both FSSH variants.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plain-root", type=Path, default=Path("output_libra/fssh"))
    p.add_argument(
        "--boltzmann-root",
        type=Path,
        default=Path("output_libra/fssh_boltzmann"),
    )
    p.add_argument("--tracked-root", type=Path, default=Path("output_tracked"))
    p.add_argument(
        "--experiment",
        type=Path,
        default=Path("experiment_shg_digitized.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_libra/final_figures_boltzmann"),
    )
    p.add_argument("--ntraj", type=int, default=10)
    p.add_argument("--max-time-fs", type=float, default=99.5)
    return p.parse_args()


def load_experiment(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    populations: list[float] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time_fs", "c60_charge_population"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain time_fs,c60_charge_population"
            )

        for row in reader:
            times.append(float(row["time_fs"]))
            populations.append(float(row["c60_charge_population"]))

    return np.asarray(times), np.asarray(populations)


def mean_sem_ci95(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = curves.shape[0]
    mean = np.mean(curves, axis=0)
    sem = np.std(curves, axis=0, ddof=1) / np.sqrt(n)
    ci95 = student_t.ppf(0.975, n - 1) * sem
    return mean, sem, ci95


def first_crossing(time_fs: np.ndarray, curve: np.ndarray, level: float) -> float:
    indices = np.flatnonzero(curve >= level)
    return float(time_fs[indices[0]]) if indices.size else float("nan")


def load_hybrid_curve(
    libra_root: Path,
    tracked_root: Path,
    trajectory: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    libra_path = (
        libra_root / f"traj_{trajectory:02d}" / "libra_fssh.npz"
    )
    tracked_path = (
        tracked_root / f"traj_{trajectory:02d}" / "tracked_data.npz"
    )

    if not libra_path.is_file():
        raise FileNotFoundError(libra_path)
    if not tracked_path.is_file():
        raise FileNotFoundError(tracked_path)

    with np.load(libra_path, allow_pickle=False) as data:
        time_fs = np.asarray(data["time_fs"], dtype=float)
        coefficients = np.asarray(data["coefficients"], dtype=complex)
        se_state = np.asarray(data["se_state_population"], dtype=float)
        coherent = np.asarray(data["se_c60_population"], dtype=float)
        active_diagonal = np.asarray(
            data["active_surface_c60_diagonal"],
            dtype=float,
        )

    with np.load(tracked_path, allow_pickle=False) as data:
        tracked_time = np.asarray(data["time_fs"], dtype=float)
        projector = np.asarray(data["projector_c60"], dtype=complex)

    nframes = len(time_fs)
    projector = projector[:nframes]

    if not np.array_equal(time_fs, tracked_time[:nframes]):
        raise RuntimeError(
            f"Time-grid mismatch for trajectory {trajectory:02d}"
        )

    projector_diagonal = np.real(
        np.diagonal(projector, axis1=1, axis2=2)
    )
    coherent_diagonal = np.sum(se_state * projector_diagonal, axis=1)
    coherent_offdiagonal = coherent - coherent_diagonal
    hybrid = active_diagonal + coherent_offdiagonal

    # Independent direct reconstruction of Eqs. 12-14.
    sh_state = None
    with np.load(libra_path, allow_pickle=False) as data:
        sh_state = np.asarray(data["sh_state_population"], dtype=float)

    density = np.einsum("ti,tj->tij", coefficients.conj(), coefficients)
    states = np.arange(coefficients.shape[1])
    density[:, states, states] = sh_state
    hybrid_direct = np.real(np.einsum("tij,tji->t", density, projector))

    error = np.max(np.abs(hybrid - hybrid_direct))
    if error > 1.0e-10:
        raise RuntimeError(
            f"Hybrid reconstruction mismatch for trajectory "
            f"{trajectory:02d}: {error:.3e}"
        )

    return time_fs, hybrid, coherent, active_diagonal


def metrics(
    name: str,
    time_fs: np.ndarray,
    mean: np.ndarray,
    experiment: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | str]:
    difference = mean[mask] - experiment
    return {
        "method": name,
        "initial": float(mean[0]),
        "final": float(mean[np.flatnonzero(mask)[-1]]),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
        "first_0.25_fs": first_crossing(time_fs, mean, 0.25),
        "first_0.50_fs": first_crossing(time_fs, mean, 0.50),
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    plain = []
    boltzmann = []
    coherent_plain = []
    coherent_boltzmann = []

    time_fs = None
    for trajectory in range(args.ntraj):
        t_plain, p_plain, c_plain, _ = load_hybrid_curve(
            args.plain_root,
            args.tracked_root,
            trajectory,
        )
        t_boltz, p_boltz, c_boltz, _ = load_hybrid_curve(
            args.boltzmann_root,
            args.tracked_root,
            trajectory,
        )

        if not np.array_equal(t_plain, t_boltz):
            raise RuntimeError(
                f"Plain/Boltzmann time mismatch for trajectory {trajectory:02d}"
            )

        if time_fs is None:
            time_fs = t_plain
        elif not np.array_equal(time_fs, t_plain):
            raise RuntimeError(
                f"Trajectory time mismatch for trajectory {trajectory:02d}"
            )

        plain.append(p_plain)
        boltzmann.append(p_boltz)
        coherent_plain.append(c_plain)
        coherent_boltzmann.append(c_boltz)

    assert time_fs is not None
    plain_all = np.stack(plain)
    boltz_all = np.stack(boltzmann)
    coherent_plain_all = np.stack(coherent_plain)
    coherent_boltz_all = np.stack(coherent_boltzmann)

    # Boltzmann scaling changes hopping populations, not the TDSE amplitudes.
    coherent_difference = float(
        np.max(np.abs(coherent_plain_all - coherent_boltz_all))
    )

    plain_mean, plain_sem, plain_ci = mean_sem_ci95(plain_all)
    boltz_mean, boltz_sem, boltz_ci = mean_sem_ci95(boltz_all)

    exp_time, exp_population = load_experiment(args.experiment)
    mask = time_fs <= args.max_time_fs
    experiment_grid = np.interp(time_fs[mask], exp_time, exp_population)

    plain_metrics = metrics(
        "Plain CPA-FSSH",
        time_fs,
        plain_mean,
        experiment_grid,
        mask,
    )
    boltz_metrics = metrics(
        "Boltzmann CPA-FSSH",
        time_fs,
        boltz_mean,
        experiment_grid,
        mask,
    )

    # Main comparison figure.
    plt.figure(figsize=(7.4, 4.9))
    exp_mask = exp_time <= args.max_time_fs
    plt.plot(
        exp_time[exp_mask],
        exp_population[exp_mask],
        linewidth=2.5,
        label="Experiment (digitized SHG)",
    )
    plt.plot(
        time_fs,
        plain_mean,
        linewidth=1.9,
        linestyle="--",
        label="Plain CPA-FSSH",
    )
    plt.plot(
        time_fs,
        boltz_mean,
        linewidth=2.4,
        label="Boltzmann-rescaled CPA-FSSH",
    )
    plt.fill_between(
        time_fs,
        boltz_mean - boltz_ci,
        boltz_mean + boltz_ci,
        alpha=0.20,
        label="Boltzmann 95% t interval",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel("C60 charge population")
    plt.xlim(0.0, args.max_time_fs)
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.outdir / "fig_plain_boltzmann_experiment.png", dpi=400)
    plt.savefig(args.outdir / "fig_plain_boltzmann_experiment.pdf")
    plt.close()

    # Difference figure.
    difference_all = boltz_all - plain_all
    difference_mean, _, difference_ci = mean_sem_ci95(difference_all)

    plt.figure(figsize=(7.4, 4.5))
    plt.axhline(0.0, linewidth=1.0)
    plt.plot(
        time_fs,
        difference_mean,
        linewidth=2.2,
        label=r"$P_{\mathrm{Boltzmann}}-P_{\mathrm{plain}}$",
    )
    plt.fill_between(
        time_fs,
        difference_mean - difference_ci,
        difference_mean + difference_ci,
        alpha=0.20,
        label="95% t interval",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel("Change in C60 population")
    plt.xlim(0.0, args.max_time_fs)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.outdir / "fig_boltzmann_effect.png", dpi=400)
    plt.savefig(args.outdir / "fig_boltzmann_effect.pdf")
    plt.close()

    # Save combined table.
    csv_path = args.outdir / "plain_boltzmann_experiment.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "time_fs",
            "experiment_digitized",
            "plain_mean",
            "plain_sem",
            "plain_ci95_low",
            "plain_ci95_high",
            "boltzmann_mean",
            "boltzmann_sem",
            "boltzmann_ci95_low",
            "boltzmann_ci95_high",
            "boltzmann_minus_plain",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        experiment_full = np.interp(time_fs, exp_time, exp_population)
        for i, t in enumerate(time_fs):
            writer.writerow(
                {
                    "time_fs": float(t),
                    "experiment_digitized": float(experiment_full[i]),
                    "plain_mean": float(plain_mean[i]),
                    "plain_sem": float(plain_sem[i]),
                    "plain_ci95_low": float(plain_mean[i] - plain_ci[i]),
                    "plain_ci95_high": float(plain_mean[i] + plain_ci[i]),
                    "boltzmann_mean": float(boltz_mean[i]),
                    "boltzmann_sem": float(boltz_sem[i]),
                    "boltzmann_ci95_low": float(
                        boltz_mean[i] - boltz_ci[i]
                    ),
                    "boltzmann_ci95_high": float(
                        boltz_mean[i] + boltz_ci[i]
                    ),
                    "boltzmann_minus_plain": float(
                        boltz_mean[i] - plain_mean[i]
                    ),
                }
            )

    text = f"""Plain and Boltzmann CPA-FSSH versus digitized experiment
===========================================================

Comparison interval: 0.0-{time_fs[np.flatnonzero(mask)[-1]]:.1f} fs
Nuclear trajectories: {args.ntraj}
Maximum coherent-curve difference between reruns:
  {coherent_difference:.3e}

Plain CPA-FSSH
  initial population: {plain_metrics['initial']:.8f}
  final population:   {plain_metrics['final']:.8f}
  RMSE:               {plain_metrics['rmse']:.8f}
  MAE:                {plain_metrics['mae']:.8f}
  first P >= 0.25:    {plain_metrics['first_0.25_fs']:.2f} fs
  first P >= 0.50:    {plain_metrics['first_0.50_fs']:.2f} fs

Boltzmann-rescaled CPA-FSSH
  initial population: {boltz_metrics['initial']:.8f}
  final population:   {boltz_metrics['final']:.8f}
  RMSE:               {boltz_metrics['rmse']:.8f}
  MAE:                {boltz_metrics['mae']:.8f}
  first P >= 0.25:    {boltz_metrics['first_0.25_fs']:.2f} fs
  first P >= 0.50:    {boltz_metrics['first_0.50_fs']:.2f} fs

Endpoint Boltzmann effect:
  {boltz_metrics['final'] - plain_metrics['final']:+.8f}

The experimental curve is an approximate digitization of Figure 3A,
not the original experimental data.
"""
    (args.outdir / "plain_boltzmann_metrics.txt").write_text(text)
    print(text)
    print(f"Wrote results to {args.outdir}")


if __name__ == "__main__":
    main()
