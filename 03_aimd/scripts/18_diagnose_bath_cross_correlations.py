#!/usr/bin/env python3
"""
Diagnose whether PCA Hamiltonian-fluctuation coordinates can be treated as
independent TENSO baths.

Ordinary PCA diagonalizes only the zero-lag covariance C_ab(0). Independent
bath coordinates additionally require the finite-lag cross-correlations

    C_ab(tau) = <x_a(t) x_b(t + tau)>

to remain small for a != b.

This script computes normalized cross-correlations, identifies strongly
coupled mode groups, and summarizes each mode's correlation time and dominant
coarse spectral feature. It does not yet fit quantum bath correlation
functions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "output_tenso_bath_analysis/"
            "hamiltonian_fluctuation_pca.npz"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_tenso_bath_crosscorr"),
    )
    parser.add_argument("--nmodes", type=int, default=18)
    parser.add_argument(
        "--max-lag-fs",
        type=float,
        default=50.0,
        help="Maximum positive lag used for cross-correlation diagnostics.",
    )
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.25,
        help=(
            "Connect two PCA modes when their maximum absolute normalized "
            "finite-lag cross-correlation exceeds this value."
        ),
    )
    parser.add_argument(
        "--low-frequency-cutoff-cm",
        type=float,
        default=500.0,
    )
    return parser.parse_args()


def normalized_cross_correlations(
    coordinates: np.ndarray,
    max_lag_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    coordinates
        Shape (ntraj, nmodes, ntime). Each trajectory is centered separately.
    max_lag_steps
        Largest nonnegative lag.

    Returns
    -------
    corr
        Shape (max_lag_steps + 1, nmodes, nmodes).
    covariance_zero
        Zero-lag covariance matrix before normalization.
    """
    ntraj, nmodes, ntime = coordinates.shape
    centered = coordinates - np.mean(coordinates, axis=2, keepdims=True)

    covariance_zero = np.zeros((nmodes, nmodes), dtype=float)
    for trajectory in range(ntraj):
        x = centered[trajectory]
        covariance_zero += x @ x.T / ntime
    covariance_zero /= ntraj

    variances = np.diag(covariance_zero)
    denominator = np.sqrt(np.outer(variances, variances))

    corr = np.zeros(
        (max_lag_steps + 1, nmodes, nmodes),
        dtype=float,
    )

    for lag in range(max_lag_steps + 1):
        count = ntime - lag
        covariance = np.zeros((nmodes, nmodes), dtype=float)

        for trajectory in range(ntraj):
            left = centered[trajectory, :, :count]
            right = centered[trajectory, :, lag:]
            covariance += left @ right.T / count

        covariance /= ntraj
        corr[lag] = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0.0,
        )

    return corr, covariance_zero


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    n = adjacency.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component = []

        while stack:
            node = stack.pop()
            component.append(node)

            neighbors = np.flatnonzero(adjacency[node])
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        components.append(sorted(component))

    return components


def first_threshold_time(
    values: np.ndarray,
    lags_fs: np.ndarray,
    threshold: float,
) -> float:
    indices = np.flatnonzero(values <= threshold)
    return float(lags_fs[indices[0]]) if indices.size else float("nan")


def first_zero_crossing(
    values: np.ndarray,
    lags_fs: np.ndarray,
) -> float:
    indices = np.flatnonzero(values <= 0.0)
    return float(lags_fs[indices[0]]) if indices.size else float("nan")


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input, allow_pickle=False) as data:
        time_fs = np.asarray(data["time_fs"], dtype=float)
        coordinates_all = np.asarray(
            data["mode_coordinates_hartree"],
            dtype=float,
        )
        frequencies_cm = np.asarray(
            data["frequency_cm_inverse"],
            dtype=float,
        )
        periodograms_all = np.asarray(
            data["mode_periodogram_hartree2_fs"],
            dtype=float,
        )
        explained_all = np.asarray(
            data["pca_explained_fraction"],
            dtype=float,
        )
        cumulative_all = np.asarray(
            data["pca_cumulative_fraction"],
            dtype=float,
        )

    available_modes = coordinates_all.shape[1]
    if args.nmodes > available_modes:
        raise RuntimeError(
            f"Requested {args.nmodes} modes, but the input contains only "
            f"{available_modes}. Rerun 17_analyze_tenso_bath_modes.py with "
            f"--nmodes {args.nmodes} first."
        )

    nmodes = args.nmodes
    coordinates = coordinates_all[:, :nmodes, :]
    periodograms = periodograms_all[:nmodes]
    explained = explained_all[:nmodes]
    cumulative = cumulative_all[:nmodes]

    dt_fs = float(time_fs[1] - time_fs[0])
    max_lag_steps = min(
        int(round(args.max_lag_fs / dt_fs)),
        len(time_fs) - 1,
    )
    lags_fs = np.arange(max_lag_steps + 1) * dt_fs

    corr, covariance_zero = normalized_cross_correlations(
        coordinates,
        max_lag_steps,
    )

    zero_lag_offdiag = corr[0].copy()
    np.fill_diagonal(zero_lag_offdiag, 0.0)

    max_pair_corr = np.zeros((nmodes, nmodes), dtype=float)
    max_pair_lag_fs = np.zeros((nmodes, nmodes), dtype=float)

    for a in range(nmodes):
        for b in range(a + 1, nmodes):
            forward = np.abs(corr[:, a, b])
            reverse = np.abs(corr[:, b, a])

            forward_index = int(np.argmax(forward))
            reverse_index = int(np.argmax(reverse))

            if forward[forward_index] >= reverse[reverse_index]:
                value = float(forward[forward_index])
                lag = float(lags_fs[forward_index])
            else:
                value = float(reverse[reverse_index])
                lag = -float(lags_fs[reverse_index])

            max_pair_corr[a, b] = value
            max_pair_corr[b, a] = value
            max_pair_lag_fs[a, b] = lag
            max_pair_lag_fs[b, a] = -lag

    adjacency = max_pair_corr >= args.group_threshold
    np.fill_diagonal(adjacency, True)
    groups = connected_components(adjacency)

    pair_rows = []
    for a in range(nmodes):
        for b in range(a + 1, nmodes):
            pair_rows.append(
                {
                    "mode_a": a + 1,
                    "mode_b": b + 1,
                    "maximum_absolute_normalized_crosscorrelation": (
                        max_pair_corr[a, b]
                    ),
                    "signed_lag_of_maximum_fs": (
                        max_pair_lag_fs[a, b]
                    ),
                    "zero_lag_normalized_crosscorrelation": (
                        corr[0, a, b]
                    ),
                    "connected_at_threshold": bool(
                        adjacency[a, b]
                    ),
                }
            )

    pair_rows.sort(
        key=lambda row: row[
            "maximum_absolute_normalized_crosscorrelation"
        ],
        reverse=True,
    )

    with (args.outdir / "pairwise_crosscorrelations.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(pair_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    mode_rows = []
    for mode in range(nmodes):
        acf = corr[:, mode, mode]
        e_fold_time = first_threshold_time(
            acf,
            lags_fs,
            np.exp(-1.0),
        )
        zero_time = first_zero_crossing(acf, lags_fs)

        psd = periodograms[mode]
        nonzero = np.arange(1, len(frequencies_cm))
        dominant_index = int(nonzero[np.argmax(psd[nonzero])])

        total_power = trapezoid(psd, frequencies_cm)
        low_mask = frequencies_cm <= args.low_frequency_cutoff_cm
        low_power = trapezoid(
            psd[low_mask],
            frequencies_cm[low_mask],
        )
        low_fraction = (
            low_power / total_power if total_power > 0.0 else np.nan
        )

        strongest_partner = int(np.argmax(max_pair_corr[mode]))

        mode_rows.append(
            {
                "mode": mode + 1,
                "explained_fraction": explained[mode],
                "cumulative_fraction": cumulative[mode],
                "acf_e_fold_time_fs": e_fold_time,
                "acf_first_zero_crossing_fs": zero_time,
                "dominant_coarse_frequency_cm_inverse": (
                    frequencies_cm[dominant_index]
                ),
                "low_frequency_power_fraction": low_fraction,
                "strongest_crosscorrelated_mode": (
                    strongest_partner + 1
                ),
                "maximum_crosscorrelation_with_partner": (
                    max_pair_corr[mode, strongest_partner]
                ),
                "lag_of_partner_maximum_fs": (
                    max_pair_lag_fs[mode, strongest_partner]
                ),
            }
        )

    with (args.outdir / "mode_memory_diagnostics.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(mode_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(mode_rows)

    np.savez_compressed(
        args.outdir / "bath_crosscorrelations.npz",
        lags_fs=lags_fs,
        normalized_crosscorrelations=corr,
        zero_lag_covariance_hartree2=covariance_zero,
        maximum_pair_crosscorrelation=max_pair_corr,
        lag_of_maximum_pair_crosscorrelation_fs=(
            max_pair_lag_fs
        ),
        adjacency=adjacency,
    )

    plt.figure(figsize=(7.2, 5.8))
    image = plt.imshow(
        max_pair_corr,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(
        image,
        label="Maximum absolute normalized cross-correlation",
    )
    plt.xlabel("PCA mode")
    plt.ylabel("PCA mode")
    plt.xticks(np.arange(nmodes), np.arange(1, nmodes + 1))
    plt.yticks(np.arange(nmodes), np.arange(1, nmodes + 1))
    plt.tight_layout()
    plt.savefig(
        args.outdir / "maximum_crosscorrelation_heatmap.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for mode in range(min(nmodes, 12)):
        plt.plot(
            lags_fs,
            corr[:, mode, mode],
            linewidth=1.3,
            label=f"Mode {mode + 1}",
        )
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Lag time (fs)")
    plt.ylabel("Normalized autocorrelation")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        args.outdir / "normalized_mode_autocorrelations.png",
        dpi=300,
    )
    plt.close()

    print(
        f"Analyzed {nmodes} PCA modes over lags "
        f"0--{lags_fs[-1]:.1f} fs."
    )
    print(
        f"Maximum zero-lag off-diagonal correlation: "
        f"{np.max(np.abs(zero_lag_offdiag)):.3e}"
    )

    print("\nStrongest finite-lag mode pairs:")
    print(
        f"{'mode a':>7} {'mode b':>7} "
        f"{'max |corr|':>12} {'lag/fs':>10}"
    )
    print("-" * 42)
    for row in pair_rows[:15]:
        print(
            f"{row['mode_a']:7d} "
            f"{row['mode_b']:7d} "
            f"{row['maximum_absolute_normalized_crosscorrelation']:12.5f} "
            f"{row['signed_lag_of_maximum_fs']:10.2f}"
        )

    print(
        f"\nMode groups at threshold "
        f"{args.group_threshold:.2f}:"
    )
    for index, group in enumerate(groups, start=1):
        labels = ", ".join(str(mode + 1) for mode in group)
        print(f"  Group {index}: {labels}")

    print("\nMode memory diagnostics:")
    print(
        f"{'mode':>5} {'frac':>9} {'tau_e/fs':>10} "
        f"{'zero/fs':>10} {'peak/cm-1':>11} "
        f"{'low-f frac':>11} {'partner':>8} {'xcorr':>8}"
    )
    print("-" * 90)
    for row in mode_rows:
        print(
            f"{row['mode']:5d} "
            f"{row['explained_fraction']:9.5f} "
            f"{row['acf_e_fold_time_fs']:10.2f} "
            f"{row['acf_first_zero_crossing_fs']:10.2f} "
            f"{row['dominant_coarse_frequency_cm_inverse']:11.1f} "
            f"{row['low_frequency_power_fraction']:11.4f} "
            f"{row['strongest_crosscorrelated_mode']:8d} "
            f"{row['maximum_crosscorrelation_with_partner']:8.4f}"
        )

    print(f"\nWrote diagnostics to {args.outdir}")


if __name__ == "__main__":
    main()
