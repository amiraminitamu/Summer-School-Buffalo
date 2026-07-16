#!/usr/bin/env python3
"""
Construct a frequency-resolved shared-bath decomposition from correlated
Hamiltonian-fluctuation PCA coordinates.

Why this is needed
------------------
PCA diagonalizes only the zero-lag covariance. The finite-lag analysis shows
that the PCA coordinates remain cross-correlated, so they should not be fed
to an open-system solver as independent scalar baths.

For each positive Fourier frequency, this script estimates the cross-spectral
matrix

    S_ab(omega) = <X_a(omega) X_b(omega)^*>

across the independent molecular trajectories.  A short frequency smoothing
window may be used because each 100 fs trajectory has coarse 333.6 cm^-1
resolution. The real symmetric part of S is diagonalized to obtain
frequency-local shared bath channels,

    A_k(omega) = sum_a v_ak(omega) Q_a,

where Q_a are the Hamiltonian-fluctuation PCA operators. The output reports
how many shared channels are needed to capture 90, 95, and 99 percent of the
power at each coarse frequency.

This is a diagnostic and model-construction step. It does not yet impose a
Drude-Lorentz or Brownian-oscillator line shape.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FS_INV_TO_WAVENUMBER = 33356.40951981521
HARTREE_TO_EV = 27.211386245988


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
        default=Path("output_tenso_shared_baths"),
    )
    parser.add_argument("--nmodes", type=int, default=18)
    parser.add_argument(
        "--max-frequency-cm",
        type=float,
        default=4000.0,
    )
    parser.add_argument(
        "--smooth-bins",
        type=int,
        default=1,
        help=(
            "Half-width of the moving frequency-bin average. "
            "1 means averaging over the target bin and one neighbor "
            "on either side."
        ),
    )
    parser.add_argument(
        "--print-power-fraction",
        type=float,
        default=0.01,
        help=(
            "Print frequency bins carrying at least this fraction of the "
            "total power below --max-frequency-cm."
        ),
    )
    return parser.parse_args()


def hermitize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def real_symmetric_psd(a: np.ndarray) -> np.ndarray:
    """
    Return the real symmetric part with tiny negative eigenvalues removed.
    """
    real_part = 0.5 * (a.real + a.real.T)
    values, vectors = np.linalg.eigh(real_part)
    values = np.maximum(values, 0.0)
    return (vectors * values[None, :]) @ vectors.T


def ranks_for_thresholds(
    eigenvalues_descending: np.ndarray,
) -> tuple[int, int, int]:
    total = float(np.sum(eigenvalues_descending))
    if total <= 0.0:
        return 0, 0, 0

    cumulative = np.cumsum(eigenvalues_descending) / total
    return tuple(
        int(np.searchsorted(cumulative, threshold) + 1)
        for threshold in (0.90, 0.95, 0.99)
    )


def dominant_operator_elements(
    operator: np.ndarray,
    count: int = 6,
) -> str:
    entries = []
    n = operator.shape[0]

    for i in range(n):
        entries.append((abs(operator[i, i]), f"({i},{i})"))
        for j in range(i + 1, n):
            entries.append(
                (
                    np.sqrt(2.0) * abs(operator[i, j]),
                    f"({i},{j})",
                )
            )

    entries.sort(reverse=True)
    return "; ".join(
        f"{label}:{value:.3f}"
        for value, label in entries[:count]
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input, allow_pickle=False) as data:
        time_fs = np.asarray(data["time_fs"], dtype=float)
        coordinates_all = np.asarray(
            data["mode_coordinates_hartree"],
            dtype=float,
        )
        mode_operators_all = np.asarray(
            data["mode_operators"],
            dtype=complex,
        )
        explained_all = np.asarray(
            data["pca_explained_fraction"],
            dtype=float,
        )

    available_modes = coordinates_all.shape[1]
    if args.nmodes > available_modes:
        raise RuntimeError(
            f"Requested {args.nmodes} modes, but the input contains only "
            f"{available_modes}. Rerun the PCA script with a larger "
            f"--nmodes value."
        )

    coordinates = coordinates_all[:, : args.nmodes, :]
    mode_operators = mode_operators_all[: args.nmodes]
    explained = explained_all[: args.nmodes]

    ntraj, nmodes, ntime = coordinates.shape
    dt_fs = float(time_fs[1] - time_fs[0])

    centered = coordinates - np.mean(
        coordinates,
        axis=2,
        keepdims=True,
    )
    window = np.hanning(ntime)
    window_norm = float(np.sum(window**2))

    frequency_fs_inv = np.fft.rfftfreq(
        ntime,
        d=dt_fs,
    )
    frequency_cm = (
        frequency_fs_inv * FS_INV_TO_WAVENUMBER
    )

    transforms = np.fft.rfft(
        centered * window[None, None, :],
        axis=2,
    )

    nfrequency = transforms.shape[2]
    spectral_matrices = np.zeros(
        (nfrequency, nmodes, nmodes),
        dtype=complex,
    )

    for frequency in range(nfrequency):
        for trajectory in range(ntraj):
            z = transforms[trajectory, :, frequency]
            spectral_matrices[frequency] += np.outer(
                z,
                z.conj(),
            )

        spectral_matrices[frequency] *= (
            dt_fs / (ntraj * window_norm)
        )

        if (
            frequency > 0
            and not (
                ntime % 2 == 0
                and frequency == nfrequency - 1
            )
        ):
            spectral_matrices[frequency] *= 2.0

    smooth = max(args.smooth_bins, 0)
    smoothed = np.zeros_like(spectral_matrices)

    for frequency in range(nfrequency):
        lower = max(0, frequency - smooth)
        upper = min(
            nfrequency,
            frequency + smooth + 1,
        )
        smoothed[frequency] = np.mean(
            spectral_matrices[lower:upper],
            axis=0,
        )

    selected = np.flatnonzero(
        (frequency_cm > 0.0)
        & (frequency_cm <= args.max_frequency_cm)
    )
    if selected.size == 0:
        raise RuntimeError(
            "No positive frequency bins satisfy the requested range."
        )

    total_power_per_bin = np.array(
        [
            float(np.trace(smoothed[index]).real)
            for index in selected
        ]
    )
    total_power = float(np.sum(total_power_per_bin))
    power_fraction = np.divide(
        total_power_per_bin,
        total_power,
        out=np.zeros_like(total_power_per_bin),
        where=total_power > 0.0,
    )

    max_channels = nmodes
    channel_eigenvalues = np.zeros(
        (selected.size, max_channels),
        dtype=float,
    )
    channel_vectors = np.zeros(
        (selected.size, max_channels, nmodes),
        dtype=float,
    )
    channel_operators = np.zeros(
        (
            selected.size,
            max_channels,
            mode_operators.shape[1],
            mode_operators.shape[2],
        ),
        dtype=complex,
    )
    imaginary_fraction = np.zeros(
        selected.size,
        dtype=float,
    )
    ranks = np.zeros((selected.size, 3), dtype=int)

    rows = []

    for local_index, frequency_index in enumerate(selected):
        matrix = hermitize(smoothed[frequency_index])
        norm_total = float(np.linalg.norm(matrix))
        imaginary_fraction[local_index] = (
            float(np.linalg.norm(matrix.imag)) / norm_total
            if norm_total > 0.0
            else 0.0
        )

        matrix_real_psd = real_symmetric_psd(matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(
            matrix_real_psd
        )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        channel_eigenvalues[local_index] = eigenvalues
        channel_vectors[local_index] = eigenvectors.T
        channel_operators[local_index] = np.einsum(
            "ka,aij->kij",
            eigenvectors.T,
            mode_operators,
        )

        rank90, rank95, rank99 = ranks_for_thresholds(
            eigenvalues
        )
        ranks[local_index] = (rank90, rank95, rank99)

        eigen_total = float(np.sum(eigenvalues))
        channel_fractions = (
            eigenvalues / eigen_total
            if eigen_total > 0.0
            else np.zeros_like(eigenvalues)
        )
        cumulative = np.cumsum(channel_fractions)

        rows.append(
            {
                "frequency_cm_inverse": (
                    frequency_cm[frequency_index]
                ),
                "power_fraction_below_cutoff": (
                    power_fraction[local_index]
                ),
                "imaginary_cross_spectrum_fraction": (
                    imaginary_fraction[local_index]
                ),
                "rank90": rank90,
                "rank95": rank95,
                "rank99": rank99,
                "channel1_fraction": channel_fractions[0],
                "channel2_cumulative_fraction": (
                    cumulative[1]
                    if len(cumulative) > 1
                    else cumulative[0]
                ),
                "channel3_cumulative_fraction": (
                    cumulative[2]
                    if len(cumulative) > 2
                    else cumulative[-1]
                ),
                "channel1_rms_scale_eV_sqrt_fs": (
                    np.sqrt(eigenvalues[0])
                    * HARTREE_TO_EV
                ),
                "channel1_dominant_operator_elements": (
                    dominant_operator_elements(
                        channel_operators[local_index, 0]
                    )
                ),
            }
        )

    with (args.outdir / "frequency_channel_summary.csv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.outdir / "shared_bath_channels.npz",
        frequency_cm_inverse=frequency_cm[selected],
        frequency_indices=selected,
        total_power_per_bin_hartree2_fs=(
            total_power_per_bin
        ),
        power_fraction_below_cutoff=power_fraction,
        smoothed_cross_spectral_matrices=smoothed[selected],
        imaginary_cross_spectrum_fraction=(
            imaginary_fraction
        ),
        channel_eigenvalues_hartree2_fs=(
            channel_eigenvalues
        ),
        channel_vectors_in_pca_basis=channel_vectors,
        channel_operators=channel_operators,
        ranks_90_95_99=ranks,
        original_pca_explained_fraction=explained,
        smoothing_half_width_bins=smooth,
    )

    weighted_mean_rank95 = float(
        np.sum(power_fraction * ranks[:, 1])
    )
    weighted_imaginary_fraction = float(
        np.sum(
            power_fraction
            * imaginary_fraction
        )
    )

    summary = {
        "ntrajectories": ntraj,
        "nframes": ntime,
        "time_step_fs": dt_fs,
        "frequency_spacing_cm_inverse": float(
            frequency_cm[1] - frequency_cm[0]
        ),
        "maximum_frequency_analyzed_cm_inverse": (
            args.max_frequency_cm
        ),
        "pca_modes_included": nmodes,
        "pca_variance_fraction_included": float(
            np.sum(explained)
        ),
        "frequency_smoothing_half_width_bins": smooth,
        "power_weighted_mean_rank95": weighted_mean_rank95,
        "power_weighted_imaginary_cross_spectrum_fraction": (
            weighted_imaginary_fraction
        ),
        "important_note": (
            "The shared operators are obtained from the real symmetric "
            "cross-spectrum. The reported imaginary fraction measures the "
            "phase-lagged part omitted by this real frequency-local "
            "decomposition."
        ),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        frequency_cm[selected],
        power_fraction,
        marker="o",
    )
    plt.xlabel(r"Frequency (cm$^{-1}$)")
    plt.ylabel("Fraction of total power below cutoff")
    plt.tight_layout()
    plt.savefig(
        args.outdir / "frequency_power_distribution.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(
        frequency_cm[selected],
        ranks[:, 0],
        marker="o",
        label="90%",
    )
    plt.plot(
        frequency_cm[selected],
        ranks[:, 1],
        marker="s",
        label="95%",
    )
    plt.plot(
        frequency_cm[selected],
        ranks[:, 2],
        marker="^",
        label="99%",
    )
    plt.xlabel(r"Frequency (cm$^{-1}$)")
    plt.ylabel("Required shared channels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        args.outdir / "frequency_channel_ranks.png",
        dpi=300,
    )
    plt.close()

    print(
        f"Frequency resolution: "
        f"{frequency_cm[1] - frequency_cm[0]:.2f} cm^-1"
    )
    print(
        f"PCA variance included in {nmodes} modes: "
        f"{np.sum(explained):.5f}"
    )
    print(
        f"Power-weighted mean 95% channel rank: "
        f"{weighted_mean_rank95:.3f}"
    )
    print(
        f"Power-weighted omitted imaginary cross-spectrum fraction: "
        f"{weighted_imaginary_fraction:.3f}"
    )

    print("\nPower-carrying frequency bins:")
    print(
        f"{'freq/cm-1':>11} {'power frac':>11} "
        f"{'imag frac':>10} {'r90':>5} {'r95':>5} "
        f"{'r99':>5} {'F1':>8} {'F12':>8} {'F123':>8}"
    )
    print("-" * 90)

    for row in rows:
        if (
            row["power_fraction_below_cutoff"]
            < args.print_power_fraction
        ):
            continue

        print(
            f"{row['frequency_cm_inverse']:11.1f} "
            f"{row['power_fraction_below_cutoff']:11.5f} "
            f"{row['imaginary_cross_spectrum_fraction']:10.4f} "
            f"{row['rank90']:5d} "
            f"{row['rank95']:5d} "
            f"{row['rank99']:5d} "
            f"{row['channel1_fraction']:8.4f} "
            f"{row['channel2_cumulative_fraction']:8.4f} "
            f"{row['channel3_cumulative_fraction']:8.4f}"
        )

    print(f"\nWrote shared-bath diagnostics to {args.outdir}")


if __name__ == "__main__":
    main()
