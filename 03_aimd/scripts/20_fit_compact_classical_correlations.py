#!/usr/bin/env python3
"""
Fit compact classical correlation models to the leading PCA bath coordinates.

For each retained PCA mode, fit the normalized within-trajectory
autocorrelation over a short, statistically reliable time window to

    C(t)/C(0) =
        f_D exp(-t/tau_D)
        + (1-f_D) exp(-t/tau_B) cos(2*pi*c*nu_B*t)

This separates a low-frequency relaxational component from one damped
vibrational band.  The fit is a classical correlation model only; the
subsequent step must apply an explicit quantum correction and translate the
parameters into the correlation-function convention expected by TENSO.

The script reports per-mode fits for a 4 -> 8 -> 12 mode convergence sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# cycles/fs for one wavenumber
CM_INV_TO_CYCLES_PER_FS = 2.99792458e-5
HARTREE_TO_EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "output_tenso_bath_analysis_hvib/"
            "hamiltonian_fluctuation_pca.npz"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_tenso_correlation_fits"),
    )
    parser.add_argument("--nmodes", type=int, default=12)
    parser.add_argument(
        "--fit-max-fs",
        type=float,
        default=20.0,
        help=(
            "Fit only the early correlation window. Later lags have much "
            "larger estimator variance for 100 fs trajectories."
        ),
    )
    return parser.parse_args()


def correlation_model(
    time_fs: np.ndarray,
    f_drude: float,
    tau_drude_fs: float,
    tau_brownian_fs: float,
    frequency_cm_inverse: float,
) -> np.ndarray:
    phase = (
        2.0
        * np.pi
        * CM_INV_TO_CYCLES_PER_FS
        * frequency_cm_inverse
        * time_fs
    )
    return (
        f_drude * np.exp(-time_fs / tau_drude_fs)
        + (1.0 - f_drude)
        * np.exp(-time_fs / tau_brownian_fs)
        * np.cos(phase)
    )


def initial_frequency(
    frequency_cm: np.ndarray,
    periodogram: np.ndarray,
) -> float:
    positive = np.flatnonzero(frequency_cm > 0.0)
    if positive.size == 0:
        return 1000.0
    index = int(positive[np.argmax(periodogram[positive])])
    return float(frequency_cm[index])


def fit_one_mode(
    time_fs: np.ndarray,
    normalized_acf: np.ndarray,
    normalized_sem: np.ndarray,
    frequency_guess_cm: float,
) -> dict:
    sigma = np.maximum(normalized_sem, 0.03)

    def residual(parameters: np.ndarray) -> np.ndarray:
        model = correlation_model(time_fs, *parameters)
        return (model - normalized_acf) / sigma

    guesses = [
        [0.10, 15.0, 6.0, frequency_guess_cm],
        [0.05, 30.0, 5.0, frequency_guess_cm],
        [0.20, 10.0, 8.0, frequency_guess_cm],
        [0.00, 20.0, 6.0, frequency_guess_cm],
    ]

    lower = np.array([0.0, 0.5, 0.5, 150.0])
    upper = np.array([0.8, 100.0, 100.0, 2500.0])

    best = None
    for guess in guesses:
        result = least_squares(
            residual,
            x0=np.clip(guess, lower, upper),
            bounds=(lower, upper),
            max_nfev=20000,
        )
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best["weighted_mse"]:
            best = {
                "parameters": result.x,
                "weighted_mse": score,
                "success": bool(result.success),
                "message": result.message,
            }

    parameters = best["parameters"]
    fitted = correlation_model(time_fs, *parameters)
    unweighted_rmse = float(
        np.sqrt(np.mean((fitted - normalized_acf) ** 2))
    )
    max_abs_error = float(
        np.max(np.abs(fitted - normalized_acf))
    )

    return {
        "f_drude": float(parameters[0]),
        "tau_drude_fs": float(parameters[1]),
        "tau_brownian_fs": float(parameters[2]),
        "frequency_cm_inverse": float(parameters[3]),
        "weighted_mse": best["weighted_mse"],
        "normalized_rmse": unweighted_rmse,
        "normalized_max_abs_error": max_abs_error,
        "success": best["success"],
        "message": best["message"],
        "fitted": fitted,
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input, allow_pickle=False) as data:
        lag_fs_all = np.asarray(data["lag_fs"], dtype=float)
        acf_all = np.asarray(
            data["mode_acf_hartree2"],
            dtype=float,
        )
        sem_all = np.asarray(
            data["mode_acf_sem_hartree2"],
            dtype=float,
        )
        periodograms_all = np.asarray(
            data["mode_periodogram_hartree2_fs"],
            dtype=float,
        )
        frequency_cm = np.asarray(
            data["frequency_cm_inverse"],
            dtype=float,
        )
        eigenvalues = np.asarray(
            data["pca_eigenvalues_hartree2"],
            dtype=float,
        )
        explained = np.asarray(
            data["pca_explained_fraction"],
            dtype=float,
        )
        cumulative = np.asarray(
            data["pca_cumulative_fraction"],
            dtype=float,
        )
        mode_operators = np.asarray(
            data["mode_operators"],
            dtype=complex,
        )

    available = acf_all.shape[0]
    if args.nmodes > available:
        raise RuntimeError(
            f"Requested {args.nmodes} modes, but only {available} are stored."
        )

    mask = lag_fs_all <= args.fit_max_fs
    lag_fs = lag_fs_all[mask]

    rows = []
    fitted_all = []

    for mode in range(args.nmodes):
        c0 = float(acf_all[mode, 0])
        if c0 <= 0.0:
            raise RuntimeError(
                f"Mode {mode + 1} has nonpositive C(0)={c0}"
            )

        normalized = acf_all[mode, mask] / c0
        normalized_sem = sem_all[mode, mask] / c0
        guess = initial_frequency(
            frequency_cm,
            periodograms_all[mode],
        )

        fit = fit_one_mode(
            lag_fs,
            normalized,
            normalized_sem,
            guess,
        )
        fitted_all.append(fit["fitted"])

        rows.append(
            {
                "mode": mode + 1,
                "variance_hartree2": eigenvalues[mode],
                "rms_amplitude_eV": (
                    np.sqrt(eigenvalues[mode]) * HARTREE_TO_EV
                ),
                "explained_fraction": explained[mode],
                "cumulative_fraction": cumulative[mode],
                "f_drude": fit["f_drude"],
                "tau_drude_fs": fit["tau_drude_fs"],
                "tau_brownian_fs": fit["tau_brownian_fs"],
                "brownian_frequency_cm_inverse": (
                    fit["frequency_cm_inverse"]
                ),
                "normalized_rmse": fit["normalized_rmse"],
                "normalized_max_abs_error": (
                    fit["normalized_max_abs_error"]
                ),
                "fit_success": fit["success"],
            }
        )

    fitted_all = np.asarray(fitted_all)

    with (args.outdir / "classical_correlation_fits.csv").open(
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
        args.outdir / "classical_correlation_fits.npz",
        fit_lag_fs=lag_fs,
        normalized_acf=(
            acf_all[: args.nmodes, mask]
            / acf_all[: args.nmodes, 0, None]
        ),
        normalized_acf_sem=(
            sem_all[: args.nmodes, mask]
            / acf_all[: args.nmodes, 0, None]
        ),
        normalized_fitted_acf=fitted_all,
        mode_variance_hartree2=eigenvalues[: args.nmodes],
        mode_operators=mode_operators[: args.nmodes],
        explained_fraction=explained[: args.nmodes],
        cumulative_fraction=cumulative[: args.nmodes],
        f_drude=np.array([row["f_drude"] for row in rows]),
        tau_drude_fs=np.array(
            [row["tau_drude_fs"] for row in rows]
        ),
        tau_brownian_fs=np.array(
            [row["tau_brownian_fs"] for row in rows]
        ),
        brownian_frequency_cm_inverse=np.array(
            [
                row["brownian_frequency_cm_inverse"]
                for row in rows
            ]
        ),
    )

    summary = {
        "input": str(args.input),
        "modes_fitted": args.nmodes,
        "variance_fraction_retained": float(
            np.sum(explained[: args.nmodes])
        ),
        "fit_window_fs": args.fit_max_fs,
        "median_normalized_rmse": float(
            np.median(
                [row["normalized_rmse"] for row in rows]
            )
        ),
        "maximum_normalized_rmse": float(
            np.max(
                [row["normalized_rmse"] for row in rows]
            )
        ),
        "important_note": (
            "These are fits to classical correlation functions. They are "
            "not yet TENSO-ready quantum bath parameters. A declared quantum "
            "correction and TENSO convention mapping are required next."
        ),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    plt.figure(figsize=(7.2, 5.2))
    for mode in range(min(args.nmodes, 12)):
        normalized = (
            acf_all[mode, mask] / acf_all[mode, 0]
        )
        plt.plot(
            lag_fs,
            normalized,
            linewidth=0.9,
            alpha=0.45,
        )
        plt.plot(
            lag_fs,
            fitted_all[mode],
            linewidth=1.2,
            linestyle="--",
        )
    plt.xlabel("Lag time (fs)")
    plt.ylabel("Normalized autocorrelation")
    plt.tight_layout()
    plt.savefig(
        args.outdir / "classical_correlation_fits.png",
        dpi=300,
    )
    plt.close()

    print(
        f"{'mode':>5} {'RMS/eV':>9} {'cum var':>9} "
        f"{'f_D':>7} {'tau_D/fs':>9} {'tau_B/fs':>9} "
        f"{'nu_B/cm-1':>11} {'RMSE':>8}"
    )
    print("-" * 82)

    for row in rows:
        print(
            f"{row['mode']:5d} "
            f"{row['rms_amplitude_eV']:9.5f} "
            f"{row['cumulative_fraction']:9.5f} "
            f"{row['f_drude']:7.4f} "
            f"{row['tau_drude_fs']:9.3f} "
            f"{row['tau_brownian_fs']:9.3f} "
            f"{row['brownian_frequency_cm_inverse']:11.1f} "
            f"{row['normalized_rmse']:8.4f}"
        )

    print("\nFit summary:")
    print(
        f"  Variance retained: "
        f"{summary['variance_fraction_retained']:.5f}"
    )
    print(
        f"  Median normalized RMSE: "
        f"{summary['median_normalized_rmse']:.4f}"
    )
    print(
        f"  Maximum normalized RMSE: "
        f"{summary['maximum_normalized_rmse']:.4f}"
    )
    print(f"\nWrote fits to {args.outdir}")


if __name__ == "__main__":
    main()
