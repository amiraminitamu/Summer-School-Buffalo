#!/usr/bin/env python3
"""
Analyze complete vibronic-Hamiltonian fluctuations of the validated 4D+3A model.

This script prepares, but does not yet fit, the bath model needed by TENSO.

Workflow
--------
1. Load the 7x7 midpoint vibronic Hamiltonian used for propagation, including matrix-log derivative couplings.
2. Apply a trajectory-wide phase convention based on the initial state and
   donor-acceptor couplings.
3. Separate:
       H_r(t) = H_system + H_static,r + delta_H_r(t)
   where H_static,r is the trajectory-specific mean offset and delta_H_r(t)
   contains only within-trajectory fluctuations.
4. Expand each traceless Hermitian fluctuation in an orthonormal generalized
   Gell-Mann basis.
5. Perform PCA on all within-trajectory fluctuations.
6. Compute per-mode autocorrelation functions and averaged periodograms
   without joining unrelated trajectory endpoints.

Outputs are intended for deciding how many noncommuting bath operators are
needed and for fitting each retained mode to Drude-Lorentz and/or Brownian
oscillator correlation functions in the following step.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HARTREE_TO_EV = 27.211386245988
FS_INV_TO_WAVENUMBER = 33356.40951981521


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path("output_reduced_4d3a"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_tenso_bath_analysis"),
    )
    p.add_argument("--ntraj", type=int, default=10)
    p.add_argument("--nmodes", type=int, default=8)
    p.add_argument(
        "--hamiltonian-key",
        default="reduced_hvib_mid_hartree",
        help=(
            "Array in model_4d3a.npz to analyze. The default includes "
            "the matrix-log derivative-coupling contribution and is the "
            "Hamiltonian actually used for reduced propagation."
        ),
    )
    return p.parse_args()


def hermitize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def remove_trace(a: np.ndarray) -> np.ndarray:
    n = a.shape[-1]
    trace = np.trace(a, axis1=-2, axis2=-1) / n
    return a - trace[..., None, None] * np.eye(n)


def generalized_gell_mann(n: int) -> tuple[np.ndarray, list[str]]:
    """
    Return an orthonormal basis of traceless Hermitian n x n matrices.

    Tr(Q_a Q_b) = delta_ab.
    """
    matrices = []
    labels = []

    # Symmetric and antisymmetric off-diagonal generators.
    for i in range(n):
        for j in range(i + 1, n):
            symmetric = np.zeros((n, n), dtype=complex)
            symmetric[i, j] = 1.0 / np.sqrt(2.0)
            symmetric[j, i] = 1.0 / np.sqrt(2.0)
            matrices.append(symmetric)
            labels.append(f"Re({i},{j})")

            antisymmetric = np.zeros((n, n), dtype=complex)
            antisymmetric[i, j] = -1j / np.sqrt(2.0)
            antisymmetric[j, i] = 1j / np.sqrt(2.0)
            matrices.append(antisymmetric)
            labels.append(f"Im({i},{j})")

    # Diagonal traceless generators.
    for k in range(1, n):
        diagonal = np.zeros((n, n), dtype=complex)
        diagonal[np.arange(k), np.arange(k)] = 1.0
        diagonal[k, k] = -float(k)
        diagonal /= np.sqrt(k * (k + 1.0))
        matrices.append(diagonal)
        labels.append(f"Diag{k}")

    basis = np.asarray(matrices)
    gram = np.einsum("aij,bji->ab", basis, basis).real
    error = np.linalg.norm(gram - np.eye(n * n - 1))
    if error > 1.0e-12:
        raise RuntimeError(
            f"Hermitian basis is not orthonormal: error={error:.3e}"
        )

    return basis, labels


def matrix_to_features(
    matrices: np.ndarray,
    operator_basis: np.ndarray,
) -> np.ndarray:
    """
    x[...,a] = Tr(Q_a H[...]).
    """
    values = np.einsum(
        "aij,...ji->...a",
        operator_basis,
        matrices,
    )
    if np.max(np.abs(values.imag)) > 1.0e-10:
        raise RuntimeError(
            "Hermitian expansion produced unexpectedly complex coefficients"
        )
    return values.real


def features_to_operators(
    vectors: np.ndarray,
    operator_basis: np.ndarray,
) -> np.ndarray:
    return np.einsum("pa,aij->pij", vectors, operator_basis)


def canonical_phase_matrix(
    h0: np.ndarray,
    c0: np.ndarray,
    ndonor: int = 4,
) -> np.ndarray:
    """
    Construct one diagonal unitary gauge for the entire trajectory.

    The most populated donor coefficient is made positive. Acceptor phases
    are fixed using their coupling to that donor. Remaining donor phases are
    fixed by their strongest coupling to the already fixed acceptors.
    """
    n = h0.shape[0]
    phases = np.ones(n, dtype=complex)

    anchor = int(np.argmax(np.abs(c0[:ndonor]) ** 2))
    if abs(c0[anchor]) > 1.0e-14:
        phases[anchor] = np.exp(1j * np.angle(c0[anchor]))

    acceptors = list(range(ndonor, n))
    for a in acceptors:
        value = h0[anchor, a]
        if abs(value) > 1.0e-12:
            phases[a] = (
                phases[anchor]
                * np.exp(-1j * np.angle(value))
            )

    for d in range(ndonor):
        if d == anchor:
            continue
        if acceptors:
            magnitudes = np.abs(h0[d, acceptors])
            a = acceptors[int(np.argmax(magnitudes))]
            value = h0[d, a]
            if abs(value) > 1.0e-12:
                phases[d] = (
                    phases[a]
                    * np.exp(1j * np.angle(value))
                )
            elif abs(c0[d]) > 1.0e-12:
                phases[d] = np.exp(1j * np.angle(c0[d]))

    return np.diag(phases)


def autocorrelation_unbiased(series: np.ndarray) -> np.ndarray:
    """
    Unbiased autocorrelation for one mean-zero time series.
    """
    n = len(series)
    result = np.empty(n, dtype=float)
    for lag in range(n):
        result[lag] = np.dot(
            series[: n - lag],
            series[lag:],
        ) / (n - lag)
    return result


def averaged_periodogram(
    series_by_trajectory: np.ndarray,
    dt_fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average one-sided Hann-window periodograms across trajectories.
    """
    ntraj, ntime = series_by_trajectory.shape
    window = np.hanning(ntime)
    normalization = np.sum(window**2)

    frequency_fs_inv = np.fft.rfftfreq(ntime, d=dt_fs)
    spectra = []

    for trajectory in range(ntraj):
        y = series_by_trajectory[trajectory]
        y = y - np.mean(y)
        transform = np.fft.rfft(window * y)
        psd = dt_fs * np.abs(transform) ** 2 / normalization

        if ntime % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0
        spectra.append(psd)

    return frequency_fs_inv, np.mean(spectra, axis=0)


def dominant_elements(
    operator: np.ndarray,
    count: int = 6,
) -> str:
    elements = []
    n = operator.shape[0]

    for i in range(n):
        elements.append((abs(operator[i, i]), f"({i},{i})"))
        for j in range(i + 1, n):
            elements.append(
                (np.sqrt(2.0) * abs(operator[i, j]), f"({i},{j})")
            )

    elements.sort(reverse=True)
    return "; ".join(
        f"{label}:{value:.3f}"
        for value, label in elements[:count]
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    hamiltonians = []
    initial_coefficients = []
    time_reference = None

    for trajectory in range(args.ntraj):
        path = (
            args.root
            / f"traj_{trajectory:02d}"
            / "model_4d3a.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            frame_time_fs = np.asarray(data["time_fs"], dtype=float)
            if args.hamiltonian_key not in data.files:
                raise KeyError(
                    f"{args.hamiltonian_key!r} is not present in {path}. "
                    f"Available arrays: {data.files}"
                )
            h = np.asarray(
                data[args.hamiltonian_key],
                dtype=complex,
            )
            c0 = np.asarray(
                data["initial_c0_reduced"],
                dtype=complex,
            )

        if len(h) == len(frame_time_fs):
            time_fs = frame_time_fs
        elif len(h) == len(frame_time_fs) - 1:
            time_fs = 0.5 * (
                frame_time_fs[:-1] + frame_time_fs[1:]
            )
        else:
            raise RuntimeError(
                f"Cannot align {args.hamiltonian_key} with time grid: "
                f"H shape={h.shape}, time shape={frame_time_fs.shape}"
            )

        if time_reference is None:
            time_reference = time_fs
        elif not np.array_equal(time_reference, time_fs):
            raise RuntimeError("Trajectory time grids do not match")

        gauge = canonical_phase_matrix(h[0], c0)
        h_gauge = np.einsum(
            "ij,tjk,kl->til",
            gauge.conj().T,
            h,
            gauge,
        )
        c0_gauge = gauge.conj().T @ c0

        hamiltonians.append(
            np.asarray([hermitize(x) for x in h_gauge])
        )
        initial_coefficients.append(c0_gauge)

    hamiltonians = np.asarray(hamiltonians)
    initial_coefficients = np.asarray(initial_coefficients)

    ntraj, ntime, nstate, _ = hamiltonians.shape
    dt_fs = float(time_reference[1] - time_reference[0])

    # Global energy shifts do not influence populations.
    h_traceless = remove_trace(hamiltonians)

    trajectory_means = np.mean(h_traceless, axis=1)
    system_hamiltonian = np.mean(trajectory_means, axis=0)
    static_offsets = trajectory_means - system_hamiltonian
    dynamic_fluctuations = (
        h_traceless - trajectory_means[:, None, :, :]
    )

    operator_basis, operator_labels = generalized_gell_mann(nstate)
    dynamic_features = matrix_to_features(
        dynamic_fluctuations,
        operator_basis,
    )
    static_features = matrix_to_features(
        static_offsets,
        operator_basis,
    )

    x = dynamic_features.reshape(ntraj * ntime, -1)
    covariance = x.T @ x / (len(x) - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    total_variance = float(np.sum(eigenvalues))
    explained = np.divide(
        eigenvalues,
        total_variance,
        out=np.zeros_like(eigenvalues),
        where=total_variance > 0.0,
    )
    cumulative = np.cumsum(explained)

    nmodes = min(args.nmodes, len(eigenvalues))
    mode_vectors = eigenvectors[:, :nmodes].T
    mode_operators = features_to_operators(
        mode_vectors,
        operator_basis,
    )

    # Coordinates have units of Hartree because Q_alpha is dimensionless and
    # Frobenius-normalized.
    mode_coordinates = np.einsum(
        "rta,pa->rpt",
        dynamic_features,
        mode_vectors,
    )

    lag_fs = np.arange(ntime) * dt_fs
    mode_acf = np.zeros((nmodes, ntime), dtype=float)
    mode_acf_sem = np.zeros_like(mode_acf)

    frequencies_cm = None
    mode_psd = []

    for mode in range(nmodes):
        individual_acf = np.stack(
            [
                autocorrelation_unbiased(
                    mode_coordinates[trajectory, mode]
                )
                for trajectory in range(ntraj)
            ]
        )
        mode_acf[mode] = np.mean(individual_acf, axis=0)
        mode_acf_sem[mode] = (
            np.std(individual_acf, axis=0, ddof=1)
            / np.sqrt(ntraj)
        )

        frequency_fs_inv, psd = averaged_periodogram(
            mode_coordinates[:, mode, :],
            dt_fs,
        )
        frequencies_cm = (
            frequency_fs_inv * FS_INV_TO_WAVENUMBER
        )
        mode_psd.append(psd)

    mode_psd = np.asarray(mode_psd)

    dynamic_rms_h = float(
        np.sqrt(np.mean(np.sum(x**2, axis=1)))
    )
    static_rms_h = float(
        np.sqrt(np.mean(np.sum(static_features**2, axis=1)))
    )

    np.savez_compressed(
        args.outdir / "hamiltonian_fluctuation_pca.npz",
        time_fs=time_reference,
        system_hamiltonian_hartree=system_hamiltonian,
        trajectory_mean_hamiltonians_hartree=trajectory_means,
        static_offsets_hartree=static_offsets,
        dynamic_fluctuations_hartree=dynamic_fluctuations,
        initial_coefficients_gauge_fixed=initial_coefficients,
        operator_basis=operator_basis,
        covariance_hartree2=covariance,
        pca_eigenvalues_hartree2=eigenvalues,
        pca_explained_fraction=explained,
        pca_cumulative_fraction=cumulative,
        mode_feature_vectors=mode_vectors,
        mode_operators=mode_operators,
        mode_coordinates_hartree=mode_coordinates,
        lag_fs=lag_fs,
        mode_acf_hartree2=mode_acf,
        mode_acf_sem_hartree2=mode_acf_sem,
        frequency_cm_inverse=frequencies_cm,
        mode_periodogram_hartree2_fs=mode_psd,
    )

    with (args.outdir / "pca_modes.csv").open(
        "w", newline=""
    ) as handle:
        fields = [
            "mode",
            "variance_hartree2",
            "rms_amplitude_eV",
            "explained_fraction",
            "cumulative_fraction",
            "acf_zero_hartree2",
            "dominant_operator_elements",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for mode in range(nmodes):
            writer.writerow(
                {
                    "mode": mode + 1,
                    "variance_hartree2": eigenvalues[mode],
                    "rms_amplitude_eV": (
                        np.sqrt(eigenvalues[mode])
                        * HARTREE_TO_EV
                    ),
                    "explained_fraction": explained[mode],
                    "cumulative_fraction": cumulative[mode],
                    "acf_zero_hartree2": mode_acf[mode, 0],
                    "dominant_operator_elements": (
                        dominant_elements(mode_operators[mode])
                    ),
                }
            )

    with (args.outdir / "mode_autocorrelations.csv").open(
        "w", newline=""
    ) as handle:
        fields = ["lag_fs"]
        for mode in range(nmodes):
            fields += [
                f"mode_{mode + 1}_acf_hartree2",
                f"mode_{mode + 1}_acf_sem_hartree2",
                f"mode_{mode + 1}_normalized_acf",
            ]

        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for index, lag in enumerate(lag_fs):
            row = {"lag_fs": lag}
            for mode in range(nmodes):
                c0 = mode_acf[mode, 0]
                row[f"mode_{mode + 1}_acf_hartree2"] = (
                    mode_acf[mode, index]
                )
                row[f"mode_{mode + 1}_acf_sem_hartree2"] = (
                    mode_acf_sem[mode, index]
                )
                row[f"mode_{mode + 1}_normalized_acf"] = (
                    mode_acf[mode, index] / c0
                    if abs(c0) > 0.0
                    else np.nan
                )
            writer.writerow(row)

    with (args.outdir / "mode_periodograms.csv").open(
        "w", newline=""
    ) as handle:
        fields = ["frequency_cm_inverse"] + [
            f"mode_{mode + 1}_psd_hartree2_fs"
            for mode in range(nmodes)
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for index, frequency in enumerate(frequencies_cm):
            row = {"frequency_cm_inverse": frequency}
            for mode in range(nmodes):
                row[f"mode_{mode + 1}_psd_hartree2_fs"] = (
                    mode_psd[mode, index]
                )
            writer.writerow(row)

    summary = {
        "hamiltonian_key": args.hamiltonian_key,
        "includes_derivative_couplings": (
            args.hamiltonian_key == "reduced_hvib_mid_hartree"
        ),
        "ntrajectories": ntraj,
        "nframes_per_trajectory": ntime,
        "time_step_fs": dt_fs,
        "trajectory_length_fs": float(
            time_reference[-1] - time_reference[0]
        ),
        "frequency_spacing_cm_inverse": float(
            frequencies_cm[1] - frequencies_cm[0]
        ),
        "system_dimension": nstate,
        "traceless_operator_dimension": nstate * nstate - 1,
        "dynamic_fluctuation_rms_eV": (
            dynamic_rms_h * HARTREE_TO_EV
        ),
        "static_disorder_rms_eV": (
            static_rms_h * HARTREE_TO_EV
        ),
        "modes_for_90_percent": int(
            np.searchsorted(cumulative, 0.90) + 1
        ),
        "modes_for_95_percent": int(
            np.searchsorted(cumulative, 0.95) + 1
        ),
        "modes_for_99_percent": int(
            np.searchsorted(cumulative, 0.99) + 1
        ),
        "important_note": (
            "Autocorrelations describe classical within-trajectory "
            "Hamiltonian fluctuations. A quantum correction and a "
            "Drude-Lorentz/Brownian-oscillator fit are still required "
            "before constructing the TENSO bath correlation functions."
        ),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    plt.figure(figsize=(7.2, 4.8))
    indices = np.arange(1, min(20, len(explained)) + 1)
    plt.plot(indices, explained[: len(indices)], marker="o")
    plt.plot(indices, cumulative[: len(indices)], marker="s")
    plt.axhline(0.90, linestyle="--", linewidth=1.0)
    plt.axhline(0.95, linestyle=":", linewidth=1.0)
    plt.xlabel("PCA mode")
    plt.ylabel("Variance fraction")
    plt.tight_layout()
    plt.savefig(args.outdir / "pca_explained_variance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for mode in range(nmodes):
        normalized = (
            mode_acf[mode] / mode_acf[mode, 0]
            if abs(mode_acf[mode, 0]) > 0.0
            else mode_acf[mode]
        )
        plt.plot(
            lag_fs,
            normalized,
            linewidth=1.4,
            label=f"Mode {mode + 1}",
        )
    plt.xlabel("Lag time (fs)")
    plt.ylabel("Normalized autocorrelation")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        args.outdir / "mode_autocorrelations.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for mode in range(nmodes):
        scale = np.max(mode_psd[mode, 1:])
        normalized = (
            mode_psd[mode] / scale
            if scale > 0.0
            else mode_psd[mode]
        )
        plt.plot(
            frequencies_cm,
            normalized,
            linewidth=1.2,
            label=f"Mode {mode + 1}",
        )
    plt.xlim(0.0, min(4000.0, frequencies_cm[-1]))
    plt.xlabel(r"Frequency (cm$^{-1}$)")
    plt.ylabel("Normalized classical power")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        args.outdir / "mode_periodograms.png",
        dpi=300,
    )
    plt.close()

    print(
        f"{'mode':>5} {'RMS/eV':>10} {'fraction':>10} "
        f"{'cumulative':>11}  dominant operator elements"
    )
    print("-" * 104)

    for mode in range(nmodes):
        print(
            f"{mode + 1:5d} "
            f"{np.sqrt(eigenvalues[mode]) * HARTREE_TO_EV:10.5f} "
            f"{explained[mode]:10.5f} "
            f"{cumulative[mode]:11.5f}  "
            f"{dominant_elements(mode_operators[mode])}"
        )

    print("\nFluctuation summary:")
    print(
        f"  Dynamic within-path RMS: "
        f"{dynamic_rms_h * HARTREE_TO_EV:.5f} eV"
    )
    print(
        f"  Static between-path RMS: "
        f"{static_rms_h * HARTREE_TO_EV:.5f} eV"
    )
    print(
        f"  Modes for 90/95/99%: "
        f"{summary['modes_for_90_percent']}/"
        f"{summary['modes_for_95_percent']}/"
        f"{summary['modes_for_99_percent']}"
    )
    print(
        f"  Frequency spacing: "
        f"{summary['frequency_spacing_cm_inverse']:.2f} cm^-1"
    )
    print(f"\nWrote TENSO bath diagnostics to {args.outdir}")


if __name__ == "__main__":
    main()
