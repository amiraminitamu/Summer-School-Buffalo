#!/usr/bin/env python3
"""
Run the PCA-diagonal 4D+3A open-system model with TENSO.

The classical correlation fits are mapped to TENSO Drude-Lorentz and
underdamped-Brownian spectral densities using the high-temperature relation

    C_cl(0) = 2 lambda k_B T.

For PCA mode alpha with variance sigma_alpha^2 and fitted Drude fraction f,

    lambda_D = f sigma_alpha^2 / (2 k_B T)
    lambda_B = (1-f) sigma_alpha^2 / (2 k_B T).

The correlation decay times are converted to TENSO spectral widths through

    gamma[cm^-1] = 1 / (2 pi c tau)
                  = 5308.837458876 / tau[fs].

TENSO then constructs the quantum bath correlation function using its
Bose-Einstein/Padé decomposition.  This is a PCA-diagonal approximation:
finite-lag cross-correlations between distinct PCA modes are neglected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tenso.prototypes.bath import gen_bcf
from tenso.prototypes.heom import system_multibath

HARTREE_TO_CM = 219474.6313705
KB_HARTREE_PER_K = 3.1668115634556e-6
TAU_FS_TO_WIDTH_CM = 5308.837458876


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bath-analysis",
        type=Path,
        default=Path(
            "output_tenso_bath_analysis_hvib/"
            "hamiltonian_fluctuation_pca.npz"
        ),
    )
    p.add_argument(
        "--fits",
        type=Path,
        default=Path(
            "output_tenso_correlation_fits/"
            "classical_correlation_fits.npz"
        ),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_tenso/pca_diagonal"),
    )
    p.add_argument("--nmodes", type=int, default=4)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--n-ltc", type=int, default=1)
    p.add_argument("--end-time-fs", type=float, default=5.0)
    p.add_argument("--dt-fs", type=float, default=0.05)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--vmf-atol", type=float, default=1.0e-7)
    p.add_argument("--ps2-atol", type=float, default=1.0e-7)
    p.add_argument("--ode-atol", type=float, default=1.0e-7)
    p.add_argument("--ode-rtol", type=float, default=1.0e-5)
    p.add_argument(
        "--max-auxiliary-rank",
        type=int,
        default=32,
    )
    p.add_argument(
        "--renormalize",
        action="store_true",
        help="Ask TENSO to renormalize during propagation.",
    )
    p.add_argument(
        "--frame-method",
        default="tree2",
        choices=("train", "tree2", "tree3", "tree4", "naive"),
    )
    p.add_argument(
        "--static-index",
        type=int,
        default=-1,
        help=(
            "-1 uses the global mean Hamiltonian. Values 0--9 add the "
            "corresponding between-trajectory static offset."
        ),
    )
    p.add_argument(
        "--save-checkpoint",
        action="store_true",
    )
    return p.parse_args()


def make_initial_density(coefficients: np.ndarray) -> np.ndarray:
    rho = np.zeros(
        (coefficients.shape[1], coefficients.shape[1]),
        dtype=complex,
    )
    for c in coefficients:
        rho += np.outer(c, c.conj())
    rho /= len(coefficients)
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho


def load_tenso_output(path: Path, system_dim: int):
    array = np.loadtxt(path, dtype=complex, comments="#")
    array = np.atleast_2d(array)

    expected = 1 + system_dim * system_dim
    if array.shape[1] != expected:
        raise RuntimeError(
            f"Expected {expected} columns in {path}, found "
            f"{array.shape[1]}"
        )

    time_fs = array[:, 0].real
    density = array[:, 1:].reshape(-1, system_dim, system_dim)
    return time_fs, density


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with np.load(args.bath_analysis, allow_pickle=False) as data:
        system_hamiltonian_h = np.asarray(
            data["system_hamiltonian_hartree"],
            dtype=complex,
        )
        static_offsets_h = np.asarray(
            data["static_offsets_hartree"],
            dtype=complex,
        )
        initial_coefficients = np.asarray(
            data["initial_coefficients_gauge_fixed"],
            dtype=complex,
        )
        mode_operators_all = np.asarray(
            data["mode_operators"],
            dtype=complex,
        )
        explained_all = np.asarray(
            data["pca_explained_fraction"],
            dtype=float,
        )

    with np.load(args.fits, allow_pickle=False) as data:
        variance_h2_all = np.asarray(
            data["mode_variance_hartree2"],
            dtype=float,
        )
        f_drude_all = np.asarray(data["f_drude"], dtype=float)
        tau_drude_all = np.asarray(
            data["tau_drude_fs"],
            dtype=float,
        )
        tau_brownian_all = np.asarray(
            data["tau_brownian_fs"],
            dtype=float,
        )
        frequency_b_all = np.asarray(
            data["brownian_frequency_cm_inverse"],
            dtype=float,
        )

    available = min(
        len(mode_operators_all),
        len(variance_h2_all),
    )
    if args.nmodes < 1 or args.nmodes > available:
        raise ValueError(
            f"--nmodes must be between 1 and {available}"
        )

    if args.static_index >= len(static_offsets_h):
        raise ValueError(
            f"--static-index must be -1 or 0--"
            f"{len(static_offsets_h) - 1}"
        )

    system_hamiltonian_h = system_hamiltonian_h.copy()
    if args.static_index >= 0:
        system_hamiltonian_h += static_offsets_h[args.static_index]

    # Remove an irrelevant scalar shift once more after adding static disorder.
    nstate = system_hamiltonian_h.shape[0]
    system_hamiltonian_h -= (
        np.trace(system_hamiltonian_h) / nstate
    ) * np.eye(nstate)

    sys_ham_cm = system_hamiltonian_h * HARTREE_TO_CM
    init_rdo = make_initial_density(initial_coefficients)

    sys_ops = []
    bath_correlations = []
    mode_rows = []

    kbt_h = KB_HARTREE_PER_K * args.temperature

    for mode in range(args.nmodes):
        variance_h2 = variance_h2_all[mode]
        f_drude = float(f_drude_all[mode])

        lambda_total_h = variance_h2 / (2.0 * kbt_h)
        lambda_drude_cm = (
            f_drude * lambda_total_h * HARTREE_TO_CM
        )
        lambda_brownian_cm = (
            (1.0 - f_drude)
            * lambda_total_h
            * HARTREE_TO_CM
        )

        width_drude_cm = (
            TAU_FS_TO_WIDTH_CM / tau_drude_all[mode]
        )
        width_brownian_cm = (
            TAU_FS_TO_WIDTH_CM / tau_brownian_all[mode]
        )
        frequency_b_cm = float(frequency_b_all[mode])

        correlation = gen_bcf(
            include_drude=True,
            re_d=[lambda_drude_cm],
            width_d=[width_drude_cm],
            include_brownian=True,
            freq_b=[frequency_b_cm],
            re_b=[lambda_brownian_cm],
            width_b=[width_brownian_cm],
            include_discrete=False,
            temperature=args.temperature,
            decomposition_method="Pade",
            n_ltc=args.n_ltc,
        )

        operator = 0.5 * (
            mode_operators_all[mode]
            + mode_operators_all[mode].conj().T
        )
        sys_ops.append(operator)
        bath_correlations.append(correlation)

        mode_rows.append(
            {
                "mode": mode + 1,
                "explained_fraction": float(
                    explained_all[mode]
                ),
                "variance_hartree2": float(variance_h2),
                "lambda_total_cm_inverse": float(
                    lambda_total_h * HARTREE_TO_CM
                ),
                "lambda_drude_cm_inverse": float(
                    lambda_drude_cm
                ),
                "lambda_brownian_cm_inverse": float(
                    lambda_brownian_cm
                ),
                "width_drude_cm_inverse": float(
                    width_drude_cm
                ),
                "width_brownian_cm_inverse": float(
                    width_brownian_cm
                ),
                "brownian_frequency_cm_inverse": float(
                    frequency_b_cm
                ),
                "correlation_features": int(correlation.k_max),
            }
        )

    prefix = args.outdir / (
        f"tenso_{args.nmodes:02d}m"
        f"_dim{args.dim}_rank{args.rank}"
        f"_ltc{args.n_ltc}"
        f"_static{args.static_index}"
    )

    parameter_report = {
        "model": (
            "Validated 4D+3A system with PCA-diagonal "
            "Drude+Brownian baths"
        ),
        "bath_analysis": str(args.bath_analysis),
        "correlation_fits": str(args.fits),
        "nmodes": args.nmodes,
        "retained_variance_fraction": float(
            np.sum(explained_all[: args.nmodes])
        ),
        "temperature_K": args.temperature,
        "quantum_correction": (
            "Classical C(0)=2 lambda kBT mapping followed by "
            "TENSO Bose-Einstein/Padé BCF construction"
        ),
        "finite_lag_crosscorrelations": (
            "Neglected in the PCA-diagonal approximation"
        ),
        "static_index": args.static_index,
        "numerical_tolerances": {
            "vmf_atol": args.vmf_atol,
            "ps2_atol": args.ps2_atol,
            "ode_atol": args.ode_atol,
            "ode_rtol": args.ode_rtol,
            "max_auxiliary_rank": args.max_auxiliary_rank,
            "renormalize": args.renormalize,
        },
        "system_hamiltonian_cm_inverse_real": (
            sys_ham_cm.real.tolist()
        ),
        "system_hamiltonian_cm_inverse_imag": (
            sys_ham_cm.imag.tolist()
        ),
        "initial_density_matrix_real": init_rdo.real.tolist(),
        "initial_density_matrix_imag": init_rdo.imag.tolist(),
        "modes": mode_rows,
    }
    Path(str(prefix) + ".parameters.json").write_text(
        json.dumps(parameter_report, indent=2) + "\n"
    )

    print(
        f"{'mode':>5} {'var frac':>9} {'lambda tot':>11} "
        f"{'lambda D':>10} {'lambda B':>10} "
        f"{'gamma D':>9} {'gamma B':>9} "
        f"{'nu B':>9} {'K':>4}"
    )
    print("-" * 96)
    for row in mode_rows:
        print(
            f"{row['mode']:5d} "
            f"{row['explained_fraction']:9.5f} "
            f"{row['lambda_total_cm_inverse']:11.2f} "
            f"{row['lambda_drude_cm_inverse']:10.2f} "
            f"{row['lambda_brownian_cm_inverse']:10.2f} "
            f"{row['width_drude_cm_inverse']:9.2f} "
            f"{row['width_brownian_cm_inverse']:9.2f} "
            f"{row['brownian_frequency_cm_inverse']:9.1f} "
            f"{row['correlation_features']:4d}"
        )

    print(
        f"\nStarting TENSO: {args.nmodes} baths, "
        f"{sum(c.k_max for c in bath_correlations)} total features, "
        f"dim={args.dim}, rank={args.rank}, "
        f"{args.end_time_fs} fs."
    )

    propagator = system_multibath(
        fname=str(prefix),
        init_rdo=init_rdo,
        sys_ham=sys_ham_cm,
        sys_ops=sys_ops,
        bath_correlations=bath_correlations,
        dim=args.dim,
        rank=args.rank,
        frame_method=args.frame_method,
        end_time=args.end_time_fs,
        step_time=args.dt_fs,
        vmf_atol=args.vmf_atol,
        ps2_atol=args.ps2_atol,
        ode_atol=args.ode_atol,
        ode_rtol=args.ode_rtol,
        max_auxiliary_rank=args.max_auxiliary_rank,
        renormalize=args.renormalize,
        save_checkpoint_to_file=args.save_checkpoint,
    )

    total_steps = int(np.ceil(args.end_time_fs / args.dt_fs))
    for _ in tqdm(propagator, total=total_steps):
        pass

    output_path = Path(str(prefix) + ".dat.log")
    time_fs, density = load_tenso_output(output_path, nstate)

    acceptor_projector = np.diag(
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    )
    acceptor_population = np.real(
        np.einsum(
            "ij,tji->t",
            acceptor_projector,
            density,
        )
    )
    trace = np.trace(density, axis1=1, axis2=2)
    hermiticity_error = np.array(
        [
            np.linalg.norm(rho - rho.conj().T)
            for rho in density
        ]
    )

    density_hermitian = 0.5 * (
        density + density.conj().transpose(0, 2, 1)
    )
    hermitian_trace = np.trace(
        density_hermitian,
        axis1=1,
        axis2=2,
    ).real
    density_trace_normalized = (
        density_hermitian
        / hermitian_trace[:, None, None]
    )
    acceptor_population_trace_normalized = np.real(
        np.einsum(
            "ij,tji->t",
            acceptor_projector,
            density_trace_normalized,
        )
    )
    minimum_density_eigenvalue = np.array(
        [
            np.min(np.linalg.eigvalsh(rho))
            for rho in density_trace_normalized
        ]
    )

    np.savez_compressed(
        str(prefix) + ".analysis.npz",
        time_fs=time_fs,
        density_matrix=density,
        density_matrix_hermitian_trace_normalized=(
            density_trace_normalized
        ),
        acceptor_population=acceptor_population,
        acceptor_population_trace_normalized=(
            acceptor_population_trace_normalized
        ),
        trace=trace,
        hermiticity_error=hermiticity_error,
        minimum_density_eigenvalue=(
            minimum_density_eigenvalue
        ),
        acceptor_projector=acceptor_projector,
    )

    analysis_summary = {
        "output": str(output_path),
        "initial_acceptor_population": float(
            acceptor_population[0]
        ),
        "final_acceptor_population": float(
            acceptor_population[-1]
        ),
        "final_acceptor_population_trace_normalized": float(
            acceptor_population_trace_normalized[-1]
        ),
        "maximum_acceptor_population": float(
            np.max(acceptor_population)
        ),
        "maximum_trace_error": float(
            np.max(np.abs(trace - 1.0))
        ),
        "maximum_hermiticity_error": float(
            np.max(hermiticity_error)
        ),
        "minimum_trace_normalized_density_eigenvalue": float(
            np.min(minimum_density_eigenvalue)
        ),
    }
    Path(str(prefix) + ".summary.json").write_text(
        json.dumps(analysis_summary, indent=2) + "\n"
    )

    print("\nTENSO smoke summary:")
    print(
        f"  Initial P_C60: "
        f"{analysis_summary['initial_acceptor_population']:.8f}"
    )
    print(
        f"  Final P_C60:   "
        f"{analysis_summary['final_acceptor_population']:.8f}"
    )
    print(
        f"  Final P_C60 (Hermitian/trace-normalized): "
        f"{analysis_summary['final_acceptor_population_trace_normalized']:.8f}"
    )
    print(
        f"  Max trace error: "
        f"{analysis_summary['maximum_trace_error']:.3e}"
    )
    print(
        f"  Max Hermiticity error: "
        f"{analysis_summary['maximum_hermiticity_error']:.3e}"
    )


if __name__ == "__main__":
    main()
