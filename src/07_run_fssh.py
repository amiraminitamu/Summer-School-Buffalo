#!/usr/bin/env python3
"""
Classical-path FSSH (CPA/NBRA-style) for a precomputed 10-state
time-dependent vibronic Hamiltonian.

The 10x10 electronic TDSE is propagated exactly with a matrix exponential,
while Libra supplies Tully FSSH hopping probabilities.

The nuclear trajectory is fixed, so there is no momentum rescaling or
electronic back-reaction. The primary physical C60 observable is the full
wavefunction expectation value c^\\dagger P_C60 c, including coherences.
The active-surface Mulliken value is saved only as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from liblibra_core import CMATRIX
from scipy.linalg import expm

# Current Libra API (preferred). Keep a compatibility fallback for older builds.
try:
    from liblibra_core import dyn_control_params, hopping_probabilities_fssh
    LIBRA_FSSH_API = "current"
except ImportError:
    from liblibra_core import compute_hopping_probabilities_fssh
    LIBRA_FSSH_API = "legacy"

FS_TO_AU = 41.3413745758
KB_HARTREE_PER_K = 3.166811563e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--nframes", type=int, default=200)
    p.add_argument("--nsh", type=int, default=20000,
                   help="Number of stochastic active-surface realizations.")
    p.add_argument("--substeps", type=int, default=20,
                   help="FSSH probability substeps per 0.5 fs nuclear interval.")
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument(
        "--boltzmann",
        action="store_true",
        help="Boltzmann-scale upward hops for fixed-path NBRA sensitivity analysis.",
    )
    p.add_argument(
        "--initial-surfaces",
        choices=("population", "dominant"),
        default="population",
        help="Sample active surfaces from |c0|^2 or place all on its dominant state.",
    )
    p.add_argument(
        "--probability-tolerance",
        type=float,
        default=1.0e-8,
        help="Tolerance for probability sanitation and normalization.",
    )
    return p.parse_args()


def array_to_cmatrix(a: np.ndarray) -> CMATRIX:
    a = np.asarray(a, dtype=complex)
    if a.ndim == 1:
        out = CMATRIX(a.shape[0], 1)
        for i, value in enumerate(a):
            out.set(i, 0, complex(value))
        return out

    if a.ndim == 2:
        out = CMATRIX(a.shape[0], a.shape[1])
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                out.set(i, j, complex(a[i, j]))
        return out

    raise ValueError(f"Expected a 1-D or 2-D array, got shape {a.shape}")



def matrix_to_array(g, nstates: int) -> np.ndarray:
    return np.array(
        [[float(g.get(i, j)) for j in range(nstates)] for i in range(nstates)],
        dtype=float,
    )


def physical_fragment_population(c: np.ndarray, projector: np.ndarray) -> float:
    value = np.vdot(c, projector @ c)
    if abs(value.imag) > 1.0e-8:
        raise RuntimeError(f"Fragment population has a large imaginary part: {value}")
    return float(value.real)


def libra_probability_matrix_for_active_states(
    coeff: CMATRIX,
    hvib: CMATRIX,
    active_states: np.ndarray,
    dt_au: float,
    temperature: float,
    use_boltzmann: bool,
    nstates: int,
    tolerance: float,
) -> tuple[np.ndarray, float, int]:
    """
    Compute FSSH probabilities only for states that are active in at least
    one stochastic realization.

    The current Libra API provides the active-state-specific overload

        hopping_probabilities_fssh(params, density_matrix, Hvib, active_state)

    and Libra's own hop(...) function normalizes the returned probability
    vector before drawing the next state. We reproduce that normalization
    here so vectorized sampling of many walkers has the same semantics.

    Returns
    -------
    probability_matrix
        Normalized rows for occupied active states.
    max_raw_sum
        Largest sum of a Libra-returned probability vector before the same
        normalization performed by Libra's hop(...) routine.
    renormalized_count
        Number of occupied-state probability vectors whose raw sum differed
        from one by more than `tolerance`.
    """
    occupied = np.unique(active_states)
    probabilities = np.zeros((nstates, nstates), dtype=float)
    max_raw_sum = 0.0
    renormalized_count = 0

    if LIBRA_FSSH_API == "current":
        params = dyn_control_params()
        params.dt = float(dt_au)
        params.Temperature = float(temperature)
        params.use_boltz_factor = int(use_boltzmann)

        denmat = coeff * coeff.H()

        for state_value in occupied:
            state = int(state_value)
            result = hopping_probabilities_fssh(
                params, denmat, hvib, state
            )
            p = np.array(
                [float(result[j]) for j in range(nstates)],
                dtype=float,
            )

            if np.any(p < -tolerance):
                raise RuntimeError(
                    f"Libra returned negative probabilities for state "
                    f"{state}: {p}"
                )
            p[p < 0.0] = 0.0

            raw_sum = float(np.sum(p))
            max_raw_sum = max(max_raw_sum, raw_sum)

            if raw_sum <= tolerance:
                p[:] = 0.0
                p[state] = 1.0
            else:
                if abs(raw_sum - 1.0) > tolerance:
                    renormalized_count += 1
                p /= raw_sum

            probabilities[state] = p

        return probabilities, max_raw_sum, renormalized_count

    # Compatibility path for older Libra builds exposing only the matrix form.
    result = compute_hopping_probabilities_fssh(coeff, hvib, float(dt_au))
    raw_matrix = matrix_to_array(result, nstates)
    energies = np.array(
        [float(hvib.get(i, i).real) for i in range(nstates)]
    )

    for state_value in occupied:
        state = int(state_value)
        p = np.maximum(raw_matrix[state], 0.0)

        if use_boltzmann:
            for j in range(nstates):
                if j != state and energies[j] > energies[state]:
                    p[j] *= np.exp(
                        -(energies[j] - energies[state])
                        / (KB_HARTREE_PER_K * temperature)
                    )

        raw_sum = float(np.sum(p))
        max_raw_sum = max(max_raw_sum, raw_sum)

        if raw_sum <= tolerance:
            p[:] = 0.0
            p[state] = 1.0
        else:
            if abs(raw_sum - 1.0) > tolerance:
                renormalized_count += 1
            p /= raw_sum

        probabilities[state] = p

    return probabilities, max_raw_sum, renormalized_count


def sample_hops(
    active_states: np.ndarray,
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """
    Vectorized equivalent of Libra's row-wise hop sampling.
    """
    new_states = active_states.copy()
    nstates = probabilities.shape[0]

    for state in range(nstates):
        indices = np.flatnonzero(active_states == state)
        if indices.size == 0:
            continue

        cumulative = np.cumsum(probabilities[state])
        cumulative[-1] = 1.0
        random_values = rng.random(indices.size)
        new_states[indices] = np.searchsorted(
            cumulative, random_values, side="right"
        )

    return new_states, int(np.count_nonzero(new_states != active_states))


def state_histogram(active_states: np.ndarray, nstates: int) -> np.ndarray:
    return np.bincount(active_states, minlength=nstates).astype(float) / len(active_states)


def main() -> None:
    args = parse_args()

    if args.nsh <= 0:
        raise ValueError("--nsh must be positive")
    if args.substeps <= 0:
        raise ValueError("--substeps must be positive")

    args.outdir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input, allow_pickle=False) as data:
        time_fs_all = np.asarray(data["time_fs"], dtype=float)
        energies_all = np.asarray(data["active_energy_hartree"], dtype=float)
        projectors_all = np.asarray(data["projector_c60"], dtype=complex)
        hvib_all = np.asarray(data["hvib_mid_hartree"], dtype=complex)
        c0 = np.asarray(data["initial_c0"], dtype=complex)
        initial_capture = float(data["initial_capture"])

    available_frames = len(time_fs_all)
    nframes = min(args.nframes, available_frames)
    if nframes < 2:
        raise ValueError("Need at least two frames")

    time_fs = time_fs_all[:nframes]
    energies = energies_all[:nframes]
    projectors = projectors_all[:nframes]
    hvib_mid = hvib_all[: nframes - 1]

    nstates = c0.shape[0]
    expected_shapes = {
        "energies": (nframes, nstates),
        "projectors": (nframes, nstates, nstates),
        "hvib_mid": (nframes - 1, nstates, nstates),
    }
    actual_shapes = {
        "energies": energies.shape,
        "projectors": projectors.shape,
        "hvib_mid": hvib_mid.shape,
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise RuntimeError(
                f"{name} shape {actual_shapes[name]} does not match {expected}"
            )

    c0_norm = float(np.vdot(c0, c0).real)
    if abs(c0_norm - 1.0) > 1.0e-10:
        raise RuntimeError(f"Initial coefficient norm is {c0_norm}, not 1")
    if np.max(np.abs(hvib_mid - np.swapaxes(hvib_mid.conj(), 1, 2))) > 1.0e-10:
        raise RuntimeError("Input Hvib is not Hermitian")

    rng = np.random.default_rng(args.seed)
    initial_state_probabilities = np.abs(c0) ** 2
    initial_state_probabilities /= initial_state_probabilities.sum()

    if args.initial_surfaces == "population":
        active_states = rng.choice(
            nstates, size=args.nsh, p=initial_state_probabilities
        ).astype(np.int16)
    else:
        active_states = np.full(
            args.nsh, int(np.argmax(initial_state_probabilities)), dtype=np.int16
        )

    coefficients = np.empty((nframes, nstates), dtype=complex)
    se_state_pop = np.empty((nframes, nstates), dtype=float)
    sh_state_pop = np.empty((nframes, nstates), dtype=float)
    se_c60 = np.empty(nframes, dtype=float)
    sh_c60_diagonal = np.empty(nframes, dtype=float)
    norm = np.empty(nframes, dtype=float)
    interval_hops = np.zeros(nframes - 1, dtype=np.int64)
    interval_max_raw_probability_sum = np.zeros(nframes - 1, dtype=float)
    interval_renormalized_vectors = np.zeros(nframes - 1, dtype=np.int64)

    c_np = c0.copy()

    coefficients[0] = c_np
    se_state_pop[0] = np.abs(c_np) ** 2
    sh_state_pop[0] = state_histogram(active_states, nstates)
    se_c60[0] = physical_fragment_population(c_np, projectors[0])
    sh_c60_diagonal[0] = float(
        np.mean(np.real(np.diag(projectors[0]))[active_states])
    )
    norm[0] = float(np.vdot(c_np, c_np).real)

    print(
        f"Frames={nframes}, states={nstates}, FSSH walkers={args.nsh}, "
        f"substeps/interval={args.substeps}, seed={args.seed}, "
        f"Libra FSSH API={LIBRA_FSSH_API}",
        flush=True,
    )
    print(
        f"Initial active-space capture={initial_capture:.10f}; "
        f"physical C60 population={se_c60[0]:.8f}; "
        f"active-surface diagonal diagnostic={sh_c60_diagonal[0]:.8f}",
        flush=True,
    )

    global_max_raw_probability_sum = 0.0
    total_renormalized_vectors = 0
    total_hops = 0

    for interval in range(nframes - 1):
        dt_fs = time_fs[interval + 1] - time_fs[interval]
        if dt_fs <= 0.0:
            raise RuntimeError(f"Non-positive time step at interval {interval}")

        subdt_au = dt_fs * FS_TO_AU / args.substeps
        h_np = hvib_mid[interval].copy()

        # Remove a scalar energy shift. This changes only a global phase.
        scalar_shift = np.trace(h_np).real / nstates
        h_np -= scalar_shift * np.eye(nstates)
        h_libra = array_to_cmatrix(h_np)
        u_half = expm(-0.5j * subdt_au * h_np)

        interval_raw_probability_sum = 0.0
        interval_renormalized = 0
        interval_hop_count = 0
        for _ in range(args.substeps):
            # Exact half-step propagation for the piecewise-constant Hvib.
            # The current Libra build exposes only the high-level
            # dyn_variables/nHamiltonian propagate_electronic signature,
            # so the small 10x10 TDSE step is evaluated directly.
            c_np = u_half @ c_np
            coeff_mid = array_to_cmatrix(c_np)

            probabilities, raw_sum, nrenorm = (
                libra_probability_matrix_for_active_states(
                    coeff_mid,
                    h_libra,
                    active_states,
                    subdt_au,
                    args.temperature,
                    args.boltzmann,
                    nstates,
                    args.probability_tolerance,
                )
            )
            interval_raw_probability_sum = max(
                interval_raw_probability_sum, raw_sum
            )
            global_max_raw_probability_sum = max(
                global_max_raw_probability_sum, raw_sum
            )
            interval_renormalized += nrenorm
            total_renormalized_vectors += nrenorm

            active_states, hop_count = sample_hops(
                active_states, probabilities, rng
            )
            interval_hop_count += hop_count
            total_hops += hop_count

            c_np = u_half @ c_np

        coefficients[interval + 1] = c_np
        se_state_pop[interval + 1] = np.abs(c_np) ** 2
        sh_state_pop[interval + 1] = state_histogram(active_states, nstates)
        se_c60[interval + 1] = physical_fragment_population(
            c_np, projectors[interval + 1]
        )
        sh_c60_diagonal[interval + 1] = float(
            np.mean(
                np.real(np.diag(projectors[interval + 1]))[active_states]
            )
        )
        norm[interval + 1] = float(np.vdot(c_np, c_np).real)
        interval_hops[interval] = interval_hop_count
        interval_max_raw_probability_sum[interval] = (
            interval_raw_probability_sum
        )
        interval_renormalized_vectors[interval] = interval_renormalized

        if (
            interval == 0
            or interval == nframes - 2
            or (interval + 1) % max(1, (nframes - 1) // 10) == 0
        ):
            print(
                f"frame {interval+1:04d}/{nframes-1:04d}: "
                f"P_C60(SE)={se_c60[interval+1]:.6f}, "
                f"P_C60(active diag)={sh_c60_diagonal[interval+1]:.6f}, "
                f"norm={norm[interval+1]:.12f}, "
                f"max raw prob sum={interval_raw_probability_sum:.4f}, "
                f"renorm vectors={interval_renormalized}, "
                f"hops={interval_hop_count}",
                flush=True,
            )

    max_norm_error = float(np.max(np.abs(norm - 1.0)))

    np.savez_compressed(
        args.outdir / "libra_fssh.npz",
        time_fs=time_fs,
        coefficients=coefficients,
        se_state_population=se_state_pop,
        sh_state_population=sh_state_pop,
        se_c60_population=se_c60,
        active_surface_c60_diagonal=sh_c60_diagonal,
        norm=norm,
        interval_hops=interval_hops,
        interval_max_raw_probability_sum=(
            interval_max_raw_probability_sum
        ),
        interval_renormalized_probability_vectors=(
            interval_renormalized_vectors
        ),
        initial_surface_probabilities=initial_state_probabilities,
        final_active_states=active_states,
        initial_capture=initial_capture,
    )

    csv_path = args.outdir / "populations.csv"
    fieldnames = [
        "time_fs",
        "se_c60_population",
        "active_surface_c60_diagonal",
        "norm",
        "interval_hops_before_frame",
        "interval_max_raw_probability_sum",
        "interval_renormalized_probability_vectors",
    ]
    fieldnames += [f"se_state_{i}" for i in range(nstates)]
    fieldnames += [f"sh_state_{i}" for i in range(nstates)]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for iframe in range(nframes):
            row = {
                "time_fs": time_fs[iframe],
                "se_c60_population": se_c60[iframe],
                "active_surface_c60_diagonal": sh_c60_diagonal[iframe],
                "norm": norm[iframe],
                "interval_hops_before_frame": (
                    0 if iframe == 0 else int(interval_hops[iframe - 1])
                ),
                "interval_max_raw_probability_sum": (
                    0.0
                    if iframe == 0
                    else float(interval_max_raw_probability_sum[iframe - 1])
                ),
                "interval_renormalized_probability_vectors": (
                    0
                    if iframe == 0
                    else int(interval_renormalized_vectors[iframe - 1])
                ),
            }
            row.update(
                {f"se_state_{i}": se_state_pop[iframe, i] for i in range(nstates)}
            )
            row.update(
                {f"sh_state_{i}": sh_state_pop[iframe, i] for i in range(nstates)}
            )
            writer.writerow(row)

    summary = {
        "method": "Classical-path FSSH with exact 10-state TDSE propagation and Libra hopping probabilities",
        "input": str(args.input),
        "nframes": nframes,
        "nuclear_dt_fs": float(np.mean(np.diff(time_fs))),
        "electronic_probability_substeps": args.substeps,
        "fssh_walkers": args.nsh,
        "seed": args.seed,
        "temperature_K": args.temperature,
        "boltzmann_scaled_upward_hops": args.boltzmann,
        "initial_surface_scheme": args.initial_surfaces,
        "initial_active_capture": initial_capture,
        "initial_physical_c60_population": float(se_c60[0]),
        "final_physical_c60_population": float(se_c60[-1]),
        "maximum_physical_c60_population": float(np.max(se_c60)),
        "initial_active_surface_diagonal_c60": float(sh_c60_diagonal[0]),
        "final_active_surface_diagonal_c60": float(sh_c60_diagonal[-1]),
        "maximum_norm_error": max_norm_error,
        "maximum_raw_probability_vector_sum": float(
            global_max_raw_probability_sum
        ),
        "renormalized_probability_vector_count": int(
            total_renormalized_vectors
        ),
        "total_accepted_surface_changes": int(total_hops),
        "probability_note": (
            "FSSH probabilities are computed only for occupied active states "
            "using Libra's active-state-specific overload. Returned vectors "
            "are normalized exactly as Libra's hop(...) routine normalizes "
            "them before stochastic state selection."
        ),
        "important_note": (
            "The primary fragment observable is se_c60_population = "
            "c^dagger P_C60 c. The active-surface diagonal quantity omits "
            "coherences and is only a diagnostic for this coherent donor initial state."
        ),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\nCompleted Libra CPA-FSSH.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()