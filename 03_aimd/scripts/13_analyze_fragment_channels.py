#!/usr/bin/env python3
"""
Diagnose the dimensionality of the fragment-reduced electronic model.

For every tracked frame:
  * split the ten-state space into 4 donor and 6 C60 directions using the
    C60-projector eigenvalues;
  * build the donor block H_DD, acceptor block H_AA, and coupling block V_DA;
  * compute the singular values of V_DA (basis-invariant coupling channels);
  * diagonalize H_DD and H_AA to inspect fragment-localized energy manifolds.

This script does not yet build the final TENSO Hamiltonian. It determines
whether one, two, or more donor/acceptor channels are required.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("output_tracked"))
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("output_reduced_channels"),
    )
    p.add_argument("--ntraj", type=int, default=10)
    p.add_argument("--projector-cutoff", type=float, default=0.5)
    return p.parse_args()


def hermitize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def fragment_blocks(
    energies_h: np.ndarray,
    projector_c60: np.ndarray,
    cutoff: float,
):
    p = hermitize(projector_c60)
    pvals, pvecs = np.linalg.eigh(p)

    donor_mask = pvals < cutoff
    acceptor_mask = ~donor_mask

    if donor_mask.sum() != 4 or acceptor_mask.sum() != 6:
        raise RuntimeError(
            f"Expected 4 donor and 6 acceptor directions; got "
            f"{donor_mask.sum()} and {acceptor_mask.sum()}. "
            f"Projector eigenvalues: {pvals}"
        )

    u_d = pvecs[:, donor_mask]
    u_a = pvecs[:, acceptor_mask]
    h = np.diag(np.asarray(energies_h, dtype=complex))

    h_dd = hermitize(u_d.conj().T @ h @ u_d)
    h_aa = hermitize(u_a.conj().T @ h @ u_a)
    v_da = u_d.conj().T @ h @ u_a

    e_d, x_d = np.linalg.eigh(h_dd)
    e_a, x_a = np.linalg.eigh(h_aa)
    v_frag_eig = x_d.conj().T @ v_da @ x_a
    singular_values = np.linalg.svd(v_da, compute_uv=False)

    return {
        "projector_eigenvalues": pvals,
        "u_d": u_d,
        "u_a": u_a,
        "donor_energies": e_d,
        "acceptor_energies": e_a,
        "coupling_block": v_da,
        "coupling_fragment_eigenbasis": v_frag_eig,
        "singular_values": singular_values,
        "x_d": x_d,
        "x_a": x_a,
    }


def safe_channel_fractions(singular_values: np.ndarray) -> np.ndarray:
    weights = singular_values**2
    total = np.sum(weights, axis=-1, keepdims=True)
    return np.divide(
        weights,
        total,
        out=np.zeros_like(weights),
        where=total > 0.0,
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    trajectory_rows = []

    print(
        f"{'traj':>5} {'<s1>/eV':>10} {'<s2>/eV':>10} "
        f"{'<s3>/eV':>10} {'<s4>/eV':>10} "
        f"{'<F1>':>8} {'<F12>':>8} {'<F123>':>8} "
        f"{'<A3-A2>/eV':>13} {'<A4-A3>/eV':>13} "
        f"{'<D span>/eV':>12} {'c0 max wt':>10}"
    )
    print("-" * 146)

    for traj in range(args.ntraj):
        path = args.root / f"traj_{traj:02d}" / "tracked_data.npz"
        with np.load(path, allow_pickle=False) as d:
            energies = np.asarray(d["active_energy_hartree"], dtype=float)
            projectors = np.asarray(d["projector_c60"], dtype=complex)
            c0 = np.asarray(d["initial_c0"], dtype=complex)

        nframes = energies.shape[0]
        singular_values = np.zeros((nframes, 4), dtype=float)
        donor_energies = np.zeros((nframes, 4), dtype=float)
        acceptor_energies = np.zeros((nframes, 6), dtype=float)
        low_acceptor_coupling_norm = np.zeros(nframes, dtype=float)
        high_acceptor_coupling_norm = np.zeros(nframes, dtype=float)

        initial_donor_weights = None

        frame_csv = args.outdir / f"traj_{traj:02d}_channels.csv"
        with frame_csv.open("w", newline="") as handle:
            fieldnames = [
                "frame",
                "s1_eV",
                "s2_eV",
                "s3_eV",
                "s4_eV",
                "frac1",
                "frac12",
                "frac123",
                "donor_span_eV",
                "acceptor_low_span_eV",
                "acceptor_high_span_eV",
                "acceptor_middle_gap_eV",
                "low_acceptor_coupling_norm_eV",
                "high_acceptor_coupling_norm_eV",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for frame in range(nframes):
                blocks = fragment_blocks(
                    energies[frame],
                    projectors[frame],
                    args.projector_cutoff,
                )

                s = blocks["singular_values"]
                ed = blocks["donor_energies"]
                ea = blocks["acceptor_energies"]
                v_eig = blocks["coupling_fragment_eigenbasis"]

                singular_values[frame] = s
                donor_energies[frame] = ed
                acceptor_energies[frame] = ea

                # Coupling Frobenius norms into the lower and upper C60 triplets.
                low_acceptor_coupling_norm[frame] = np.linalg.norm(v_eig[:, :3])
                high_acceptor_coupling_norm[frame] = np.linalg.norm(v_eig[:, 3:])

                fractions = safe_channel_fractions(s[None, :])[0]
                cumulative = np.cumsum(fractions)

                writer.writerow(
                    {
                        "frame": frame,
                        "s1_eV": s[0] * HARTREE_TO_EV,
                        "s2_eV": s[1] * HARTREE_TO_EV,
                        "s3_eV": s[2] * HARTREE_TO_EV,
                        "s4_eV": s[3] * HARTREE_TO_EV,
                        "frac1": cumulative[0],
                        "frac12": cumulative[1],
                        "frac123": cumulative[2],
                        "donor_span_eV": (ed[-1] - ed[0]) * HARTREE_TO_EV,
                        "acceptor_low_span_eV": (
                            ea[2] - ea[0]
                        ) * HARTREE_TO_EV,
                        "acceptor_high_span_eV": (
                            ea[5] - ea[3]
                        ) * HARTREE_TO_EV,
                        "acceptor_middle_gap_eV": (
                            ea[3] - ea[2]
                        ) * HARTREE_TO_EV,
                        "low_acceptor_coupling_norm_eV": (
                            low_acceptor_coupling_norm[frame] * HARTREE_TO_EV
                        ),
                        "high_acceptor_coupling_norm_eV": (
                            high_acceptor_coupling_norm[frame] * HARTREE_TO_EV
                        ),
                    }
                )

                if frame == 0:
                    c_d = (
                        blocks["x_d"].conj().T
                        @ blocks["u_d"].conj().T
                        @ c0
                    )
                    initial_donor_weights = np.abs(c_d) ** 2
                    initial_donor_weights /= initial_donor_weights.sum()

        fractions = safe_channel_fractions(singular_values)
        cumulative = np.cumsum(fractions, axis=1)

        mean_s = np.mean(singular_values, axis=0) * HARTREE_TO_EV
        mean_frac1 = float(np.mean(cumulative[:, 0]))
        mean_frac12 = float(np.mean(cumulative[:, 1]))
        mean_frac123 = float(np.mean(cumulative[:, 2]))

        mean_a_low_gap = float(
            np.mean(acceptor_energies[:, 2] - acceptor_energies[:, 1])
            * HARTREE_TO_EV
        )
        mean_a_middle_gap = float(
            np.mean(acceptor_energies[:, 3] - acceptor_energies[:, 2])
            * HARTREE_TO_EV
        )
        mean_donor_span = float(
            np.mean(donor_energies[:, -1] - donor_energies[:, 0])
            * HARTREE_TO_EV
        )
        c0_max_weight = float(np.max(initial_donor_weights))

        row = {
            "trajectory": traj,
            "mean_s1_eV": mean_s[0],
            "mean_s2_eV": mean_s[1],
            "mean_s3_eV": mean_s[2],
            "mean_s4_eV": mean_s[3],
            "mean_fraction_first_channel": mean_frac1,
            "mean_fraction_first_two_channels": mean_frac12,
            "mean_fraction_first_three_channels": mean_frac123,
            "mean_acceptor_e2_minus_e1_eV": mean_a_low_gap,
            "mean_acceptor_e3_minus_e2_eV": mean_a_middle_gap,
            "mean_donor_span_eV": mean_donor_span,
            "mean_low_acceptor_coupling_norm_eV": float(
                np.mean(low_acceptor_coupling_norm) * HARTREE_TO_EV
            ),
            "mean_high_acceptor_coupling_norm_eV": float(
                np.mean(high_acceptor_coupling_norm) * HARTREE_TO_EV
            ),
            "initial_donor_weight_0": float(initial_donor_weights[0]),
            "initial_donor_weight_1": float(initial_donor_weights[1]),
            "initial_donor_weight_2": float(initial_donor_weights[2]),
            "initial_donor_weight_3": float(initial_donor_weights[3]),
            "initial_donor_max_weight": c0_max_weight,
        }
        trajectory_rows.append(row)

        print(
            f"{traj:5d} "
            f"{mean_s[0]:10.4f} {mean_s[1]:10.4f} "
            f"{mean_s[2]:10.4f} {mean_s[3]:10.4f} "
            f"{mean_frac1:8.4f} {mean_frac12:8.4f} "
            f"{mean_frac123:8.4f} "
            f"{mean_a_low_gap:13.4f} {mean_a_middle_gap:13.4f} "
            f"{mean_donor_span:12.4f} {c0_max_weight:10.4f}"
        )

    aggregate = args.outdir / "aggregate_channel_summary.csv"
    with aggregate.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(trajectory_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(trajectory_rows)

    print(f"\nWrote channel diagnostics to {args.outdir}")


if __name__ == "__main__":
    main()
