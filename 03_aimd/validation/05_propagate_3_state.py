#!/usr/bin/env python3
"""Coherent propagation through tracked states 1, 2, and 3."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


FS_TO_AU = 41.3413745758

TRACKED_FILE = Path("output_tracking_test/tracked_states.csv")
NAC_FILE = Path("output_tracking_test/nac_summary.csv")
OUTPUT_FILE = Path("output_tracking_test/three_state_populations.csv")
PLOT_FILE = Path("output_tracking_test/three_state_populations.png")

STATES = (1, 2, 3)


def read_electronic_states():
    energies = {}
    times = {}

    with TRACKED_FILE.open() as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            state = int(row["tracked_state"])

            times[frame] = float(row["time_fs"])

            if state in STATES:
                energies[(frame, state)] = float(row["energy_hartree"])

    return energies, times


def main():
    energies, times = read_electronic_states()

    with NAC_FILE.open() as handle:
        intervals = list(csv.DictReader(handle))

    first_frame = int(intervals[0]["frame_i"])

    # Local ordering: [tracked state 1, tracked state 2, tracked state 3].
    # Begin entirely in the predominantly donor-like tracked state 3.
    coefficients = np.array([0.0, 0.0, 1.0], dtype=complex)

    history = [
        {
            "frame": first_frame,
            "time_fs": times[first_frame],
            "population_state_1": 0.0,
            "population_state_2": 0.0,
            "population_state_3": 1.0,
            "acceptor_population": 0.0,
            "norm": 1.0,
        }
    ]

    for row in intervals:
        frame_i = int(row["frame_i"])
        frame_j = int(row["frame_j"])

        dt_fs = times[frame_j] - times[frame_i]
        dt_au = dt_fs * FS_TO_AU

        # Midpoint adiabatic energies.
        midpoint_energies = np.array(
            [
                0.5 * (
                    energies[(frame_i, state)]
                    + energies[(frame_j, state)]
                )
                for state in STATES
            ]
        )

        # Remove an irrelevant common energy to avoid accumulating
        # a very large global phase.
        midpoint_energies -= midpoint_energies.mean()

        d12 = float(row["nac_12_au"])
        d13 = float(row["nac_13_au"])
        d23 = float(row["nac_23_au"])

        # Real antisymmetric time-derivative coupling matrix.
        derivative_coupling = np.array(
            [
                [0.0,  d12,  d13],
                [-d12, 0.0,  d23],
                [-d13, -d23, 0.0],
            ]
        )

        # Vibronic Hamiltonian in the adiabatic representation.
        #
        # i dc/dt = (E - iD)c
        vibronic_hamiltonian = (
            np.diag(midpoint_energies)
            - 1j * derivative_coupling
        )

        hermiticity_error = np.linalg.norm(
            vibronic_hamiltonian
            - vibronic_hamiltonian.conj().T
        )

        if hermiticity_error > 1.0e-10:
            raise RuntimeError(
                f"Non-Hermitian Hamiltonian at {frame_i}-{frame_j}: "
                f"{hermiticity_error:.3e}"
            )

        propagator = expm(-1j * vibronic_hamiltonian * dt_au)
        coefficients = propagator @ coefficients

        populations = np.abs(coefficients) ** 2
        norm = populations.sum()

        history.append(
            {
                "frame": frame_j,
                "time_fs": times[frame_j],
                "population_state_1": populations[0],
                "population_state_2": populations[1],
                "population_state_3": populations[2],
                "acceptor_population": populations[0] + populations[1],
                "norm": norm,
            }
        )

    with OUTPUT_FILE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    time = np.array([row["time_fs"] for row in history])
    p1 = np.array([row["population_state_1"] for row in history])
    p2 = np.array([row["population_state_2"] for row in history])
    p3 = np.array([row["population_state_3"] for row in history])
    p_acceptor = p1 + p2
    norms = np.array([row["norm"] for row in history])

    plt.figure(figsize=(7, 5))
    plt.plot(time, p1, marker="o", label="State 1")
    plt.plot(time, p2, marker="o", label="State 2")
    plt.plot(time, p3, marker="o", label="State 3: donor-like")
    plt.plot(
        time,
        p_acceptor,
        marker="s",
        linestyle="--",
        label="Acceptor total: states 1 + 2",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel("Electronic population")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)

    print(f"Final state-1 population: {p1[-1]:.8f}")
    print(f"Final state-2 population: {p2[-1]:.8f}")
    print(f"Final state-3 population: {p3[-1]:.8f}")
    print(f"Final acceptor population: {p_acceptor[-1]:.8f}")
    print(f"Maximum norm error: {np.max(np.abs(norms - 1.0)):.3e}")
    print(f"Wrote {OUTPUT_FILE}")
    print(f"Wrote {PLOT_FILE}")


if __name__ == "__main__":
    main()
