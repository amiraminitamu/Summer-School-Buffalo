#!/usr/bin/env python3
"""Convert tracked overlaps into time-derivative coupling matrices."""

from pathlib import Path
import csv
import re

import numpy as np
from scipy.linalg import logm


DT_FS = 0.5
FS_TO_AU = 41.3413745758
HARTREE_TO_MEV = 27211.386245988

TRACKED_CSV = Path("output_tracking_test/tracked_states.csv")
OVERLAP_DIR = Path("output_tracking_test/overlaps")
OUTPUT_CSV = Path("output_tracking_test/nac_summary.csv")


def load_energies():
    energies = {}

    with TRACKED_CSV.open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["frame"]), int(row["tracked_state"]))
            energies[key] = float(row["energy_hartree"])

    return energies


def closest_orthogonal(matrix):
    """Polar/SVD projection onto the nearest orthogonal matrix."""
    left, _, right_t = np.linalg.svd(matrix)
    unitary = left @ right_t

    if np.linalg.det(unitary) < 0:
        raise RuntimeError(
            "Nearest overlap matrix has determinant < 0. "
            "Check orbital phases or permutations."
        )

    return unitary


def main():
    energies = load_energies()
    rows = []

    paths = sorted(OVERLAP_DIR.glob("overlap_*_*_tracked.csv"))

    for path in paths:
        match = re.search(r"overlap_(\d+)_(\d+)_tracked", path.stem)
        if not match:
            continue

        frame_i = int(match.group(1))
        frame_j = int(match.group(2))

        overlap = np.loadtxt(path, delimiter=",", skiprows=1)

        # Remove the small nonunitarity caused by truncating to 10 states.
        unitary_overlap = closest_orthogonal(overlap)

        # Time-derivative coupling D = log(U) / dt.
        dt_au = DT_FS * FS_TO_AU
        nac = logm(unitary_overlap) / dt_au
        nac = np.real_if_close(nac, tol=1000).real

        # Enforce the required antisymmetry numerically.
        nac = 0.5 * (nac - nac.T)

        row = {
            "frame_i": frame_i,
            "frame_j": frame_j,
            "time_mid_fs": 0.5 * (frame_i + frame_j) * DT_FS,
        }

        for state_a, state_b in ((1, 2), (1, 3), (2, 3)):
            ea = 0.5 * (
                energies[(frame_i, state_a)]
                + energies[(frame_j, state_a)]
            )
            eb = 0.5 * (
                energies[(frame_i, state_b)]
                + energies[(frame_j, state_b)]
            )

            coupling_au = nac[state_a, state_b]

            row[f"gap_{state_a}{state_b}_meV"] = (
                abs(eb - ea) * HARTREE_TO_MEV
            )
            row[f"nac_{state_a}{state_b}_au"] = coupling_au
            row[f"hvib_{state_a}{state_b}_meV"] = (
                abs(coupling_au) * HARTREE_TO_MEV
            )

        rows.append(row)

        print(
            f"{frame_i:03d}-{frame_j:03d}: "
            f"|hbar*d12|={row['hvib_12_meV']:.3f} meV, "
            f"|hbar*d13|={row['hvib_13_meV']:.3f} meV, "
            f"|hbar*d23|={row['hvib_23_meV']:.3f} meV"
        )

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
