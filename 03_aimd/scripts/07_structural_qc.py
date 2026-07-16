#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np


NATOMS = 176
BLOCK = NATOMS + 2

# Atom indices, Python convention:
# 0:58     H2Pc 1
# 58:116   H2Pc 2
# 116:176  C60
FRAGMENTS = {
    "H2Pc1": slice(0, 58),
    "H2Pc2": slice(58, 116),
    "C60": slice(116, 176),
}

MASSES = {
    "H": 1.00784,
    "C": 12.011,
    "N": 14.007,
}


def read_xyz_trajectory(path):
    lines = path.read_text().splitlines()
    nframes = len(lines) // BLOCK

    symbols = None
    frames = []

    for iframe in range(nframes):
        block = lines[iframe * BLOCK:(iframe + 1) * BLOCK]

        if len(block) != BLOCK or int(block[0]) != NATOMS:
            raise ValueError(f"Incomplete frame {iframe} in {path}")

        frame_symbols = []
        coordinates = []

        for line in block[2:]:
            fields = line.split()
            frame_symbols.append(fields[0])
            coordinates.append([float(x) for x in fields[1:4]])

        if symbols is None:
            symbols = frame_symbols
        elif frame_symbols != symbols:
            raise ValueError(f"Atom ordering changed in frame {iframe}")

        frames.append(coordinates)

    return symbols, np.asarray(frames)


def center_of_mass(coordinates, symbols, fragment):
    masses = np.array([MASSES[s] for s in symbols[fragment]])
    xyz = coordinates[fragment]

    return np.sum(xyz * masses[:, None], axis=0) / masses.sum()


def kabsch_rmsd(reference, mobile):
    ref = reference - reference.mean(axis=0)
    mob = mobile - mobile.mean(axis=0)

    covariance = mob.T @ ref
    u, _, vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))

    rotation = u @ correction @ vt
    aligned = mob @ rotation

    return np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1)))


def main():
    rows = []

    print(
        f"{'traj':>5} {'frames':>6} "
        f"{'Pc1-C60 min/max':>21} "
        f"{'Pc2-C60 min/max':>21} "
        f"{'Pc1-Pc2 min/max':>21} "
        f"{'RMSD final/max':>18}"
    )
    print("-" * 105)

    for replica in range(10):
        name = f"{replica:02d}"
        path = Path(
            f"output_production/traj_{name}/aimd.md.xyz"
        )

        symbols, frames = read_xyz_trajectory(path)
        reference = frames[0]

        d1 = []
        d2 = []
        d12 = []
        rmsd = []

        for coordinates in frames:
            com1 = center_of_mass(
                coordinates, symbols, FRAGMENTS["H2Pc1"]
            )
            com2 = center_of_mass(
                coordinates, symbols, FRAGMENTS["H2Pc2"]
            )
            com60 = center_of_mass(
                coordinates, symbols, FRAGMENTS["C60"]
            )

            d1.append(np.linalg.norm(com1 - com60))
            d2.append(np.linalg.norm(com2 - com60))
            d12.append(np.linalg.norm(com1 - com2))
            rmsd.append(kabsch_rmsd(reference, coordinates))

        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d12 = np.asarray(d12)
        rmsd = np.asarray(rmsd)

        rows.append({
            "trajectory": name,
            "frames": len(frames),
            "H2Pc1_C60_mean_A": d1.mean(),
            "H2Pc1_C60_min_A": d1.min(),
            "H2Pc1_C60_max_A": d1.max(),
            "H2Pc2_C60_mean_A": d2.mean(),
            "H2Pc2_C60_min_A": d2.min(),
            "H2Pc2_C60_max_A": d2.max(),
            "H2Pc1_H2Pc2_mean_A": d12.mean(),
            "H2Pc1_H2Pc2_min_A": d12.min(),
            "H2Pc1_H2Pc2_max_A": d12.max(),
            "final_RMSD_A": rmsd[-1],
            "maximum_RMSD_A": rmsd.max(),
        })

        print(
            f"{name:>5} {len(frames):6d} "
            f"{d1.min():8.3f}/{d1.max():8.3f} "
            f"{d2.min():8.3f}/{d2.max():8.3f} "
            f"{d12.min():8.3f}/{d12.max():8.3f} "
            f"{rmsd[-1]:8.3f}/{rmsd.max():8.3f}"
        )

    output = Path("output_production/structural_qc.csv")

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
