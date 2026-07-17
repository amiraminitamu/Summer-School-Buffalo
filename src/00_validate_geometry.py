#!/usr/bin/env python3
"""Validate the published geometry and fragment ordering."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
XYZ = ROOT / "data" / "2H2Pc_C60.xyz"


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    natoms = int(lines[0])
    rows = [line.split() for line in lines[2 : 2 + natoms]]
    if len(rows) != natoms:
        raise ValueError(f"{path}: expected {natoms} atoms, found {len(rows)}")
    elements = [row[0] for row in rows]
    xyz = np.array([[float(value) for value in row[1:4]] for row in rows])
    return elements, xyz


def formula(elements: list[str]) -> str:
    counts = Counter(elements)
    return "".join(f"{element}{counts[element]}" for element in ("C", "H", "N") if counts[element])


def closest_pair(xyz: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, int, int]:
    distance = np.linalg.norm(xyz[first][:, None, :] - xyz[second][None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(distance), distance.shape)
    return float(distance[i, j]), int(first[i]), int(second[j])


def main() -> None:
    elements, xyz = read_xyz(XYZ)
    assert len(elements) == 176
    assert formula(elements[:58]) == "C32H18N8"
    assert formula(elements[58:116]) == "C32H18N8"
    assert formula(elements[116:]) == "C60"

    pc_far = np.arange(0, 58)
    pc_near = np.arange(58, 116)
    c60 = np.arange(116, 176)

    print("Geometry validated")
    print("  Full system: 176 atoms, formula C124H36N16")
    print("  Atoms 1-58: H2Pc farther from C60")
    print("  Atoms 59-116: H2Pc in direct contact with C60")
    print("  Atoms 117-176: C60")

    for label, first, second in (
        ("Pc(far)-Pc(near)", pc_far, pc_near),
        ("Pc(near)-C60", pc_near, c60),
    ):
        distance, i, j = closest_pair(xyz, first, second)
        print(f"  Closest {label} contact: {distance:.4f} A (atoms {i + 1} and {j + 1})")


if __name__ == "__main__":
    main()
