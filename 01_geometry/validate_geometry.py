#!/usr/bin/env python3
"""Validate the SI geometry and fragment ordering."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    rows = [line.split() for line in lines[2:2+n]]
    elements = [r[0] for r in rows]
    xyz = np.array([[float(v) for v in r[1:4]] for r in rows])
    return elements, xyz


def formula(elements):
    count = Counter(elements)
    return ''.join(f'{e}{count[e]}' for e in ('C', 'H', 'N') if count[e])


def closest_pair(xyz, a, b):
    d = np.linalg.norm(xyz[a][:, None, :] - xyz[b][None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(d[i, j]), int(a[i]), int(b[j])


elements, xyz = read_xyz(HERE / '2H2Pc_C60.xyz')
assert len(elements) == 176
assert formula(elements[:58]) == 'C32H18N8'
assert formula(elements[58:116]) == 'C32H18N8'
assert formula(elements[116:]) == 'C60'

pc1 = np.arange(0, 58)
pc2 = np.arange(58, 116)
c60 = np.arange(116, 176)

print('Geometry validated')
print(f'  Full system: {len(elements)} atoms, formula C124H36N16')
print('  Atoms 1-58: H2Pc farther from C60')
print('  Atoms 59-116: H2Pc in direct contact with C60')
print('  Atoms 117-176: C60')
for label, a, b in [('Pc(far)-Pc(near)', pc1, pc2), ('Pc(near)-C60', pc2, c60)]:
    dist, i, j = closest_pair(xyz, a, b)
    print(f'  Closest {label} contact: {dist:.4f} A (atoms {i+1} and {j+1})')
