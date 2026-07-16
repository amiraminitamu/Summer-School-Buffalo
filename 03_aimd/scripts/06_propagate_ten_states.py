#!/usr/bin/env python3
"""
Propagate the full tracked LUMO...LUMO+9 active space from 95.0 to 99.5 fs.

The initial state is the isolated-donor LUMO projected into the active
space at frame 190. C60 populations are evaluated using the complete
Hermitian symmetrized-Mulliken population matrix, including coherences.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto
from scipy.linalg import expm, logm


FS_TO_AU = 41.3413745758
NSTATES = 10

TRACKING_ROOT = Path("output_tracking_test")
FRONTIER_FILE = Path(
    "output_frontier_test/frame190/frontier_data.npz"
)
XYZ_ROOT = Path("tracking_test")

OUTPUT_CSV = TRACKING_ROOT / "ten_state_populations.csv"
OUTPUT_PLOT = TRACKING_ROOT / "ten_state_populations.png"


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    natoms = int(lines[0])

    atoms = []
    for line in lines[2:2 + natoms]:
        fields = line.split()
        atoms.append(
            (
                fields[0],
                (
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                ),
            )
        )

    if len(atoms) != natoms:
        raise ValueError(
            f"Expected {natoms} atoms in {path}, found {len(atoms)}"
        )

    return atoms


def xyz_for_frame(frame: int):
    matches = sorted(
        XYZ_ROOT.glob(f"frame_{frame:03d}_t*.xyz")
    )

    if len(matches) != 1:
        raise ValueError(
            f"Expected one XYZ file for frame {frame}, found {matches}"
        )

    return matches[0]


def ao_mask_for_atoms(mol, atom_start: int, atom_stop: int):
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    slices = mol.aoslice_by_atom()

    for atom in range(atom_start, atom_stop):
        ao_start, ao_stop = slices[atom, 2:4]
        mask[ao_start:ao_stop] = True

    return mask


def fragment_population_matrix(mol, coefficients, atom_start, atom_stop):
    """
    Hermitian symmetrized-Mulliken population matrix in the MO subspace.

    P_A = 1/2 [C^† M_A S C + C^† S M_A C]

    Its diagonal elements reproduce the Mulliken-like orbital weights
    used in the earlier scripts.
    """
    overlap = mol.intor_symmetric("int1e_ovlp")
    overlap_times_coeff = overlap @ coefficients

    mask = ao_mask_for_atoms(mol, atom_start, atom_stop)

    gross_matrix = (
        coefficients[mask].conj().T
        @ overlap_times_coeff[mask]
    )

    population_matrix = 0.5 * (
        gross_matrix + gross_matrix.conj().T
    )

    return population_matrix, overlap


def closest_orthogonal(matrix):
    left, _, right_t = np.linalg.svd(matrix)
    orthogonal = left @ right_t

    determinant = np.linalg.det(orthogonal)
    if determinant < 0:
        raise RuntimeError(
            "Tracked overlap has negative determinant. "
            "Check state phases and permutations."
        )

    return orthogonal


def load_tracked_energies():
    energies = {}
    times = {}

    path = TRACKING_ROOT / "tracked_states.csv"

    with path.open() as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            state = int(row["tracked_state"])

            times[frame] = float(row["time_fs"])
            energies[(frame, state)] = float(
                row["energy_hartree"]
            )

    return energies, times


def load_tracked_coefficients(frame):
    path = TRACKING_ROOT / "frames" / f"frame_{frame:03d}.npz"

    with np.load(path) as data:
        coefficients = data["tracked_coefficients"].copy()

    if coefficients.shape[1] != NSTATES:
        raise ValueError(
            f"Expected {NSTATES} states in {path}, "
            f"found shape {coefficients.shape}"
        )

    return coefficients


def build_population_operators(frame):
    xyz_path = xyz_for_frame(frame)
    atoms = read_xyz(xyz_path)

    mol = gto.M(
        atom=atoms,
        basis="6-31g",
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry=False,
        verbose=0,
    )

    coefficients = load_tracked_coefficients(frame)

    # Atoms 0:116 are the two H2Pc molecules.
    donor_matrix, overlap = fragment_population_matrix(
        mol,
        coefficients,
        0,
        116,
    )

    # Atoms 116:176 are C60.
    c60_matrix, _ = fragment_population_matrix(
        mol,
        coefficients,
        116,
        176,
    )

    mo_orthogonality_error = np.linalg.norm(
        coefficients.conj().T @ overlap @ coefficients
        - np.eye(NSTATES)
    )

    population_sum_error = np.linalg.norm(
        donor_matrix + c60_matrix - np.eye(NSTATES)
    )

    return (
        c60_matrix,
        donor_matrix,
        overlap,
        coefficients,
        mo_orthogonality_error,
        population_sum_error,
    )


def prepare_initial_state(overlap, tracked_coefficients):
    """
    Map the projected donor LUMO from the independently calculated
    frontier basis into the frame-190 tracked basis.
    """
    with np.load(FRONTIER_FILE) as data:
        active_indices = data["active_indices"].astype(int)
        frontier_coefficients = data["active_mo_coeff"].copy()
        full_projection = data["c0"].copy()

    frontier_amplitudes = full_projection[active_indices].astype(complex)

    expected_active_norm = np.vdot(
        frontier_amplitudes,
        frontier_amplitudes,
    ).real

    # AO representation of the donor state projected into the
    # frontier active space.
    donor_ao = frontier_coefficients @ frontier_amplitudes

    # Map that AO wavefunction into the tracked active-space basis.
    tracked_amplitudes = (
        tracked_coefficients.conj().T
        @ overlap
        @ donor_ao
    )

    mapped_norm = np.vdot(
        tracked_amplitudes,
        tracked_amplitudes,
    ).real

    # Conditional normalization within the selected ten-state space.
    tracked_amplitudes /= np.sqrt(mapped_norm)

    basis_overlap = (
        tracked_coefficients.conj().T
        @ overlap
        @ frontier_coefficients
    )

    singular_values = np.linalg.svd(
        basis_overlap,
        compute_uv=False,
    )

    print("Initial-state preparation")
    print("-------------------------")
    print(
        f"Frontier active-space norm: {expected_active_norm:.10f}"
    )
    print(
        f"Norm after mapping to tracked basis: {mapped_norm:.10f}"
    )
    print(
        "Tracked/frontier subspace singular values: "
        f"{singular_values.min():.10f} to "
        f"{singular_values.max():.10f}"
    )

    return tracked_amplitudes


def expectation(coefficients, operator):
    value = np.vdot(
        coefficients,
        operator @ coefficients,
    )

    return float(np.real_if_close(value).real)


def main():
    energies, times = load_tracked_energies()
    frames = sorted(times)

    if frames != list(range(190, 200)):
        raise ValueError(f"Unexpected frames: {frames}")

    operators = {}

    print("Building fragment-population matrices")

    for frame in frames:
        (
            c60_matrix,
            donor_matrix,
            overlap,
            tracked_coefficients,
            orth_error,
            sum_error,
        ) = build_population_operators(frame)

        operators[frame] = {
            "c60": c60_matrix,
            "donor": donor_matrix,
            "overlap": overlap,
            "coefficients": tracked_coefficients,
        }

        print(
            f"Frame {frame}: "
            f"MO orthogonality error={orth_error:.3e}, "
            f"fragment sum error={sum_error:.3e}"
        )

    first_frame = frames[0]

    coefficients = prepare_initial_state(
        operators[first_frame]["overlap"],
        operators[first_frame]["coefficients"],
    )

    history = []

    def record(frame):
        populations = np.abs(coefficients) ** 2
        c60_matrix = operators[frame]["c60"]

        c60_coherent = expectation(
            coefficients,
            c60_matrix,
        )

        # What one would obtain by retaining only diagonal populations.
        c60_diagonal = float(
            np.sum(
                populations
                * np.real(np.diag(c60_matrix))
            )
        )

        donor_population = expectation(
            coefficients,
            operators[frame]["donor"],
        )

        row = {
            "frame": frame,
            "time_fs": times[frame],
            "C60_population_coherent": c60_coherent,
            "C60_population_diagonal_only": c60_diagonal,
            "C60_coherence_contribution":
                c60_coherent - c60_diagonal,
            "donor_population": donor_population,
            "fragment_population_sum":
                donor_population + c60_coherent,
            "wavefunction_norm":
                float(np.vdot(coefficients, coefficients).real),
        }

        for state in range(NSTATES):
            row[f"population_state_{state}"] = populations[state]

        history.append(row)

    record(first_frame)

    for frame_i, frame_j in zip(frames[:-1], frames[1:]):
        overlap_path = (
            TRACKING_ROOT
            / "overlaps"
            / f"overlap_{frame_i:03d}_{frame_j:03d}_tracked.csv"
        )

        tracked_overlap = np.loadtxt(
            overlap_path,
            delimiter=",",
            skiprows=1,
        )

        unitary_overlap = closest_orthogonal(tracked_overlap)

        dt_fs = times[frame_j] - times[frame_i]
        dt_au = dt_fs * FS_TO_AU

        derivative_coupling = logm(unitary_overlap) / dt_au

        imaginary_error = np.max(
            np.abs(np.imag(derivative_coupling))
        )

        if imaginary_error > 1.0e-8:
            raise RuntimeError(
                f"Large imaginary component in derivative coupling "
                f"for {frame_i}-{frame_j}: {imaginary_error:.3e}"
            )

        derivative_coupling = np.real(
            derivative_coupling
        )

        # Enforce antisymmetry against numerical noise.
        derivative_coupling = 0.5 * (
            derivative_coupling
            - derivative_coupling.T
        )

        midpoint_energies = np.array(
            [
                0.5 * (
                    energies[(frame_i, state)]
                    + energies[(frame_j, state)]
                )
                for state in range(NSTATES)
            ]
        )

        # Remove the physically irrelevant common energy.
        midpoint_energies -= midpoint_energies.mean()

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
                f"Non-Hermitian Hamiltonian for "
                f"{frame_i}-{frame_j}: {hermiticity_error:.3e}"
            )

        propagator = expm(
            -1j * vibronic_hamiltonian * dt_au
        )

        coefficients = propagator @ coefficients
        record(frame_j)

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    time = np.array([row["time_fs"] for row in history])
    c60 = np.array(
        [row["C60_population_coherent"] for row in history]
    )
    c60_diagonal = np.array(
        [row["C60_population_diagonal_only"] for row in history]
    )
    coherence = np.array(
        [row["C60_coherence_contribution"] for row in history]
    )
    norms = np.array(
        [row["wavefunction_norm"] for row in history]
    )
    fragment_sums = np.array(
        [row["fragment_population_sum"] for row in history]
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        time,
        c60,
        marker="o",
        label="C60 population: full operator",
    )
    plt.plot(
        time,
        c60_diagonal,
        marker="s",
        linestyle="--",
        label="Diagonal-only estimate",
    )
    plt.plot(
        time,
        coherence,
        marker="^",
        linestyle=":",
        label="Coherence contribution",
    )
    plt.xlabel("Time (fs)")
    plt.ylabel("Population")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)

    print("\nPropagation results")
    print("-------------------")
    print(f"Initial C60 population: {c60[0]:.8f}")
    print(f"Maximum C60 population: {c60.max():.8f}")
    print(f"Final C60 population:   {c60[-1]:.8f}")
    print(
        "Maximum wavefunction norm error: "
        f"{np.max(np.abs(norms - 1.0)):.3e}"
    )
    print(
        "Maximum donor+C60 sum error: "
        f"{np.max(np.abs(fragment_sums - 1.0)):.3e}"
    )
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
