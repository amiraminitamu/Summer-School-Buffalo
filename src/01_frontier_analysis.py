#!/usr/bin/env python3
"""Analyze frontier orbitals and the donor-LUMO projection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from pyscf import dft, gto, lib
from pyscf.tools import molden

HARTREE_TO_EV = 27.211386245988
ROOT = Path(__file__).resolve().parent.parent


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    natoms = int(lines[0])
    rows = [line.split() for line in lines[2 : 2 + natoms]]
    if len(rows) != natoms:
        raise ValueError(f"{path}: expected {natoms} atoms, found {len(rows)}")
    return [(row[0], tuple(float(value) for value in row[1:4])) for row in rows]


def build_mol(atoms, basis: str, memory_mb: int):
    return gto.M(
        atom=atoms,
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        symmetry=False,
        cart=False,
        max_memory=memory_mb,
        verbose=4,
    )


def build_ks(mol, args, checkpoint: Path):
    mf = dft.RKS(mol, xc=args.xc)
    mf.grids.level = args.grid_level
    mf.grids.prune = dft.gen_grid.nwchem_prune
    mf.small_rho_cutoff = 1e-7
    mf.conv_tol = args.conv_tol
    mf.conv_tol_grad = max(args.conv_tol**0.5, 1e-5)
    mf.max_cycle = 100
    mf.chkfile = str(checkpoint)
    return mf.density_fit()


def run_scf(mf, label: str) -> float:
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{label} SCF did not converge")
    print(f"{label} total energy: {energy:.12f} Eh")
    return float(energy)


def atom_mask(mol, indices) -> np.ndarray:
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    slices = mol.aoslice_by_atom()
    for index in indices:
        p0, p1 = slices[index, 2], slices[index, 3]
        mask[p0:p1] = True
    return mask


def mulliken_weights(coefficients, overlap, mask) -> np.ndarray:
    sc = overlap @ coefficients
    return np.einsum("pi,pi->i", coefficients[mask].conj(), sc[mask]).real


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, default=ROOT / "data" / "2H2Pc_C60.xyz")
    parser.add_argument("--fragments", type=Path, default=ROOT / "data" / "fragments.json")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="pbe-d3bj")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=56000)
    parser.add_argument("--conv-tol", type=float, default=1e-8)
    parser.add_argument("--outdir", type=Path, default=ROOT / "output_static")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    lib.num_threads(args.threads)

    atoms = read_xyz(args.xyz)
    fragments = json.loads(args.fragments.read_text())
    donor_stop = int(fragments["donor_2H2Pc"]["end_1based"])
    c60_start = int(fragments["acceptor_C60"]["start_1based"]) - 1
    if len(atoms) != 176 or donor_stop != 116 or c60_start != 116:
        raise ValueError("Unexpected geometry or fragment map")

    full_mol = build_mol(atoms, args.basis, args.memory_mb)
    donor_mol = build_mol(atoms[:donor_stop], args.basis, args.memory_mb)
    donor_mf = build_ks(donor_mol, args, args.outdir / "donor.chk")
    full_mf = build_ks(full_mol, args, args.outdir / "complex.chk")

    run_scf(donor_mf, "2H2Pc donor")
    run_scf(full_mf, "2H2Pc/C60 complex")

    donor_lumo = int(np.flatnonzero(donor_mf.mo_occ == 0)[0])
    full_lumo = int(np.flatnonzero(full_mf.mo_occ == 0)[0])
    active = np.arange(full_lumo, full_lumo + 10)
    overlap = full_mol.intor_symmetric("int1e_ovlp")

    donor_weight = mulliken_weights(
        full_mf.mo_coeff,
        overlap,
        atom_mask(full_mol, range(donor_stop)),
    )
    c60_weight = mulliken_weights(
        full_mf.mo_coeff,
        overlap,
        atom_mask(full_mol, range(c60_start, len(atoms))),
    )

    donor_nao = int(full_mol.aoslice_by_atom()[donor_stop - 1, 3])
    if donor_nao != donor_mol.nao_nr():
        raise RuntimeError("Donor AO ordering differs between calculations")

    donor_orbital = np.zeros(full_mol.nao_nr())
    donor_orbital[:donor_nao] = donor_mf.mo_coeff[:, donor_lumo]
    donor_orbital /= np.sqrt(np.vdot(donor_orbital, overlap @ donor_orbital).real)
    c0 = full_mf.mo_coeff.conj().T @ overlap @ donor_orbital

    capture_all = float(np.vdot(c0, c0).real)
    capture_virtual = float(np.vdot(c0[full_lumo:], c0[full_lumo:]).real)
    capture_active = float(np.vdot(c0[active], c0[active]).real)

    csv_path = args.outdir / "frontier_LUMO_to_LUMO9.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "mo_index_0based",
            "label",
            "energy_hartree",
            "energy_eV",
            "donor_Mulliken_weight",
            "C60_Mulliken_weight",
            "c0_real",
            "c0_imag",
            "initial_probability",
        ])
        for index in active:
            writer.writerow([
                index,
                f"LUMO+{index - full_lumo}",
                full_mf.mo_energy[index],
                full_mf.mo_energy[index] * HARTREE_TO_EV,
                donor_weight[index],
                c60_weight[index],
                c0[index].real,
                c0[index].imag,
                abs(c0[index]) ** 2,
            ])

    np.savez_compressed(
        args.outdir / "frontier_data.npz",
        full_lumo=full_lumo,
        active_indices=active,
        mo_energy=full_mf.mo_energy,
        active_mo_coeff=full_mf.mo_coeff[:, active],
        overlap=overlap,
        donor_weight=donor_weight,
        c60_weight=c60_weight,
        c0=c0,
        donor_lumo_coeff=donor_mf.mo_coeff[:, donor_lumo],
    )

    with (args.outdir / "frontier_LUMO_to_LUMO9.molden").open("w") as handle:
        molden.header(full_mol, handle)
        molden.orbital_coeff(
            full_mol,
            handle,
            full_mf.mo_coeff[:, active],
            ene=full_mf.mo_energy[active],
            occ=np.zeros(len(active)),
        )

    summary = {
        "basis": args.basis,
        "xc": args.xc,
        "full_nelectron": full_mol.nelectron,
        "full_homo_0based": full_lumo - 1,
        "full_lumo_0based": full_lumo,
        "donor_lumo_0based": donor_lumo,
        "projection_norm_all": capture_all,
        "projection_norm_virtual": capture_virtual,
        "projection_norm_LUMO_to_LUMO9": capture_active,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Ten-state donor-LUMO capture: {capture_active:.10f}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        if "dftd" in str(exc).lower():
            print("Install pyscf-dispersion or use --xc pbe.", file=sys.stderr)
        raise
