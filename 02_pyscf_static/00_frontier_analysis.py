#!/usr/bin/env python3
"""PySCF frontier-orbital and donor-LUMO projection analysis for 2H2Pc/C60.

This is the first electronic-structure stage of the PySCF -> Libra/TENSO
workflow. It performs consistent donor and full-complex KS-DFT calculations,
projects the isolated donor LUMO into the full-complex MO basis, and reports
fragment Mulliken weights for LUMO through LUMO+9.
"""
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
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_XYZ = ROOT / '01_geometry' / '2H2Pc_C60.xyz'
DEFAULT_FRAGMENTS = ROOT / '01_geometry' / 'fragments.json'


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    natm = int(lines[0])
    rows = [line.split() for line in lines[2:2 + natm]]
    if len(rows) != natm:
        raise ValueError(f'{path}: expected {natm} coordinate lines, found {len(rows)}')
    return [(r[0], tuple(float(v) for v in r[1:4])) for r in rows]


def build_mol(atoms, basis: str, memory_mb: int, verbose: int = 4):
    return gto.M(
        atom=atoms,
        unit='Angstrom',
        basis=basis,
        charge=0,
        spin=0,
        symmetry=False,
        cart=False,
        max_memory=memory_mb,
        verbose=verbose,
    )


def build_ks(mol, xc: str, grid_level: int, chkfile: Path, conv_tol: float):
    # Strings such as pbe-d3bj require the pyscf-dispersion extension.
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = grid_level
    mf.grids.prune = dft.gen_grid.nwchem_prune
    mf.small_rho_cutoff = 1e-7
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = max(conv_tol ** 0.5, 1e-5)
    mf.max_cycle = 100
    mf.chkfile = str(chkfile)
    mf = mf.density_fit()
    return mf


def atom_ao_mask(mol, atom_indices):
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    slices = mol.aoslice_by_atom()
    for ia in atom_indices:
        p0, p1 = slices[ia, 2], slices[ia, 3]
        mask[p0:p1] = True
    return mask


def mulliken_mo_weights(mo_coeff, overlap, ao_mask):
    # w_i(F) = sum_{mu in F,nu} C_{mu i} S_{mu nu} C_{nu i}
    sc = overlap @ mo_coeff
    return np.einsum('pi,pi->i', mo_coeff[ao_mask].conj(), sc[ao_mask]).real


def run_or_fail(mf, label: str):
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f'{label} SCF did not converge')
    print(f'{label} total energy: {energy:.12f} Eh')
    return energy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', type=Path, default=DEFAULT_XYZ)
    p.add_argument('--fragments', type=Path, default=DEFAULT_FRAGMENTS)
    p.add_argument('--basis', default='sto-3g',
                   help='Use sto-3g for plumbing tests; def2-svp is the first serious calculation.')
    p.add_argument('--xc', default='pbe-d3bj',
                   help='Ground-state method. pbe-d3bj requires pyscf-dispersion.')
    p.add_argument('--grid-level', type=int, default=1)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--memory-mb', type=int, default=56000)
    p.add_argument('--conv-tol', type=float, default=1e-8)
    p.add_argument('--outdir', type=Path, default=HERE / 'output_static')
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    lib.num_threads(args.threads)

    atoms = read_xyz(args.xyz)
    fragment_info = json.loads(args.fragments.read_text())
    donor_end = fragment_info['donor_2H2Pc']['end_1based']
    c60_start = fragment_info['acceptor_C60']['start_1based'] - 1

    if len(atoms) != 176 or donor_end != 116 or c60_start != 116:
        raise ValueError('Unexpected geometry or fragment map')

    donor_atoms = atoms[:donor_end]
    full_mol = build_mol(atoms, args.basis, args.memory_mb)
    donor_mol = build_mol(donor_atoms, args.basis, args.memory_mb)

    print(f'PySCF threads: {lib.num_threads()}')
    print(f'Full complex: natm={full_mol.natm}, nelec={full_mol.nelectron}, nao={full_mol.nao_nr()}')
    print(f'Donor dimer: natm={donor_mol.natm}, nelec={donor_mol.nelectron}, nao={donor_mol.nao_nr()}')

    donor_mf = build_ks(
        donor_mol, args.xc, args.grid_level,
        args.outdir / 'donor.chk', args.conv_tol,
    )
    full_mf = build_ks(
        full_mol, args.xc, args.grid_level,
        args.outdir / 'complex.chk', args.conv_tol,
    )

    run_or_fail(donor_mf, '2H2Pc donor')
    run_or_fail(full_mf, '2H2Pc/C60 complex')

    donor_lumo = int(np.flatnonzero(donor_mf.mo_occ == 0)[0])
    full_lumo = int(np.flatnonzero(full_mf.mo_occ == 0)[0])
    full_homo = full_lumo - 1

    s_full = full_mol.intor_symmetric('int1e_ovlp')
    donor_mask = atom_ao_mask(full_mol, range(0, donor_end))
    c60_mask = atom_ao_mask(full_mol, range(c60_start, len(atoms)))

    donor_weight = mulliken_mo_weights(full_mf.mo_coeff, s_full, donor_mask)
    c60_weight = mulliken_mo_weights(full_mf.mo_coeff, s_full, c60_mask)

    # Embed the isolated donor LUMO into the full-system AO basis. Because the
    # donor atoms are first and the basis is identical, the donor AO block has
    # exactly the same ordering in both calculations.
    n_donor_ao_in_full = int(full_mol.aoslice_by_atom()[donor_end - 1, 3])
    if n_donor_ao_in_full != donor_mol.nao_nr():
        raise RuntimeError('Donor AO ordering/count does not match the full complex')

    phi_d = np.zeros(full_mol.nao_nr())
    phi_d[:n_donor_ao_in_full] = donor_mf.mo_coeff[:, donor_lumo]
    phi_d /= np.sqrt(np.vdot(phi_d, s_full @ phi_d).real)

    # c_i(0) = <phi_i(full)|phi_LUMO(donor)>
    c0 = full_mf.mo_coeff.conj().T @ s_full @ phi_d
    norm_all = float(np.vdot(c0, c0).real)
    norm_virtual = float(np.vdot(c0[full_lumo:], c0[full_lumo:]).real)

    active = np.arange(full_lumo, min(full_lumo + 10, full_mol.nao_nr()))
    norm_active = float(np.vdot(c0[active], c0[active]).real)

    print(f'Full HOMO index (0-based): {full_homo}')
    print(f'Full LUMO index (0-based): {full_lumo}')
    print(f'Donor LUMO index (0-based): {donor_lumo}')
    print(f'Projection norm over all full MOs: {norm_all:.10f}')
    print(f'Projection captured by all virtual MOs: {norm_virtual:.10f}')
    print(f'Projection captured by LUMO..LUMO+9: {norm_active:.10f}')

    csv_path = args.outdir / 'frontier_LUMO_to_LUMO9.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mo_index_0based', 'label', 'energy_hartree', 'energy_eV',
            'donor_Mulliken_weight', 'C60_Mulliken_weight',
            'c0_real', 'c0_imag', 'initial_probability',
        ])
        for i in active:
            writer.writerow([
                i, f'LUMO+{i-full_lumo}', full_mf.mo_energy[i],
                full_mf.mo_energy[i] * HARTREE_TO_EV,
                donor_weight[i], c60_weight[i],
                c0[i].real, c0[i].imag, abs(c0[i]) ** 2,
            ])

    np.savez_compressed(
        args.outdir / 'frontier_data.npz',
        full_lumo=full_lumo,
        active_indices=active,
        mo_energy=full_mf.mo_energy,
        active_mo_coeff=full_mf.mo_coeff[:, active],
        overlap=s_full,
        donor_weight=donor_weight,
        c60_weight=c60_weight,
        c0=c0,
        donor_lumo_coeff=donor_mf.mo_coeff[:, donor_lumo],
    )

    with (args.outdir / 'frontier_LUMO_to_LUMO9.molden').open('w') as f:
        molden.header(full_mol, f)
        molden.orbital_coeff(full_mol, f, full_mf.mo_coeff[:, active],
                             ene=full_mf.mo_energy[active], occ=np.zeros(len(active)))

    summary = {
        'basis': args.basis,
        'xc': args.xc,
        'grid_level': args.grid_level,
        'full_nelectron': full_mol.nelectron,
        'full_nao': full_mol.nao_nr(),
        'full_homo_0based': full_homo,
        'full_lumo_0based': full_lumo,
        'donor_lumo_0based': donor_lumo,
        'projection_norm_all': norm_all,
        'projection_norm_virtual': norm_virtual,
        'projection_norm_LUMO_to_LUMO9': norm_active,
    }
    (args.outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Wrote {csv_path}')


if __name__ == '__main__':
    try:
        main()
    except ModuleNotFoundError as exc:
        if 'dftd3' in str(exc).lower() or 'dftd4' in str(exc).lower():
            print('Dispersion extension missing. Install pyscf-dispersion or use --xc pbe.', file=sys.stderr)
        raise
