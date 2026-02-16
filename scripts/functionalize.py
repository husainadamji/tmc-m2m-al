#!/usr/bin/env python3
"""
Tetradentate and monodentate ligand functionalization.

All paths are CLI-configurable. For SAScore/SCScore run score_ligands.py on the output CSV.

Usage:
  python functionalize.py --axial-ligands-dir data/axial_ligands \\
    --tetra-csv data/tetra_ligands.csv --out-csv data/ligands_functionalized.csv
  python score_ligands.py --csv data/ligands_functionalized.csv \\
    --out data/ligands_functionalized_scored.csv --sc-model /path/to/model.ckpt.as_numpy.pickle
"""
import argparse
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from molSimplify.Classes.mol3D import mol3D, atom3D
import molSimplify.Scripts.geometry as geom_funs
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
np.set_printoptions(threshold=10_000)


# -----------------------------------------------------------------------------
# Chemistry / analysis helpers
# -----------------------------------------------------------------------------

def get_metal_coordinating_atoms(mol2):
    """Return metal-coordinating atom indices from mol2 string (with dummy metal X)."""
    mol = mol3D()
    mol.readfrommol2(mol2, readstring=True)
    mol_metal = mol.findAtomsbySymbol("X")[0]
    return mol.getBondedAtomsSmart(mol_metal, oct=False)


def count_aromatic_rings_from_mol2(name, mol2):
    """Count aromatic rings in ligand (mol2 string with metal); strip metal then use RDKit."""
    mol3d = mol3D()
    mol3d.readfrommol2(mol2, readstring=True)
    metal_idx = mol3d.findAtomsbySymbol("X")[0]
    mol3d.deleteatom(metal_idx)
    mol2_lig = mol3d.writemol2("tmp", writestring=True)
    del mol3d
    mol = Chem.MolFromMol2Block(mol2_lig, sanitize=True, removeHs=False)
    if mol is None:
        print(f"Warning: Failed to read molecule from {name}")
        return None
    ring_info = mol.GetRingInfo()
    return sum(
        1
        for ring_atoms in ring_info.AtomRings()
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atoms)
    )


def get_functionalizable_ring_carbons(name, mol2, metal_bound_atoms):
    """Return list of symmetry-equivalent groups of carbon indices (in mol2 with metal) that can be functionalized."""
    mol3d = mol3D()
    mol3d.readfrommol2(mol2, readstring=True)
    metal_idx = mol3d.findAtomsbySymbol("X")[0]
    old_to_new_idx = {}
    new_to_old_idx = {}
    new_idx = 0
    for i in range(mol3d.natoms):
        if i == metal_idx:
            continue
        old_to_new_idx[i] = new_idx
        new_to_old_idx[new_idx] = i
        new_idx += 1
    mol3d.deleteatom(metal_idx)
    mol2_lig = mol3d.writemol2("tmp", writestring=True)
    del mol3d
    remapped = [old_to_new_idx[idx] for idx in metal_bound_atoms if idx in old_to_new_idx]
    mol = Chem.MolFromMol2Block(mol2_lig, sanitize=True, removeHs=False)
    if mol is None:
        print(f"Warning: Failed to read molecule from {name}")
        return []
    ring_atom_idxs = set(idx for ring in mol.GetRingInfo().AtomRings() for idx in ring)
    aromatic_ring_carbons = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic() and atom.GetAtomicNum() == 6 and atom.GetIdx() in ring_atom_idxs
    ]
    exclude_carbons = set()
    for coord_idx in remapped:
        for neighbor in mol.GetAtomWithIdx(coord_idx).GetNeighbors():
            if neighbor.GetAtomicNum() == 6:
                exclude_carbons.add(neighbor.GetIdx())
    eligible = []
    for idx in aromatic_ring_carbons:
        if idx in exclude_carbons:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if [n.GetSymbol() for n in atom.GetNeighbors()].count("H") != 1:
            continue
        if not all(
            n.GetIdx() in ring_atom_idxs
            for n in atom.GetNeighbors()
            if n.GetAtomicNum() > 1
        ):
            continue
        eligible.append(idx)
    symm_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
    symmetric_groups = defaultdict(list)
    for idx in eligible:
        symmetric_groups[symm_classes[idx]].append(idx)
    return [[new_to_old_idx[i] for i in group] for group in symmetric_groups.values()]


def get_monodentate_para_carbon(name, mol2_path):
    """Return list of one representative para carbon per symmetry group (for monodentate mol2 file)."""
    mol = Chem.MolFromMol2File(mol2_path, sanitize=True, removeHs=False)
    if mol is None:
        print(f"Warning: Failed to read molecule from {name}")
        return []
    ring_atom_idxs = set(idx for ring in mol.GetRingInfo().AtomRings() for idx in ring)
    aromatic_ring_carbons = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic() and atom.GetAtomicNum() == 6 and atom.GetIdx() in ring_atom_idxs
    ]
    atom0_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(0).GetNeighbors()]
    symm_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
    symmetric_groups = defaultdict(list)
    for idx in aromatic_ring_carbons:
        symmetric_groups[symm_classes[idx]].append(idx)
    return [
        group[0]
        for group in symmetric_groups.values()
        if len(group) == 1 and group[0] not in atom0_neighbors
    ]


# -----------------------------------------------------------------------------
# Geometry / FG application (mol3D in place)
# -----------------------------------------------------------------------------

def replace_h_with_connecting_atom(mol3d_mol, carbon_idx, connecting_atom_id, connecting_atom_bl):
    """Replace H on carbon with connecting atom; return its index."""
    neighbors = mol3d_mol.getBondedAtomsSmart(carbon_idx, oct=False)
    H_idx = next(x for x in neighbors if mol3d_mol.getAtom(x).symbol() == "H")
    mol3d_mol.getAtom(H_idx).mutate(newType=connecting_atom_id)
    connecting_atom_idx = int(H_idx)
    mol3d_mol.BCM(connecting_atom_idx, carbon_idx, connecting_atom_bl)
    return connecting_atom_idx


def compute_direction_vector(mol3d_mol, from_idx, to_idx):
    return np.array(geom_funs.normalize(
        np.subtract(mol3d_mol.getAtomCoords(from_idx), mol3d_mol.getAtomCoords(to_idx))
    ))


def add_fg_atom(mol3d_mol, origin_coords, direction, fg_atom_id, fg_atom_bl, bond_to_idx, fg_atom_bo, angle_atom_idx=None, angle=None):
    coords = np.add(origin_coords, fg_atom_bl * direction)
    fg_atom = atom3D(fg_atom_id, coords)
    mol3d_mol.addAtom(fg_atom, auto_populate_BO_dict=True)
    fg_idx = mol3d_mol.natoms - 1
    mol3d_mol.add_bond(bond_to_idx, fg_idx, fg_atom_bo)
    if angle is not None and angle_atom_idx is not None:
        mol3d_mol.ACM(fg_idx, bond_to_idx, angle_atom_idx, angle)
    mol3d_mol.BCM(fg_idx, bond_to_idx, fg_atom_bl)
    return fg_idx


def add_atomic_fg(mol3d_mol, carbon_idxs_to_func, connecting_atom_id, connecting_atom_bl):
    for c_idx in carbon_idxs_to_func:
        replace_h_with_connecting_atom(mol3d_mol, c_idx, connecting_atom_id, connecting_atom_bl)


def add_linear_fg(mol3d_mol, carbon_idxs_to_func, connecting_atom_id, connecting_atom_bl, fg_atoms_id, fg_atoms_bl, fg_atoms_bo):
    for c_idx in carbon_idxs_to_func:
        connecting_atom_idx = replace_h_with_connecting_atom(mol3d_mol, c_idx, connecting_atom_id, connecting_atom_bl)
        n = compute_direction_vector(mol3d_mol, connecting_atom_idx, c_idx)
        add_fg_atom(mol3d_mol, mol3d_mol.getAtomCoords(connecting_atom_idx), n, fg_atoms_id, fg_atoms_bl, connecting_atom_idx, fg_atoms_bo)


def add_bent_fg(mol3d_mol, carbon_idxs_to_func, connecting_atom_id, connecting_atom_bl, fg_atoms_id, fg_atoms_bl, fg_atoms_angle, fg_atoms_bo):
    for c_idx in carbon_idxs_to_func:
        connecting_atom_idx = replace_h_with_connecting_atom(mol3d_mol, c_idx, connecting_atom_id, connecting_atom_bl)
        n = compute_direction_vector(mol3d_mol, connecting_atom_idx, c_idx)
        add_fg_atom(mol3d_mol, mol3d_mol.getAtomCoords(connecting_atom_idx), n, fg_atoms_id, fg_atoms_bl, connecting_atom_idx, fg_atoms_bo, angle_atom_idx=c_idx, angle=fg_atoms_angle)


def add_trigonal_planar_fg(mol3d_mol, carbon_idxs_to_func, connecting_atom_id, connecting_atom_bl, fg_atoms_id, fg_atoms_bl, fg_atoms_angle, fg_atoms_bo):
    for c_idx in carbon_idxs_to_func:
        connecting_atom_idx = replace_h_with_connecting_atom(mol3d_mol, c_idx, connecting_atom_id, connecting_atom_bl)
        n = compute_direction_vector(mol3d_mol, connecting_atom_idx, c_idx)
        fg1_idx = add_fg_atom(mol3d_mol, mol3d_mol.getAtomCoords(connecting_atom_idx), n, fg_atoms_id, fg_atoms_bl, connecting_atom_idx, fg_atoms_bo, angle_atom_idx=c_idx, angle=fg_atoms_angle)
        fg2_coords = geom_funs.PointRotateAxis(n, mol3d_mol.getAtomCoords(connecting_atom_idx), mol3d_mol.getAtomCoords(fg1_idx), 2.0944)
        fg2 = atom3D(fg_atoms_id, fg2_coords)
        mol3d_mol.addAtom(fg2, auto_populate_BO_dict=True)
        mol3d_mol.add_bond(connecting_atom_idx, mol3d_mol.natoms - 1, fg_atoms_bo)


def add_tetrahedral_fg(mol3d_mol, carbon_idxs_to_func, connecting_atom_id, connecting_atom_bl, fg_atoms_id, fg_atoms_bl, fg_atoms_angle, fg_atoms_bo):
    for c_idx in carbon_idxs_to_func:
        connecting_atom_idx = replace_h_with_connecting_atom(mol3d_mol, c_idx, connecting_atom_id, connecting_atom_bl)
        n = compute_direction_vector(mol3d_mol, connecting_atom_idx, c_idx)
        fg1_idx = add_fg_atom(mol3d_mol, mol3d_mol.getAtomCoords(connecting_atom_idx), n, fg_atoms_id, fg_atoms_bl, connecting_atom_idx, fg_atoms_bo, angle_atom_idx=c_idx, angle=fg_atoms_angle)
        coords1 = mol3d_mol.getAtomCoords(fg1_idx)
        origin = mol3d_mol.getAtomCoords(connecting_atom_idx)
        for angle in (2.0944, 2.0944 * 2):
            coords = geom_funs.PointRotateAxis(n, origin, coords1, angle)
            fg = atom3D(fg_atoms_id, coords)
            mol3d_mol.addAtom(fg, auto_populate_BO_dict=True)
            mol3d_mol.add_bond(connecting_atom_idx, mol3d_mol.natoms - 1, fg_atoms_bo)


def add_carbonyl(mol3d_mol, carbon_idxs_to_func):
    for c_idx in carbon_idxs_to_func:
        connecting_atom_idx = replace_h_with_connecting_atom(mol3d_mol, c_idx, "C", 1.5)
        n = compute_direction_vector(mol3d_mol, connecting_atom_idx, c_idx)
        O_idx = add_fg_atom(mol3d_mol, mol3d_mol.getAtomCoords(connecting_atom_idx), n, "O", 1.22, connecting_atom_idx, 2, angle_atom_idx=c_idx, angle=120)
        coordsO = mol3d_mol.getAtomCoords(O_idx)
        origin = mol3d_mol.getAtomCoords(connecting_atom_idx)
        coordsH = geom_funs.PointRotateAxis(n, origin, coordsO, 3.14159)
        H_atom = atom3D("H", coordsH)
        mol3d_mol.addAtom(H_atom, auto_populate_BO_dict=True)
        mol3d_mol.BCM(connecting_atom_idx, mol3d_mol.natoms - 1, 1.1)
        mol3d_mol.add_bond(connecting_atom_idx, mol3d_mol.natoms - 1, 1)


# -----------------------------------------------------------------------------
# Functional group builders (take mol3d, carbon indices; return mol2 string)
# -----------------------------------------------------------------------------

def _add_nh2(mol3d_mol, carbon_idxs_to_func):
    add_trigonal_planar_fg(mol3d_mol, carbon_idxs_to_func, "N", 1.4, "H", 1.0, 110, 1)
    return mol3d_mol.writemol2("tmp", writestring=True)


def _add_oh(mol3d_mol, carbon_idxs_to_func):
    add_bent_fg(mol3d_mol, carbon_idxs_to_func, "O", 1.4, "H", 0.97, 110, 1)
    return mol3d_mol.writemol2("tmp", writestring=True)


def _add_ch3(mol3d_mol, carbon_idxs_to_func):
    add_tetrahedral_fg(mol3d_mol, carbon_idxs_to_func, "C", 1.5, "H", 1.1, 109, 1)
    return mol3d_mol.writemol2("tmp", writestring=True)


def _add_f(mol3d_mol, carbon_idxs_to_func):
    add_atomic_fg(mol3d_mol, carbon_idxs_to_func, "F", 1.34)
    return mol3d_mol.writemol2("tmp", writestring=True)


def _add_sh(mol3d_mol, carbon_idxs_to_func):
    add_bent_fg(mol3d_mol, carbon_idxs_to_func, "S", 1.8, "H", 1.34, 97, 1)
    return mol3d_mol.writemol2("tmp", writestring=True)


def _add_cl(mol3d_mol, carbon_idxs_to_func):
    add_atomic_fg(mol3d_mol, carbon_idxs_to_func, "Cl", 1.72)
    return mol3d_mol.writemol2("tmp", writestring=True)


def _add_br(mol3d_mol, carbon_idxs_to_func):
    add_atomic_fg(mol3d_mol, carbon_idxs_to_func, "Br", 1.89)
    return mol3d_mol.writemol2("tmp", writestring=True)


def _add_cf3(mol3d_mol, carbon_idxs_to_func):
    add_tetrahedral_fg(mol3d_mol, carbon_idxs_to_func, "C", 1.5, "F", 1.36, 109, 1)
    return mol3d_mol.writemol2("tmp", writestring=True)


def _add_cn(mol3d_mol, carbon_idxs_to_func):
    add_linear_fg(mol3d_mol, carbon_idxs_to_func, "C", 1.5, "N", 1.16, 3)
    return mol3d_mol.writemol2("tmp", writestring=True)


def _add_cho(mol3d_mol, carbon_idxs_to_func):
    add_carbonyl(mol3d_mol, carbon_idxs_to_func)
    return mol3d_mol.writemol2("tmp", writestring=True)


FUNCTIONAL_GROUPS = {
    "NH2": _add_nh2,
    "OH": _add_oh,
    "CH3": _add_ch3,
    "F": _add_f,
    "SH": _add_sh,
    "Cl": _add_cl,
    "Br": _add_br,
    "CF3": _add_cf3,
    "CN": _add_cn,
    "CHO": _add_cho,
}


# -----------------------------------------------------------------------------
# Workflow steps
# -----------------------------------------------------------------------------

def run_monodentate_functionalization(axial_dir: str) -> None:
    """Write functionalized monodentate mol2 files (base_name_FG.mol2) into axial_dir."""
    if not os.path.isdir(axial_dir):
        raise FileNotFoundError(f"Axial ligands directory not found: {axial_dir}")
    for mol2_f in os.listdir(axial_dir):
        if not mol2_f.endswith(".mol2"):
            continue
        name = mol2_f.replace(".mol2", "")
        if "_" in name:
            continue
        mol2_path = os.path.join(axial_dir, mol2_f)
        para_carbon = get_monodentate_para_carbon(name, mol2_path)
        for fg_name, fg_func in FUNCTIONAL_GROUPS.items():
            mol_lig = mol3D()
            mol_lig.readfrommol2(mol2_path)
            mod_lig_str = fg_func(deepcopy(mol_lig), para_carbon)
            new_name = name + "_" + fg_name
            with open(os.path.join(axial_dir, new_name + ".mol2"), "w") as f:
                f.write(mod_lig_str)


def load_and_filter_tetradentate(tetra_csv_path: str):
    """
    Load tetradentate CSV, add aromatic_ring_count, filter to 0 < ring_count <= 8.
    Returns (tetra_df, excluded_tetra_df, original_tetra_df with ring counts).
    """
    if not os.path.isfile(tetra_csv_path):
        raise FileNotFoundError(f"Tetradentate CSV not found: {tetra_csv_path}")
    tetra_df = pd.read_csv(tetra_csv_path)
    tetra_df["tetradentate_lig_charge"] = pd.to_numeric(tetra_df["tetradentate_lig_charge"], errors="coerce")
    tetra_df = tetra_df[tetra_df["tetradentate_lig_charge"].notna()]
    original_tetra_df = tetra_df.copy()
    lig_names = tetra_df["csd_ligand"].values
    metal_mol2s = tetra_df["mol2_dummy_core"].values

    aromatic_ring_count_list = []
    for i, lig_name in enumerate(lig_names):
        aromatic_ring_count_list.append(count_aromatic_rings_from_mol2(lig_name, metal_mol2s[i]))
    print(f"Could not parse file for {sum(x is None for x in aromatic_ring_count_list)} ligands")

    original_tetra_df["aromatic_ring_count"] = aromatic_ring_count_list
    tetra_df["aromatic_ring_count"] = aromatic_ring_count_list
    tetra_df = tetra_df[
        tetra_df["aromatic_ring_count"].notna()
        & (tetra_df["aromatic_ring_count"] > 0)
        & (tetra_df["aromatic_ring_count"] <= 8)
    ].reset_index(drop=True)
    excluded_tetra_df = original_tetra_df[~original_tetra_df["csd_ligand"].isin(tetra_df["csd_ligand"])].copy()
    return tetra_df, excluded_tetra_df


def build_tetradentate_rows(tetra_df: pd.DataFrame, excluded_tetra_df: pd.DataFrame) -> pd.DataFrame:
    """Build DataFrame of original + all functionalized tetradentate rows."""
    rows = []
    for idx, row in tetra_df.iterrows():
        ligand_name = row["csd_ligand"]
        lig_type = row["csd_ligand_type"]
        lig_hash = row["csd_graph_hash"]
        refcodes = row["refcodes"]
        charge = row["tetradentate_lig_charge"]
        aromatic_ring_count = row["aromatic_ring_count"]
        mol2_w_metal = row["mol2_dummy_core"]
        rows.append({
            "csd_ligand": ligand_name,
            "csd_ligand_type": lig_type,
            "csd_graph_hash": lig_hash,
            "refcodes": refcodes,
            "tetradentate_lig_charge": charge,
            "aromatic_ring_count": aromatic_ring_count,
            "functionalized_sites_mol2_w_metal": "-",
            "functional_group": "-",
            "mol2_w_metal_functionalized": mol2_w_metal,
            "metal_connecting_atoms_mol2_w_metal": get_metal_coordinating_atoms(mol2_w_metal),
        })
        connecting_atom_idxs_metal_mol2 = get_metal_coordinating_atoms(mol2_w_metal)
        ring_carbons_w_metal = get_functionalizable_ring_carbons(ligand_name, mol2_w_metal, connecting_atom_idxs_metal_mol2)
        for carbons_metal in ring_carbons_w_metal:
            for fg_name, fg_func in FUNCTIONAL_GROUPS.items():
                mol_metal = mol3D()
                mol_metal.readfrommol2(mol2_w_metal, readstring=True)
                mod_dummy_str = fg_func(deepcopy(mol_metal), carbons_metal)
                rows.append({
                    "csd_ligand": ligand_name,
                    "csd_ligand_type": lig_type,
                    "csd_graph_hash": lig_hash,
                    "refcodes": refcodes,
                    "tetradentate_lig_charge": charge,
                    "aromatic_ring_count": aromatic_ring_count,
                    "functionalized_sites_mol2_w_metal": carbons_metal,
                    "functional_group": fg_name,
                    "mol2_w_metal_functionalized": mod_dummy_str,
                    "metal_connecting_atoms_mol2_w_metal": connecting_atom_idxs_metal_mol2,
                })

    for _, row in excluded_tetra_df.iterrows():
        rows.append({
            "csd_ligand": row["csd_ligand"],
            "csd_ligand_type": row["csd_ligand_type"],
            "csd_graph_hash": row["csd_graph_hash"],
            "refcodes": row["refcodes"],
            "tetradentate_lig_charge": row["tetradentate_lig_charge"],
            "aromatic_ring_count": row["aromatic_ring_count"],
            "functionalized_sites_mol2_w_metal": "-",
            "functional_group": "-",
            "mol2_w_metal_functionalized": row["mol2_dummy_core"],
            "metal_connecting_atoms_mol2_w_metal": get_metal_coordinating_atoms(row["mol2_dummy_core"]),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Tetradentate and monodentate ligand functionalization; paths configurable."
    )
    p.add_argument(
        "--axial-ligands-dir",
        default="data/axial_ligands",
        help="Directory containing monodentate ligand mol2 files",
    )
    p.add_argument(
        "--tetra-csv",
        default="data/tetra_ligands.csv",
        help="Input CSV of tetradentate ligands with mol2_dummy_core column",
    )
    p.add_argument(
        "--out-csv",
        default="data/ligands_functionalized.csv",
        help="Output CSV after functionalization",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_monodentate_functionalization(args.axial_ligands_dir)
    tetra_df, excluded_tetra_df = load_and_filter_tetradentate(args.tetra_csv)
    all_ligands_df = build_tetradentate_rows(tetra_df, excluded_tetra_df)
    all_ligands_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    original_count = (all_ligands_df["functional_group"] == "-").sum()
    functionalized_count = len(all_ligands_df) - original_count
    print(f"Original ligands count: {original_count}")
    print(f"Functionalized ligands count: {functionalized_count}")
    print("To add SAScore/SCScore, run: score_ligands.py --csv <out-csv> --out <scored-csv> [--sc-model ...]")


if __name__ == "__main__":
    main()
