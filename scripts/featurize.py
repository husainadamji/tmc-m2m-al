#!/usr/bin/env python3
"""
Featurize complexes (metal, oxidation, spin, tetradentate, axial ligands) for the active learning design space.

Supports two modes:
  initial         – Pre-functionalization: smaller tetra set, base axial ligands only. Writes candidates_initial.csv + failed log CSV.
  functionalized  – Post-functionalization: full tetra + functionalized axial. Writes features_<idx>.json only (failed structures in each JSON's "failed" key).

Output feature schema matches active_learning.data.design_space and train/predict/select_batch.

Usage:
  # Initial (small) design space
  python featurize.py --mode initial --tetra-csv data/tetra_ligands.csv --axial-dir data/axial_ligands --out-csv data/candidates_initial.csv

  # Functionalized (large) design space; writes data/featurization/features_*.json only (no combined CSV)
  python featurize.py --mode functionalized --tetra-csv data/ligands_functionalized.csv --axial-dir data/axial_ligands --out-dir data/featurization
"""
import argparse
import ast
import json
import os
import tempfile
from itertools import combinations

import numpy as np
import pandas as pd

from molSimplify.Classes.mol3D import mol3D, atom3D
from molSimplify.Classes.mol2D import Mol2D
from molSimplify.Classes.ligand import ligand
from molSimplify.Informatics.RACassemble import assemble_connectivity_from_parts
from molSimplify.Informatics.lacRACAssemble import get_descriptor_vector
from molSimplify.Informatics.graph_racs import ligand_racs, ligand_racs_names

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

METALS = ["mn", "fe"]
SPIN_DICT = {
    "fe": {4: [1, 3, 5], 5: [2, 4]},
    "mn": {4: [2, 4], 5: [1, 3]},
}
RAC_DEPTH = 4

DEFAULT_OG_AXIAL_NAMES = [
    "Nbenzene", "Nnegbenzene", "Obenzene", "Onegbenzene",
    "Sbenzene", "Snegbenzene", "Pbenzene", "Pnegbenzene",
]
DEFAULT_OG_AXIAL_CHARGES = {
    "Nbenzene": 0, "Nnegbenzene": -1, "Obenzene": 0, "Onegbenzene": -1,
    "Sbenzene": 0, "Snegbenzene": -1, "Pbenzene": 0, "Pnegbenzene": -1,
}


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def safe_eval(x):
    if isinstance(x, str) and x.strip() == "-":
        return "-"
    return ast.literal_eval(x) if isinstance(x, str) else x


def load_tetradentate_data(csv_path: str, mode: str) -> pd.DataFrame:
    """Load tetradentate ligand CSV. Required columns depend on mode."""
    df = pd.read_csv(csv_path)
    if mode == "initial":
        required = [
            "csd_ligand", "csd_ligand_type", "tetradentate_lig_charge",
            "mol2_lig", "mol2_dummy_core", "metal_connecting_atoms",
        ]
    else:
        required = [
            "new_ligand_name", "csd_ligand_type", "tetradentate_lig_charge",
            "mol2_lig_functionalized", "mol2_w_metal_functionalized",
            "metal_connecting_atoms", "functional_group", "functionalized_sites_mol2_w_metal",
        ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tetradentate CSV missing columns: {missing}")
    return df


def load_axial_ligand_properties(axial_dir: str, mode: str):
    """
    Returns (og_properties, func_properties). For mode=initial, func_properties is empty.
    """
    og_properties = {}
    for name in DEFAULT_OG_AXIAL_NAMES:
        path = os.path.join(axial_dir, name + ".mol2")
        if os.path.isfile(path):
            og_properties[name] = {
                "mol2_path": path,
                "charge": DEFAULT_OG_AXIAL_CHARGES.get(name, 0),
                "functional_group": "-",
            }

    func_properties = {}
    if mode == "functionalized" and os.path.isdir(axial_dir):
        for fname in os.listdir(axial_dir):
            if "_" not in fname or not fname.endswith(".mol2"):
                continue
            base = fname[:-5]
            parts = base.split("_", 1)
            og_name, func_group = parts[0], parts[1]
            if og_name not in og_properties:
                continue
            func_properties[base] = {
                "mol2_path": os.path.join(axial_dir, fname),
                "charge": og_properties[og_name]["charge"],
                "functional_group": func_group,
            }
    return og_properties, func_properties


# -----------------------------------------------------------------------------
# Geometry / ligand helpers
# -----------------------------------------------------------------------------

def get_angle(coord_list, pair, m_coord):
    try:
        p1 = np.squeeze(np.array(coord_list[pair[0]]))
        p2 = np.squeeze(np.array(coord_list[pair[1]]))
        v1u = (m_coord - p1) / np.linalg.norm(m_coord - p1)
        v2u = (m_coord - p2) / np.linalg.norm(m_coord - p2)
        return float(np.rad2deg(np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0))))
    except (IndexError, ZeroDivisionError):
        return 0.0


def smiles_to_ligand(smiles: str, denticity: int):
    mol = mol3D.from_smiles(smiles, gen3d=False)
    lig = ligand(mol3D(), [], denticity)
    lig.mol = mol
    return lig


def construct_ligand(mol2_path: str, denticity: int):
    mol = mol3D()
    mol.readfrommol2(mol2_path)
    lig = ligand(mol3D(), [], denticity)
    lig.mol = mol
    return lig


def make_ligand_dictionary_from_ligands(
    tetradentate,
    tetradentate_type: str,
    tetradentate_w_metal_mol,
    monodentate1,
    monodentate2,
    tetradentate_con,
    monodentate1_con,
    monodentate2_con,
    tetradentate_func_sites_w_metal=None,
    tetradentate_func_group: str = "-",
    loud: bool = False,
):
    """Build eq/ax ligand lists and connection indices for octahedral assembly."""
    assert tetradentate.dent + monodentate1.dent + monodentate2.dent == 6

    if tetradentate_type == "square planar":
        eq_number = int(4 / tetradentate.dent)
        eq_cons = eq_number * [tetradentate_con]
        eq_ligs = eq_number * [tetradentate]
        ax_ligs = [monodentate1, monodentate2]
        ax_cons = [monodentate1_con, monodentate2_con]
    elif tetradentate_type == "seesaw":
        m_idx = tetradentate_w_metal_mol.findAtomsbySymbol("X")[0]
        m_coord = np.array(tetradentate_w_metal_mol.getAtom(m_idx).coords())
        m_connecting_atoms = tetradentate_w_metal_mol.getBondedAtoms(m_idx)
        coord_list = np.array([tetradentate_w_metal_mol.getAtom(ii).coords() for ii in m_connecting_atoms])
        pair_combos = list(combinations([0, 1, 2, 3], 2))
        angle_list = []
        pair_list = []
        for pair in pair_combos:
            pair_list.append(list(pair))
            angle_list.append(get_angle(coord_list, pair_list[-1], m_coord))
        axial_not_allowed = [tetradentate_con[val] for val in pair_list[np.argmax(np.array(angle_list))]]
        possible_axial = list(set(tetradentate_con) - set(axial_not_allowed))
        ax_cons = [[possible_axial[0]], monodentate2_con]
        ax_ligs = [tetradentate, monodentate2]
        tetradentate_eq_cons = list(set(tetradentate_con) - set(ax_cons[0]))
        eq_cons = [tetradentate_eq_cons, monodentate1_con]
        eq_ligs = [tetradentate, monodentate1]
    else:
        raise ValueError(f"Unknown tetradentate_type: {tetradentate_type}")

    if loud:
        print("ax_cons:", ax_cons, "eq_cons:", eq_cons)

    return {
        "eq_ligand_list": eq_ligs,
        "ax_ligand_list": ax_ligs,
        "eq_con_int_list": eq_cons,
        "ax_con_int_list": ax_cons,
    }


def _serialize_ligand_dict(custom_ligand_dict):
    return {
        "eq_ligand_list": [str(lig) for lig in custom_ligand_dict["eq_ligand_list"]],
        "ax_ligand_list": [str(lig) for lig in custom_ligand_dict["ax_ligand_list"]],
        "eq_con_int_list": custom_ligand_dict["eq_con_int_list"],
        "ax_con_int_list": custom_ligand_dict["ax_con_int_list"],
    }


def _append_racs(features, complex_3d, complex_2d, custom_ligand_dict):
    """Add RAC descriptor columns to features dict. Raises on failure."""
    feature_names, racs = get_descriptor_vector(
        complex_3d, custom_ligand_dict=custom_ligand_dict, depth=RAC_DEPTH
    )
    features.update(dict(zip(feature_names, racs)))
    lig_racs = ligand_racs(complex_2d, depth=RAC_DEPTH).flatten()
    lig_feature_names = [
        f"lig_{i}_{name}" for i in range(6) for name in ligand_racs_names(depth=RAC_DEPTH)
    ]
    features.update(dict(zip(lig_feature_names, lig_racs)))


# -----------------------------------------------------------------------------
# Initial mode: one tetra row at a time, no JSON, collect then write CSV
# -----------------------------------------------------------------------------

def featurize_initial_single_row(
    idx: int,
    tetra_df: pd.DataFrame,
    og_axial: dict,
) -> tuple:
    """Featurize all complexes for one tetradentate row (initial mode). Returns (features_list, failed_list)."""
    row = tetra_df.iloc[idx]
    tetralig = row["csd_ligand"]
    tetralig_type = row["csd_ligand_type"]
    tetralig_charge = row["tetradentate_lig_charge"]
    tetralig_mol2 = row["mol2_lig"]
    tetralig_mol2_w_metal = row["mol2_dummy_core"]
    tetralig_metal_conatoms = row["metal_connecting_atoms"]
    if isinstance(tetralig_metal_conatoms, str):
        tetralig_metal_conatoms = ast.literal_eval(tetralig_metal_conatoms)

    with tempfile.TemporaryDirectory(prefix="featurize_") as tmpdir:
        mol2_path = os.path.join(tmpdir, "tetra.mol2")
        mol2_w_metal_path = os.path.join(tmpdir, "tetra_w_metal.mol2")
        with open(mol2_path, "w") as f:
            f.write(tetralig_mol2)
        with open(mol2_w_metal_path, "w") as f:
            f.write(tetralig_mol2_w_metal)
        this_tetralig = construct_ligand(mol2_path, 4)
        this_tetralig_w_metal = mol3D()
        this_tetralig_w_metal.readfrommol2(mol2_w_metal_path)

    all_features = []
    failed_structures = []

    for monolig, mono_props in og_axial.items():
        monolig_charge = mono_props["charge"]
        monolig_metal_conatoms = [0]
        this_monolig = construct_ligand(mono_props["mol2_path"], 1)
        oxo_lig = smiles_to_ligand("[O--]", 1)
        oxo_lig_metal_conatoms = [0]

        custom_ligand_dict = make_ligand_dictionary_from_ligands(
            this_tetralig,
            tetralig_type,
            this_tetralig_w_metal,
            this_monolig,
            oxo_lig,
            tetralig_metal_conatoms,
            monolig_metal_conatoms,
            oxo_lig_metal_conatoms,
        )
        custom_ligand_dict_serializable = _serialize_ligand_dict(custom_ligand_dict)

        for metal in METALS:
            metal_mol = mol3D()
            metal_mol.addAtom(atom3D(metal.capitalize()))
            complex_3d = assemble_connectivity_from_parts(metal_mol, custom_ligand_dict)
            complex_2d = Mol2D.from_mol3d(assemble_connectivity_from_parts(metal_mol, custom_ligand_dict))

            for ox_state in SPIN_DICT[metal].keys():
                for spin in SPIN_DICT[metal][ox_state]:
                    complex_name = (
                        f"{metal}_{ox_state}_{spin}_{str(this_tetralig.mol)[6:-1]}_"
                        f"{int(tetralig_charge)}_{monolig}_{int(monolig_charge)}_oxo_-2"
                    )
                    features = {
                        "name": complex_name,
                        "tetradentate_ligand_name": tetralig,
                        "tetradentate_ligand_type": tetralig_type,
                        "monodentate_ligand_name": monolig,
                        "OHE_mn": int(metal == "mn"),
                        "OHE_fe": int(metal == "fe"),
                        "ox": ox_state,
                        "spin": spin,
                        "monocharge": int(monolig_charge),
                        "tetracharge": int(tetralig_charge),
                        "custom_ligand_dict": custom_ligand_dict_serializable,
                    }
                    try:
                        _append_racs(features, complex_3d, complex_2d, custom_ligand_dict)
                        all_features.append(features)
                    except Exception as e:
                        print(f"Error generating RACs for {complex_name}: {e}")
                        features["error_message"] = str(e)
                        failed_structures.append(features)

    return all_features, failed_structures


# -----------------------------------------------------------------------------
# Functionalized mode: per-index JSON + combined CSV
# -----------------------------------------------------------------------------

def featurize_functionalized_one_tetradentate(
    idx: int,
    tetra_df: pd.DataFrame,
    og_axial: dict,
    func_axial: dict,
    out_dir: str,
    skip_existing: bool = True,
) -> tuple:
    """Featurize all complexes for tetradentate at row idx (functionalized mode). Writes features_<idx>.json."""
    out_path = os.path.join(out_dir, f"features_{idx}.json")
    if skip_existing and os.path.isfile(out_path):
        with open(out_path) as f:
            data = json.load(f)
        return data["features"], data["failed"]

    row = tetra_df.iloc[idx]
    tetralig = row["new_ligand_name"]
    tetralig_type = row["csd_ligand_type"]
    tetralig_charge = row["tetradentate_lig_charge"]
    tetralig_mol2 = row["mol2_lig_functionalized"]
    tetralig_mol2_w_metal = row["mol2_w_metal_functionalized"]
    tetralig_metal_conatoms = row["metal_connecting_atoms"]
    if isinstance(tetralig_metal_conatoms, str):
        tetralig_metal_conatoms = ast.literal_eval(tetralig_metal_conatoms)
    tetralig_fg = row["functional_group"]
    this_tetralig_func_sites_w_metal = safe_eval(row["functionalized_sites_mol2_w_metal"])

    with tempfile.TemporaryDirectory(prefix="featurize_") as tmpdir:
        mol2_path = os.path.join(tmpdir, "tetra.mol2")
        mol2_w_metal_path = os.path.join(tmpdir, "tetra_w_metal.mol2")
        with open(mol2_path, "w") as f:
            f.write(tetralig_mol2)
        with open(mol2_w_metal_path, "w") as f:
            f.write(tetralig_mol2_w_metal)
        this_tetralig = construct_ligand(mol2_path, 4)
        this_tetralig_w_metal = mol3D()
        this_tetralig_w_metal.readfrommol2(mol2_w_metal_path)

    if tetralig_fg == "-":
        ohe_tetra_func = 0
        ohe_mono_func = 1
        axial_ligand_properties = func_axial if func_axial else og_axial
    else:
        ohe_tetra_func = 1
        ohe_mono_func = 0
        axial_ligand_properties = og_axial

    all_features = []
    failed_structures = []

    for monolig, mono_props in axial_ligand_properties.items():
        monolig_charge = mono_props["charge"]
        monolig_func_group = mono_props["functional_group"]
        monolig_metal_conatoms = [0]
        this_monolig = construct_ligand(mono_props["mol2_path"], 1)
        oxo_lig = smiles_to_ligand("[O--]", 1)
        oxo_lig_metal_conatoms = [0]

        custom_ligand_dict = make_ligand_dictionary_from_ligands(
            this_tetralig,
            tetralig_type,
            this_tetralig_w_metal,
            this_monolig,
            oxo_lig,
            tetralig_metal_conatoms,
            monolig_metal_conatoms,
            oxo_lig_metal_conatoms,
            tetradentate_func_sites_w_metal=this_tetralig_func_sites_w_metal,
            tetradentate_func_group=tetralig_fg,
        )
        custom_ligand_dict_serializable = _serialize_ligand_dict(custom_ligand_dict)

        for metal in METALS:
            metal_mol = mol3D()
            metal_mol.addAtom(atom3D(metal.capitalize()))
            complex_3d = assemble_connectivity_from_parts(metal_mol, custom_ligand_dict)
            complex_2d = Mol2D.from_mol3d(assemble_connectivity_from_parts(metal_mol, custom_ligand_dict))

            for ox_state in SPIN_DICT[metal].keys():
                for spin in SPIN_DICT[metal][ox_state]:
                    complex_name = (
                        f"{metal}_{ox_state}_{spin}_{str(this_tetralig.mol)[6:-1]}_{tetralig}_"
                        f"{int(tetralig_charge)}_{monolig}_{int(monolig_charge)}_oxo_-2"
                    )
                    features = {
                        "name": complex_name,
                        "tetradentate_ligand_name": tetralig,
                        "tetradentate_ligand_type": tetralig_type,
                        "tetradentate_functional_group": tetralig_fg,
                        "monodentate_ligand_name": monolig,
                        "monodentate_ligand_functional_group": monolig_func_group,
                        "OHE_mn": int(metal == "mn"),
                        "OHE_fe": int(metal == "fe"),
                        "ox": ox_state,
                        "spin": spin,
                        "monocharge": int(monolig_charge),
                        "tetracharge": int(tetralig_charge),
                        "custom_ligand_dict": custom_ligand_dict_serializable,
                    }
                    try:
                        _append_racs(features, complex_3d, complex_2d, custom_ligand_dict)
                    except Exception as e:
                        print(f"Error generating RACs for {complex_name}: {e}")
                        features["error_message"] = str(e)
                        failed_structures.append(features)
                        continue
                    features["OHE_tetra_func"] = ohe_tetra_func
                    features["OHE_mono_func"] = ohe_mono_func
                    all_features.append(features)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"features": all_features, "failed": failed_structures}, f)
    return all_features, failed_structures


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Featurize complexes for active learning design space.")
    p.add_argument(
        "--mode",
        choices=["initial", "functionalized"],
        default="functionalized",
        help="initial = pre-functionalization (small tetra set, CSV + failed log); functionalized = JSON only (failed in each JSON)",
    )
    p.add_argument(
        "--tetra-csv",
        default=None,
        help="Tetradentate ligand CSV (default: data/tetra_ligands.csv for initial, data/ligands_functionalized.csv for functionalized)",
    )
    p.add_argument(
        "--axial-dir",
        default="data/axial_ligands",
        help="Directory of axial (monodentate) mol2 files",
    )
    p.add_argument(
        "--out-dir",
        default="data/featurization",
        help="Output directory for features_<idx>.json (functionalized mode only)",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV of featurized complexes (initial mode only; default: data/candidates_initial.csv). Functionalized mode writes JSON only.",
    )
    p.add_argument(
        "--failed-csv",
        default="data/featurization_failed.csv",
        help="Output CSV of failed structures (initial mode only; functionalized keeps failed in each JSON)",
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Recompute all (functionalized: do not skip existing JSON)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    tetra_csv = args.tetra_csv or (
        "data/tetra_ligands.csv" if mode == "initial" else "data/ligands_functionalized.csv"
    )
    out_csv = args.out_csv or "data/candidates_initial.csv" if mode == "initial" else None

    tetra_df = load_tetradentate_data(tetra_csv, mode)
    og_axial, func_axial = load_axial_ligand_properties(args.axial_dir, mode)
    n = len(tetra_df)
    print(f"Mode: {mode} | Tetradentates: {n} | Axial (og): {len(og_axial)} | Axial (func): {len(func_axial)}")

    if mode == "initial":
        all_features = []
        failed_structures = []
        for idx in range(n):
            feats, failed = featurize_initial_single_row(idx, tetra_df, og_axial)
            all_features.extend(feats)
            failed_structures.extend(failed)
            if (idx + 1) % 10 == 0 or idx == n - 1:
                print(f"Done {idx + 1}/{n}")
        features_df = pd.DataFrame(all_features) if all_features else pd.DataFrame()
        failed_df = pd.DataFrame(failed_structures) if failed_structures else pd.DataFrame()
        features_df.to_csv(out_csv, index=False)
        failed_df.to_csv(args.failed_csv, index=False)
        print(f"Wrote {out_csv} ({len(features_df)} rows) and {args.failed_csv} ({len(failed_df)} rows).")
    else:
        for idx in range(n):
            featurize_functionalized_one_tetradentate(
                idx,
                tetra_df,
                og_axial,
                func_axial,
                args.out_dir,
                skip_existing=not args.no_skip,
            )
            if (idx + 1) % 10 == 0 or idx == n - 1:
                print(f"Done {idx + 1}/{n}")
        print(f"Wrote {args.out_dir}/features_*.json (design space; failed structures are in each JSON's \"failed\" key).")


if __name__ == "__main__":
    main()
