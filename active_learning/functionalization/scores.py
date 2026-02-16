"""
SAScore and SCScore for ligands (optional: RDKit, SA_Score, SCScore).

SAScore: synthetic accessibility (RDKit Contrib SA_Score).
SCScore: synthetic complexity (SCScore model; requires model path).
"""

from typing import Optional

import numpy as np
import pandas as pd

from .mol2_utils import mol2_ligand_from_mol2_metal


def _import_rdkit():
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError as e:
        raise ImportError("scores require RDKit.") from e


def _import_sascorer():
    try:
        from SA_Score import sascorer
        return sascorer
    except ImportError as e:
        raise ImportError(
            "SAScore requires SA_Score (RDKit Contrib). Ensure SA_Score is on your path."
        ) from e


def _import_scscorer():
    try:
        from standalone_model_numpy import SCScorer
        return SCScorer
    except ImportError as e:
        raise ImportError(
            "SCScore requires the standalone_model_numpy SCScorer. "
            "Add the scscore package to your path."
        ) from e


def compute_sa_score(mol) -> float:
    """
    Compute SAScore for an RDKit Mol.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule (ligand only, no metal).

    Returns
    -------
    float
        SAScore value.
    """
    sascorer = _import_sascorer()
    return sascorer.calculateScore(mol)


def compute_sc_score(
    mol2_lig_str: str,
    model_path: str,
    fp_len: int = 1024,
    fp_rad: int = 2,
):
    """
    Compute SCScore for a ligand mol2 string using a trained SCScore model.

    Parameters
    ----------
    mol2_lig_str : str
        Mol2 content of the ligand (no metal).
    model_path : str
        Path to the SCScore model checkpoint (e.g. .as_numpy.pickle).
    fp_len : int
        Fingerprint length (model-dependent).
    fp_rad : int
        Fingerprint radius (model-dependent).

    Returns
    -------
    float
        SCScore value.
    """
    SCScorer = _import_scscorer()
    model = SCScorer()
    model.restore(weight_path=model_path, FP_rad=fp_rad, FP_len=fp_len)
    return model.get_score_from_mol2_str(mol2_lig_str)


def add_scores_to_dataframe(
    df: pd.DataFrame,
    mol2_col: str = "mol2_w_metal_functionalized",
    sc_model_path: Optional[str] = None,
    fp_len: int = 1024,
    fp_rad: int = 2,
    name_col: Optional[str] = "csd_ligand",
) -> pd.DataFrame:
    """
    Add ligand_SAScore and ligand_SCScore columns to a DataFrame.

    Each row should contain a mol2 string (with metal) in mol2_col. Metal is
    stripped for RDKit/SAScore and SCScore. If sc_model_path is None, only
    SAScore is computed and ligand_SCScore is filled with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a column of mol2 strings (metal + ligand).
    mol2_col : str
        Column name for the mol2 content.
    sc_model_path : str, optional
        Path to SCScore model. If None, SCScore is not computed.
    fp_len, fp_rad : int
        SCScore model fingerprint parameters.
    name_col : str, optional
        Column name for row label (used in warning messages).

    Returns
    -------
    pd.DataFrame
        New DataFrame with ligand_SAScore and ligand_SCScore columns (same index as df).
    """
    Chem = _import_rdkit()
    sascorer = _import_sascorer()
    sc_model = None
    if sc_model_path:
        SCScorer = _import_scscorer()
        sc_model = SCScorer()
        sc_model.restore(weight_path=sc_model_path, FP_rad=fp_rad, FP_len=fp_len)

    out = df.copy()
    sascores = []
    scscores = []

    for idx, row in df.iterrows():
        mol2_w_metal = row[mol2_col]
        name = row.get(name_col, idx) if name_col and name_col in row else idx
        try:
            mol2_lig = mol2_ligand_from_mol2_metal(mol2_w_metal)
        except Exception as e:
            sascores.append(np.nan)
            scscores.append(np.nan)
            continue
        mol = Chem.MolFromMol2Block(mol2_lig, sanitize=True, removeHs=False)
        if mol is None:
            sascores.append(np.nan)
            scscores.append(np.nan)
            continue
        sascores.append(sascorer.calculateScore(mol))
        if sc_model is not None:
            scscores.append(sc_model.get_score_from_mol2_str(mol2_lig))
        else:
            scscores.append(np.nan)

    out["ligand_SAScore"] = sascores
    out["ligand_SCScore"] = scscores
    return out
