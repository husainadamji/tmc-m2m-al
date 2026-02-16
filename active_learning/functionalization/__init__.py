"""
Design-space functionalization and ligand complexity scores (optional subpackage).

This subpackage requires optional dependencies that are not needed for the core
active learning loop: RDKit, molSimplify, SA_Score (RDKit Contrib), and SCScore
for SAScore and SCScore. Install them separately and ensure paths (e.g. SCScore
model, RDKit Contrib) are set in your environment or passed to the functions.

Public API:
- add_scores_to_dataframe: add ligand_SAScore and ligand_SCScore columns to a DataFrame.
- compute_sa_score: compute SAScore for an RDKit Mol.
- compute_sc_score: compute SCScore for a ligand mol2 string (requires loaded model).
"""

from .scores import add_scores_to_dataframe, compute_sa_score, compute_sc_score
from .mol2_utils import mol2_ligand_from_mol2_metal

__all__ = [
    "add_scores_to_dataframe",
    "compute_sa_score",
    "compute_sc_score",
    "mol2_ligand_from_mol2_metal",
]
