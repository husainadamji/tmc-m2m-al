"""
Multi-objective active learning for catalyst design.

Uses 2D Expected Hypervolume Improvement (EHVI) with neural network surrogates
to select candidates for DFT evaluation on two objectives:
HAT (hydrogen atom transfer) and rebound energetics.
"""

__version__ = "0.1.0"

from .similarity import (
    filter_candidates_by_tanimoto,
    max_tanimoto_to_reference,
    tanimoto_similarity,
    tanimoto_similarity_matrix,
)
