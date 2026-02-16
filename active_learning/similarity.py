"""
Tanimoto (extended Jaccard) similarity on real-valued feature vectors (e.g. RACs).

Used to screen candidates by excluding those too similar to a reference set
(e.g. failed DFT calculations) before EHVI batch selection.
"""

import numpy as np


def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Tanimoto coefficient for two 1D feature vectors.

    T(a, b) = dot(a,b) / (||a||^2 + ||b||^2 - dot(a,b)).
    Returns 0 when both vectors are zero (0/0).

    Parameters
    ----------
    a, b : np.ndarray
        Shape (n_features,). Non-negative RACs recommended.

    Returns
    -------
    float
        Similarity in [0, 1].
    """
    dot = float(np.dot(a, b))
    na2 = float(np.dot(a, a))
    nb2 = float(np.dot(b, b))
    denom = na2 + nb2 - dot
    if denom <= 0:
        return 0.0
    return dot / denom


def tanimoto_similarity_matrix(
    candidate_features: np.ndarray,
    reference_features: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Tanimoto similarity: each candidate vs each reference.

    Uses T(a,b) = dot(a,b) / (||a||^2 + ||b||^2 - dot(a,b)) with broadcasting.

    Parameters
    ----------
    candidate_features : np.ndarray
        Shape (n_candidates, n_features).
    reference_features : np.ndarray
        Shape (n_reference, n_features).

    Returns
    -------
    np.ndarray
        Shape (n_candidates, n_reference). Entry [i, j] = Tanimoto(candidate_i, reference_j).
    """
    # (n_cand, n_ref)
    dots = candidate_features @ reference_features.T
    # (n_cand,) and (n_ref,)
    norm_c2 = np.sum(candidate_features * candidate_features, axis=1)
    norm_r2 = np.sum(reference_features * reference_features, axis=1)
    # (n_cand, n_ref)
    denom = norm_c2[:, np.newaxis] + norm_r2[np.newaxis, :] - dots
    return np.where(denom > 0, dots / denom, 0.0)


def max_tanimoto_to_reference(
    candidate_features: np.ndarray,
    reference_features: np.ndarray,
) -> np.ndarray:
    """
    For each candidate, maximum Tanimoto similarity to any reference point.

    Parameters
    ----------
    candidate_features : np.ndarray
        Shape (n_candidates, n_features).
    reference_features : np.ndarray
        Shape (n_reference, n_features).

    Returns
    -------
    np.ndarray
        Shape (n_candidates,). Max similarity per candidate.
    """
    mat = tanimoto_similarity_matrix(candidate_features, reference_features)
    return np.max(mat, axis=1)


def filter_candidates_by_tanimoto(
    candidate_features: np.ndarray,
    reference_features: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Boolean mask of candidates to keep (max similarity to reference < threshold).

    Parameters
    ----------
    candidate_features : np.ndarray
        Shape (n_candidates, n_features).
    reference_features : np.ndarray
        Shape (n_reference, n_features).
    threshold : float
        Candidates with max Tanimoto >= threshold are excluded (mask False).

    Returns
    -------
    np.ndarray
        Shape (n_candidates,) bool. True = keep.
    """
    max_sim = max_tanimoto_to_reference(candidate_features, reference_features)
    return max_sim < threshold
