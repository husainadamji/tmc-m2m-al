"""
Feature column resolution from design-space or labeled DataFrames.
"""

from typing import List, Optional

import pandas as pd

from .constants import CORE_FEATURE_NAMES, INVARIANT_FEATURES


def get_feature_columns(
    df: pd.DataFrame,
    *,
    include_f_racs: bool = True,
    exclude_invariant: bool = True,
) -> List[str]:
    """
    Build the list of feature columns from a DataFrame.

    Includes: core features, lc-* (excluding lig), and optionally f-* RACs
    (excluding lig, lc, D_, mc). Optionally removes invariant columns.

    Parameters
    ----------
    df : pd.DataFrame
        Design space or labeled feature table.
    include_f_racs : bool
        If True, include columns matching f-* (full-complex RACs).
    exclude_invariant : bool
        If True, drop columns in INVARIANT_FEATURES.

    Returns
    -------
    List[str]
        Sorted list of feature column names present in df.
    """
    features: List[str] = []
    # Core (only those present)
    for name in CORE_FEATURE_NAMES:
        if name in df.columns:
            features.append(name)
    # lc-* but not lig
    lc = [n for n in df.columns if "lc-" in n and "lig" not in n]
    features.extend(lc)
    if include_f_racs:
        f_racs = [
            n
            for n in df.columns
            if "f-" in n and not any(s in n for s in ["lig", "lc", "D_", "mc"])
        ]
        features.extend(f_racs)
    if exclude_invariant:
        features = [f for f in features if f not in INVARIANT_FEATURES]
    return features


def get_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    **kwargs,
) -> tuple:
    """
    Extract feature matrix and ensure columns exist.

    Parameters
    ----------
    df : pd.DataFrame
        Table with feature columns.
    feature_columns : list, optional
        If provided, use these; otherwise call get_feature_columns(df, **kwargs).
    **kwargs
        Passed to get_feature_columns if feature_columns is None.

    Returns
    -------
    X : np.ndarray
        Shape (n_samples, n_features).
    columns : list
        Feature column names used.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(df, **kwargs)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[feature_columns].values, feature_columns
