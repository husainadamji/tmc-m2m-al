#!/usr/bin/env python3
"""
Compute 2D EHVI, take top-K by EHVI, then K-medoids to select a diverse batch.

Reads HAT and rebound prediction CSVs (with name, json_idx, pred, std), merges them,
loads Pareto data, optionally screens by Tanimoto similarity to a reference set,
computes EHVI, selects top_k, attaches features, runs K-medoids, and writes the batch CSV.

Optional: --reference-csv and --tanimoto-threshold exclude candidates whose max
Tanimoto (over RACs) to any reference point is >= threshold before EHVI.

Usage (run folder layout: runs/g<N>/):
  # Initial design space (CSV only):
  python select_batch.py --hat-pred ... --rebound-pred ... --pareto-npz ... \\
    --features-csv data/candidates_initial.csv --out-batch runs/g0/batch_next.csv

  # Post-functionalization (pass both: initial CSV + JSON dir):
  python select_batch.py --hat-pred ... --rebound-pred ... --pareto-npz ... \\
    --features-csv data/candidates_initial.csv --json-dir data/featurization \\
    --out-batch runs/g7/batch_next.csv

  # With Tanimoto screening:
  python select_batch.py ... --reference-csv runs/g7/failed.csv --tanimoto-threshold 0.842
"""

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_learning.acquisition.ehvi import get_2D_EHVI_min_min
from active_learning.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EHVI_TOP_K,
    DEFAULT_N_CLUSTERS,
    DEFAULT_TANIMOTO_THRESHOLD,
)
from active_learning.features import get_feature_columns
from active_learning.similarity import filter_candidates_by_tanimoto


def load_features_for_candidates(
    df: pd.DataFrame,
    features: list,
    old_ds: Optional[pd.DataFrame],
    json_dir: Optional[str],
) -> np.ndarray:
    """Fill feature matrix for rows in df; use old_ds for json_idx=='' or no json_dir, else load from JSON."""
    name_to_idx = {}
    old_X = np.zeros((0, len(features)))
    if old_ds is not None:
        old_names = old_ds["name"].values
        old_X = old_ds[features].values
        name_to_idx = {n: i for i, n in enumerate(old_names)}
    n_entries = len(df)
    n_feat = len(features)
    X = np.zeros((n_entries, n_feat))
    for pos, (_, row) in enumerate(tqdm(df.iterrows(), total=n_entries)):
        name = row["name"]
        json_idx = row.get("json_idx", "")
        if pd.isna(json_idx) or str(json_idx).strip() == "" or not json_dir:
            idx = name_to_idx.get(name)
            if idx is not None:
                X[pos, :] = old_X[idx]
            continue
        path = os.path.join(json_dir, f"features_{int(float(json_idx))}.json")
        with open(path) as f:
            data = json.load(f)
        for d in data.get("features", []):
            if d["name"] == name:
                X[pos, :] = [d.get(f, np.nan) for f in features]
                break
    return X


def main():
    p = argparse.ArgumentParser(description="EHVI + K-medoids batch selection")
    p.add_argument("--hat-pred", required=True, help="HAT predictions CSV (name, json_idx, pred, std)")
    p.add_argument("--rebound-pred", required=True, help="Rebound predictions CSV")
    p.add_argument("--pareto-npz", required=True, help="NPZ with pareto_points and r")
    p.add_argument("--features-csv", required=True, help="Design space CSV for feature lookup (initial candidates; pass with --json-dir when also using functionalized space)")
    p.add_argument("--json-dir", default=None, help="Directory of features_*.json (optional; pass with --features-csv when design space includes functionalized candidates)")
    p.add_argument("--out-batch", required=True, help="Output CSV for selected batch")
    p.add_argument("--top-k", type=int, default=DEFAULT_EHVI_TOP_K)
    p.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--out-ehvi-csv", default=None, help="Optional: save full EHVI-ranked CSV")
    p.add_argument("--reference-csv", default=None, help="Optional: CSV of reference points (same features); enables Tanimoto screening")
    p.add_argument("--tanimoto-threshold", type=float, default=DEFAULT_TANIMOTO_THRESHOLD, help="Max Tanimoto to reference to exclude (default: %(default)s)")
    args = p.parse_args()

    old_ds = pd.read_csv(args.features_csv)
    features = get_feature_columns(old_ds, include_f_racs=True, exclude_invariant=True)
    features = [f for f in features if f in old_ds.columns]

    if args.json_dir:
        for fname in sorted(os.listdir(args.json_dir)):
            if not fname.startswith("features_") or not fname.endswith(".json"):
                continue
            with open(os.path.join(args.json_dir, fname)) as f:
                data = json.load(f)
            feats_list = data.get("features", [])
            if not feats_list:
                continue
            json_keys = set(feats_list[0].keys())
            missing = [f for f in features if f not in json_keys]
            if missing:
                raise ValueError(
                    f"Feature list from CSV does not match JSON: missing in {fname}: {missing[:10]}{'...' if len(missing) > 10 else ''}"
                )
            break

    hat_df = pd.read_csv(args.hat_pred)
    reb_df = pd.read_csv(args.rebound_pred)
    assert list(hat_df["name"].values) == list(reb_df["name"].values), "HAT and rebound names must match"
    assert list(hat_df["json_idx"].astype(str)) == list(reb_df["json_idx"].astype(str)), "json_idx must match"

    df = pd.DataFrame({
        "name": hat_df["name"],
        "json_idx": hat_df["json_idx"],
        "predicted_HAT (kcal/mol)": hat_df["pred"].values,
        "HAT_std_errors (kcal/mol)": hat_df["std"].values,
        "predicted_rebound (kcal/mol)": reb_df["pred"].values,
        "rebound_std_errors (kcal/mol)": reb_df["std"].values,
    })

    use_tanimoto = args.reference_csv is not None
    X_filtered = None
    if use_tanimoto:
        ref_df = pd.read_csv(args.reference_csv)
        features = [f for f in features if f in ref_df.columns]
        if not features:
            raise ValueError("Reference CSV has no overlapping feature columns with design space.")
        X_all = load_features_for_candidates(df, features, old_ds, args.json_dir)
        X_ref = ref_df[features].values.astype(np.float64)
        mask = filter_candidates_by_tanimoto(X_all, X_ref, args.tanimoto_threshold)
        n_before, n_after = len(df), int(mask.sum())
        print(f"Tanimoto screening: {n_after} / {n_before} candidates retained (threshold={args.tanimoto_threshold})")
        df = df.loc[mask].reset_index(drop=True)
        X_filtered = X_all[mask]

    pareto_data = np.load(args.pareto_npz)
    pareto_points = pareto_data["pareto_points"]
    r = pareto_data["r"]

    ehvi_list = []
    for i in range(0, len(df), args.batch_size):
        batch = df.iloc[i : i + args.batch_size]
        pred_mean = np.column_stack((
            batch["predicted_HAT (kcal/mol)"].values,
            np.abs(batch["predicted_rebound (kcal/mol)"].values),
        ))
        pred_std = np.column_stack((
            batch["HAT_std_errors (kcal/mol)"].values,
            batch["rebound_std_errors (kcal/mol)"].values,
        ))
        ehvi = get_2D_EHVI_min_min(pred_mean, pred_std, pareto_points, r)
        ehvi_list.append(ehvi)
    df["2D_EHVI"] = np.concatenate(ehvi_list)

    df_sorted = df.sort_values(by="2D_EHVI", ascending=False)
    top_df = df_sorted.head(args.top_k)

    if args.out_ehvi_csv:
        top_df.to_csv(args.out_ehvi_csv, index=False)

    if use_tanimoto:
        X_top = X_filtered[top_df.index]
    else:
        X_top = load_features_for_candidates(top_df, features, old_ds, args.json_dir)
    top_df = top_df.reset_index(drop=True)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_top)
    kmedoids = KMedoids(n_clusters=args.n_clusters, init="k-medoids++", random_state=42)
    kmedoids.fit(X_norm)
    medoid_indices = kmedoids.medoid_indices_
    batch_df = top_df.iloc[medoid_indices].reset_index(drop=True)
    batch_df.to_csv(args.out_batch, index=False)
    print(f"Wrote {args.out_batch} ({len(batch_df)} candidates)")


if __name__ == "__main__":
    main()
