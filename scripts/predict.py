#!/usr/bin/env python3
"""
Predict mean and uncertainty for HAT or rebound over the design space.

Design space: initial = CSV only (candidates_initial.csv). Post-functionalization = both
initial CSV and JSON dir (featurization/) so the full space is old + new. Use --skip-from
to exclude already-initiated. Run folder layout: runs/g<N>/labeled.csv, models/, etc.

Usage:
  # Initial design space (CSV only):
  python predict.py --target HAT --model-dir runs/g0/models/HAT \\
    --design-space-csv data/candidates_initial.csv --out runs/g0/predictions_HAT.csv

  # Post-functionalization (pass both: initial CSV then JSON dir):
  python predict.py --target HAT --model-dir runs/g7/models/HAT \\
    --design-space-csv data/candidates_initial.csv --json-dir data/featurization \\
    --skip-from runs/g1/batch_next.csv ... runs/g6/batch_next.csv \\
    --out runs/g7/predictions_HAT.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_learning.constants import INVARIANT_FEATURES
from active_learning.data.design_space import load_design_space_batches
from active_learning.features import get_feature_columns
from active_learning.surrogate.model import add_ones_to_input
from active_learning.surrogate.uncertainty import propagate_uncertainty_last_layer


def load_names_to_skip(paths):
    """Load union of 'name' column from each CSV."""
    names = set()
    for path in paths or []:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if "name" in df.columns:
                names.update(df["name"].astype(str).tolist())
    return names


def main():
    p = argparse.ArgumentParser(description="Predict HAT or rebound with uncertainty")
    p.add_argument("--target", choices=["HAT", "rebound"], required=True)
    p.add_argument("--model-dir", required=True, help="Directory with nn_*.h5 and inv_hessian_and_scale_factor.npz")
    p.add_argument("--design-space-csv", default=None, help="CSV with design space features")
    p.add_argument("--json-dir", default=None, help="Directory of features_*.json")
    p.add_argument("--skip-from", nargs="*", default=[], help="CSVs whose 'name' to skip")
    p.add_argument("--out", required=True, help="Output CSV: name, json_idx, pred, std")
    p.add_argument("--batch-size", type=int, default=10_000)
    args = p.parse_args()

    model_path = os.path.join(args.model_dir, f"nn_{args.target}.h5")
    npz_path = os.path.join(args.model_dir, "inv_hessian_and_scale_factor.npz")
    if not os.path.isfile(model_path) or not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Model or uncertainty data not found in {args.model_dir}")

    model = keras.models.load_model(
        model_path,
        safe_mode=False,
        custom_objects={"add_ones_to_input": add_ones_to_input},
    )
    la_data = np.load(npz_path)
    inv_hessian = tf.convert_to_tensor(la_data["inv_hessian"], dtype=tf.float32)
    scale_factor = float(la_data["scale_factor"])

    names_to_skip = load_names_to_skip(args.skip_from)
    if os.path.exists(args.out):
        os.remove(args.out)
    first_batch = True

    for batch, json_idx_val in tqdm(
        load_design_space_batches(
            csv_path=args.design_space_csv,
            json_dir=args.json_dir,
            batch_size=args.batch_size,
            names_to_skip=names_to_skip,
        )
    ):
        features = get_feature_columns(batch, include_f_racs=True, exclude_invariant=True)
        features = [f for f in features if f in batch.columns]
        if not features:
            continue
        X = batch[features].values
        pred = model.predict(X, verbose=0)
        std_errors = propagate_uncertainty_last_layer(
            model, X, inv_hessian, scale_factor=scale_factor
        )
        json_idx_col = [""] * len(batch) if json_idx_val == "" else [json_idx_val] * len(batch)
        out_df = pd.DataFrame({
            "name": batch["name"].values,
            "json_idx": json_idx_col,
            "pred": pred.flatten(),
            "std": std_errors.flatten(),
        })
        out_df.to_csv(args.out, index=False, mode="a", header=first_batch)
        first_batch = False

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
