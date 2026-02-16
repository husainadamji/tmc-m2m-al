#!/usr/bin/env python3
"""
Compute 2D Pareto front (min HAT, min |rebound|) from labeled data and save NPZ for EHVI.

Output NPZ contains pareto_points and r (reference point). Users can plot as they like.

Usage (run folder layout):
  python compute_pareto.py --data-csv runs/g7/labeled.csv --out-npz runs/g7/pareto.npz
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_learning.acquisition.ehvi import get_2D_pareto_indices_min_min
from active_learning.constants import TARGET_HAT, TARGET_REBOUND

# Offset for reference point r (used in EHVI)
REFERENCE_POINT_EPSILON = 10.0


def main():
    p = argparse.ArgumentParser(description="Compute Pareto front from labeled data; save NPZ for EHVI.")
    p.add_argument("--data-csv", required=True, help="CSV with HAT and rebound columns")
    p.add_argument("--out-npz", required=True, help="Output NPZ: pareto_points, r")
    p.add_argument("--hat-col", default=TARGET_HAT)
    p.add_argument("--rebound-col", default=TARGET_REBOUND)
    args = p.parse_args()

    df = pd.read_csv(args.data_csv)
    if args.hat_col not in df.columns or args.rebound_col not in df.columns:
        raise ValueError(f"Need columns {args.hat_col} and {args.rebound_col}")

    HAT = df[args.hat_col].values
    abs_rebound = np.abs(df[args.rebound_col].values)
    points = np.column_stack((HAT, abs_rebound))

    pareto_indices = get_2D_pareto_indices_min_min(points)
    pareto_points = points[pareto_indices]
    r = np.array([
        np.max(points[:, 0]) + REFERENCE_POINT_EPSILON,
        np.max(points[:, 1]) + REFERENCE_POINT_EPSILON,
    ])
    np.savez(args.out_npz, pareto_points=pareto_points, r=r)
    print(f"Saved {args.out_npz} with {len(pareto_points)} Pareto points")


if __name__ == "__main__":
    main()
