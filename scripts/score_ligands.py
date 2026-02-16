#!/usr/bin/env python3
"""
Add SAScore and SCScore to a DataFrame of functionalized ligands (mol2 strings).

Uses active_learning.functionalization. Optional deps: RDKit, molSimplify,
SA_Score (RDKit Contrib), SCScore model. Set paths via env or CLI.

Usage:
  python score_ligands.py --csv data/ligands_functionalized.csv \\
    --out data/ligands_functionalized_scored.csv \\
    --sc-model /path/to/model.ckpt.as_numpy.pickle
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_learning.functionalization import add_scores_to_dataframe


def main():
    p = argparse.ArgumentParser(description="Add SAScore and SCScore to ligand CSV")
    p.add_argument("--csv", required=True, help="Input CSV with mol2 column")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--mol2-col", default="mol2_w_metal_functionalized", help="Column with mol2 content")
    p.add_argument("--sc-model", default=None, help="Path to SCScore model (.as_numpy.pickle). If not set, only SAScore is computed.")
    p.add_argument("--name-col", default="csd_ligand")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out = add_scores_to_dataframe(
        df,
        mol2_col=args.mol2_col,
        sc_model_path=args.sc_model,
        name_col=args.name_col,
    )
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with ligand_SAScore and ligand_SCScore columns.")


if __name__ == "__main__":
    main()
