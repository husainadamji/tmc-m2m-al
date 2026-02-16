# TMC-M2M-AL: Multi-Objective Active Learning for Methane-to-Methanol Catalysis

This repository implements a **multi-objective Bayesian active learning** pipeline for exploring **transition-metal complex (TMC)** design spaces toward **methane-to-methanol (M2M)** catalysis. The method uses **2D Expected Hypervolume Improvement (EHVI)** with neural network surrogates to select candidates for DFT evaluations on two objectives:

- **HAT** – Hydrogen atom transfer barrier ΔE(HAT) (minimize)
- **Rebound** – |ΔE(rebound)| (minimize)

## Installation

```bash
git clone <repo-url>
cd tmc-m2m-al
pip install -r requirements.txt
```

Run all script commands from the repository root (e.g. `python scripts/train.py ...`).

## Project structure

```
tmc-m2m-al/
├── active_learning/           # Python package
│   ├── __init__.py
│   ├── constants.py          # Feature names, invariant RACs, defaults
│   ├── features.py           # get_feature_columns(), get_feature_matrix()
│   ├── acquisition/           # EHVI and Pareto indices
│   │   └── ehvi.py            # 2D EHVI (min-min, min-max), Pareto indices
│   ├── similarity.py         # Tanimoto on RACs (vectorized); filter by reference set
│   ├── surrogate/             # NN surrogates and uncertainty
│   │   ├── model.py           # MLP builder, add_ones_to_input
│   │   ├── uncertainty.py     # Last-layer Laplace (FIM, calibration)
│   ├── data/                  # Design space loading
│   │   └── design_space.py    # Iterate CSV + JSON featurization batches
│   └── functionalization/     # Optional: ligand functionalization and scores
│       ├── scores.py          # SAScore, SCScore (add_scores_to_dataframe)
│       └── mol2_utils.py      # mol2 metal stripping (molSimplify)
├── scripts/                   # CLI entry points
│   ├── train.py              # Train or retrain HAT/rebound model
│   ├── predict.py            # Predict mean + uncertainty over design space
│   ├── select_batch.py       # EHVI + top-K + K-medoids → next batch CSV (optional Tanimoto pre-screen)
│   ├── compute_pareto.py     # Pareto front + reference point (NPZ for EHVI)
│   ├── featurize.py          # Featurize design space (--mode initial | functionalized)
│   ├── functionalize.py      # Tetradentate + monodentate functionalization (optional deps)
│   └── score_ligands.py      # Add SAScore/SCScore to ligand CSV (uses package)
├── config.example.yaml
├── requirements.txt
└── README.md
```

## Run layout: one folder per run

Use **one folder per run** (one run = all data through generation N). Each run folder holds that run's labeled data, Pareto, models, predictions, and the next batch to evaluate.

- **Run folder:** `runs/g<N>/` (e.g. `runs/g0/`, `runs/g7/`). Generation N means you have N+1 batches of labeled data (gen 0 through N).
- **Inside a run folder:**
  - `labeled.csv` — cumulative labeled data through generation N (features + `HAT (kcal/mol)` + `rebound (kcal/mol)`).
  - `pareto.npz` — Pareto front from `labeled.csv` (for EHVI).
  - `models/HAT/`, `models/rebound/` — trained surrogates and uncertainty artifacts.
  - `predictions_HAT.csv`, `predictions_rebound.csv` — predictions over the design space from this run's models.
  - `batch_next.csv` — selected candidates for the next DFT round (run DFT, merge labels, then create the next run folder `runs/g<N+1>/`).

Shared inputs (same for all runs):

- **Design space:** Initial: `data/candidates_initial.csv`. Post-functionalization: pass **both** that CSV and **`data/featurization/`** (directory of `features_<N>.json`) so the full space is initial + functionalized. Same feature schema as labeled data. JSONs from **`featurize.py --mode functionalized`** (see **Featurization** below).

Example layout:

```
runs/
  g0/
    labeled.csv              # gen 0 labeled data
    pareto.npz
    models/HAT/              # nn_HAT.h5, inv_hessian_and_scale_factor.npz, ...
    models/rebound/
    predictions_HAT.csv
    predictions_rebound.csv
    batch_next.csv           # candidates for gen 1 (run DFT → then create g1)
  g1/
    labeled.csv              # cumulative through gen 1
    pareto.npz
    models/HAT/
    models/rebound/
    ...
data/
  candidates_initial.csv     # design space pre-expansion (from featurize --mode initial)
  featurization/             # design space post-expansion: features_0.json, features_1.json, ... (no combined CSV)
  tetra_ligands.csv          # optional: for functionalize.py
  axial_ligands/             # optional: monodentate .mol2 for functionalize.py
```

## Training

Use `scripts/train.py` for **initial** training (run g0 or post-expansion with f_RACs) and **retrain** (run g1, g2, ... with same feature set).

- **Initial (run g0):** train on `runs/g0/labeled.csv` without f_RACs; write to `runs/g0/models/HAT` and `runs/g0/models/rebound`.
- **Initial with f_RACs:** after design-space expansion and re-featurization, run `--mode initial --include-f-racs` on the cumulative labeled CSV to retrain from scratch with the full feature set (including f_RACs).
- **Retrain (run g<N>):** train on `runs/g<N>/labeled.csv`, load previous run's model dirs (same feature set), write to `runs/g<N>/models/`.

```bash
# Initial training (run g0)
python scripts/train.py --mode initial --target HAT --data-csv runs/g0/labeled.csv --model-dir runs/g0/models/HAT
python scripts/train.py --mode initial --target rebound --data-csv runs/g0/labeled.csv --model-dir runs/g0/models/rebound

# Retrain (e.g. run g7; load from g6)
python scripts/train.py --mode retrain --target HAT --data-csv runs/g7/labeled.csv --model-dir runs/g7/models/HAT --load-from runs/g6/models/HAT
python scripts/train.py --mode retrain --target rebound --data-csv runs/g7/labeled.csv --model-dir runs/g7/models/rebound --load-from runs/g6/models/rebound
```

Each `--model-dir` will contain `nn_HAT.h5` or `nn_rebound.h5`, `inv_hessian_and_scale_factor.npz`, `best_hyperparameters.json`, and performance/parity outputs.

## Workflow (one generation)

For run **g<N>** (you have labeled data through generation N):

1. **Pareto** from this run's labeled data:
   ```bash
   python scripts/compute_pareto.py --data-csv runs/g7/labeled.csv --out-npz runs/g7/pareto.npz
   ```

2. **Predict** over the design space (skip candidates already in previous batches):
   ```bash
   python scripts/predict.py --target HAT --model-dir runs/g7/models/HAT \
     --design-space-csv data/candidates_initial.csv --json-dir data/featurization \
     --skip-from runs/g1/batch_next.csv runs/g2/batch_next.csv runs/g3/batch_next.csv runs/g4/batch_next.csv runs/g5/batch_next.csv runs/g6/batch_next.csv \
     --out runs/g7/predictions_HAT.csv

   python scripts/predict.py --target rebound --model-dir runs/g7/models/rebound \
     --design-space-csv data/candidates_initial.csv --json-dir data/featurization \
     --skip-from runs/g1/batch_next.csv ... runs/g6/batch_next.csv \
     --out runs/g7/predictions_rebound.csv
   ```

3. **Select next batch** (EHVI + K-medoids). Optionally screen by Tanimoto similarity to a reference set (e.g. failed DFT) before EHVI:
   ```bash
   python scripts/select_batch.py \
     --hat-pred runs/g7/predictions_HAT.csv --rebound-pred runs/g7/predictions_rebound.csv \
     --pareto-npz runs/g7/pareto.npz \
     --features-csv data/candidates_initial.csv --json-dir data/featurization \
     --out-batch runs/g7/batch_next.csv
   # With Tanimoto screening (exclude candidates too similar to reference):
   python scripts/select_batch.py ... --reference-csv runs/g7/failed.csv --tanimoto-threshold 0.842
   ```

4. Run DFT on `runs/g7/batch_next.csv`, merge new labels into your feature table, and create **run g8**: put the cumulative labeled CSV at `runs/g8/labeled.csv`, then retrain and repeat from step 1 for g8.

## Data file naming (canonical)

All scripts and docs use these names for consistency:

- **Run folder:** `runs/g<N>/` with `labeled.csv`, `pareto.npz`, `models/HAT/`, `models/rebound/`, `predictions_HAT.csv`, `predictions_rebound.csv`, `batch_next.csv`.
- **Design space (pre-expansion):** `data/candidates_initial.csv` (from `featurize.py --mode initial`).
- **Design space (post-functionalization):** pass both `data/candidates_initial.csv` and `data/featurization/` (JSON dir) so the full space is initial + functionalized.
- **Featurization:** `data/featurization_failed.csv` (failed log, initial mode only), `data/tetra_ligands.csv`, `data/axial_ligands/`. Functionalized mode: failed are in each JSON.
- **Functionalization:** `data/ligands_functionalized.csv`, `data/ligands_functionalized_scored.csv`.

## Data files you need

### Active learning

| Purpose | What you provide | Required columns / structure |
|--------|-------------------|------------------------------|
| **Run folder (e.g. runs/g7/)** | Create it; put `labeled.csv` there (cumulative through gen N). | `labeled.csv`: `name`, feature columns, **`HAT (kcal/mol)`**, **`rebound (kcal/mol)`**. |
| **Design space** | Initial: `data/candidates_initial.csv` (CSV only). Post-functionalization: pass **both** that CSV and `data/featurization/` (JSON dir). | CSV: **`name`** + feature columns. JSON: each item has **`name`** and feature keys. |
| **Skip-from (predict)** | Previous runs' `batch_next.csv` (e.g. `runs/g1/batch_next.csv` ... `runs/g6/batch_next.csv`). | Column **`name`**. |
| **Models, Pareto** | Produced by `train.py` and `compute_pareto.py` inside the run folder. | — |

At **start**: create `runs/g0/` and put your first labeled CSV there as `runs/g0/labeled.csv`. Run initial training so `runs/g0/models/HAT` and `runs/g0/models/rebound` exist. Then run compute_pareto, predict, and select_batch to get `runs/g0/batch_next.csv`. After DFT and merging labels, create `runs/g1/labeled.csv` and retrain, etc.


### Functionalization

| Purpose | What you provide |
|--------|-------------------|
| **functionalize.py** | **`--axial-ligands-dir`**: directory of monodentate `.mol2` files (e.g. `data/axial_ligands/`). **`--tetra-csv`**: tetradentate input (e.g. `data/tetra_ligands.csv`). **`--out-csv`**: e.g. `data/ligands_functionalized.csv`. |
| **score_ligands.py** | CSV with mol2 column (e.g. output of functionalize.py). **`--out`**: e.g. `data/ligands_functionalized_scored.csv`. Optionally **`--sc-model`** for SCScore. |


## Design space functionalization

Outside the active learning loop, the design space can be **functionalized** (derive new ligands by adding functional groups) and annotated with **SAScore** and **SCScore**. This is supported in an optional, modular way.

- **`scripts/functionalize.py`** – Tetradentate and monodentate ligand functionalization. All paths via CLI (`--axial-ligands-dir`, `--tetra-csv`, `--out-csv`). For SAScore/SCScore, run `score_ligands.py` on the output CSV.

- **`active_learning.functionalization`** – Optional subpackage that provides:
  - **SAScore / SCScore**: `add_scores_to_dataframe(df, mol2_col=..., sc_model_path=...)` to add `ligand_SAScore` and `ligand_SCScore` columns; helpers `compute_sa_score(mol)` and `compute_sc_score(mol2_lig_str, model_path=...)`.
  - **Mol2 helpers**: `mol2_ligand_from_mol2_metal(mol2_str)` to strip the dummy metal and get the ligand mol2 string (uses molSimplify).

- **`scripts/score_ligands.py`** – Thin CLI that uses the package: reads a CSV with a mol2 column, adds SAScore and (if `--sc-model` is given) SCScore, writes the result. Use this to score an existing functionalized CSV without re-running the full functionalization.

  ```bash
  python scripts/score_ligands.py --csv data/ligands_functionalized.csv \
    --out data/ligands_functionalized_scored.csv \
    --sc-model /path/to/model.ckpt.as_numpy.pickle
  ```

Install optional deps (see `requirements-functionalization.txt`) and set paths for SA_Score and SCScore as needed. The core active learning package does not depend on them.

## Featurization (design space)

Before **predict** and **select_batch** can run, the design space must be featurized: each complex (metal, oxidation, spin, tetradentate, axial ligands) is turned into a feature vector (RACs, core features) that the surrogate models use.

**`scripts/featurize.py`** featurizes the design space in two modes:

- **`--mode initial`** — Pre-functionalization: smaller tetradentate set, base axial ligands only. Reads a tetra CSV with `csd_ligand`, `mol2_lig`, `mol2_dummy_core`, etc. Writes a single CSV (e.g. `data/candidates_initial.csv`). Use for the small design space before any functionalization.
- **`--mode functionalized`** — Post-functionalization: full tetradentate + functionalized axial set. Reads a tetra CSV with `new_ligand_name`, `mol2_lig_functionalized`, etc. Writes **`data/featurization/features_<idx>.json`** only (one per tetradentate index; no combined CSV). In `predict.py` and `select_batch.py`, pass **both** `--design-space-csv data/candidates_initial.csv` (or `--features-csv`) and **`--json-dir data/featurization`** so the full design space is initial + functionalized.

Run **initial** featurization when you have the original tetra CSV and axial mol2s; it writes one CSV and a failed log. Run **functionalized** after `functionalize.py` (and any merging); it writes only `features_<idx>.json` (no combined CSV; failed structures are stored in each JSON's `"failed"` key). Use `--tetra-csv`, `--axial-dir`, `--out-dir`, `--out-csv`, `--failed-csv` (initial only for the latter two). The output feature schema (e.g. `OHE_mn`, `OHE_fe`, `ox`, `spin`, `monocharge`, `tetracharge`, lc-*, lig_*, etc.) must match what `active_learning.features.get_feature_columns` and the rest of the pipeline expect.

**Workflow order:** Functionalization (optional) → **Featurization** (this script) → design space ready → Training (on labeled subset) → compute_pareto → predict → select_batch → DFT → repeat.


## Notes

- **Feature columns**: The package expects core columns `OHE_mn`, `OHE_fe`, `ox`, `spin`, `monocharge`, `tetracharge` (and lc-* / f-* RACs). If your data use different names (e.g. `axcharge`, `eqcharge`), rename columns or extend `active_learning.constants.CORE_FEATURE_NAMES` and pass a feature list where supported.
- **Entry points**: `train.py`, `predict.py`, `compute_pareto.py`, and `select_batch.py` are the main pipeline; `featurize.py`, `functionalize.py`, and `score_ligands.py` support design-space setup and optional functionalization.

## License

See repository license file.
