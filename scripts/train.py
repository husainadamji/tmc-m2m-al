#!/usr/bin/env python3
"""
Train or retrain HAT or rebound surrogate model.

Modes:
  initial  - Hyperparameter search (Keras Tuner), then train final model,
              compute last-layer inverse FIM, calibrate uncertainty, save model and artifacts.
  retrain  - Load existing model and uncertainty artifacts, retrain on new data,
              recompute FIM and calibration, save to a new model dir.

Usage (run folder layout: runs/g0/, runs/g1/, ...):
  # Initial training (run g0; no f_RACs by default)
  python train.py --mode initial --target HAT --data-csv runs/g0/labeled.csv \\
    --model-dir runs/g0/models/HAT

  # Initial with f_RACs (e.g. after design-space expansion and re-featurization)
  python train.py --mode initial --target HAT --data-csv runs/g4/labeled.csv \\
    --model-dir runs/g5/models/HAT --include-f-racs

  # Retrain (e.g. run g7; load from g6; same feature set)
  python train.py --mode retrain --target HAT --data-csv runs/g7/labeled.csv \\
    --model-dir runs/g7/models/HAT --load-from runs/g6/models/HAT
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_learning.features import get_feature_columns
from active_learning.constants import TARGET_HAT, TARGET_REBOUND
from active_learning.surrogate.model import add_ones_to_input, build_model, build_mlp
from active_learning.surrogate.uncertainty import (
    compute_inv_fim_last_layer,
    propagate_uncertainty_last_layer,
    error_calibration,
)


def _get_target_col(target: str) -> str:
    return TARGET_HAT if target.upper() == "HAT" else TARGET_REBOUND


def run_initial(
    data_path: str,
    model_dir: str,
    target: str,
    max_trials: int = 200,
    epochs: int = 4000,
    patience: int = 100,
    include_f_racs: bool = False,
):
    """Hyperopt + train + uncertainty + save."""
    import keras_tuner as kt

    df = pd.read_csv(data_path)
    target_col = _get_target_col(target)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in {list(df.columns)}")

    # Initial training: default no f_RACs (small design space). Use include_f_racs=True after design-space expansion.
    features = get_feature_columns(
        df, include_f_racs=include_f_racs, exclude_invariant=include_f_racs
    )
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=[target_col])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    df_train, df_val = train_test_split(df_train, test_size=0.125, random_state=0)

    X_train = df_train[features].values
    X_val = df_val[features].values
    X_test = df_test[features].values
    y_train = df_train[target_col].values.reshape(-1, 1)
    y_val = df_val[target_col].values.reshape(-1, 1)
    y_test = df_test[target_col].values.reshape(-1, 1)

    input_scaler = keras.layers.Normalization(name="input_norm")
    input_scaler.adapt(X_train)
    output_scaler = keras.layers.Normalization(invert=True, name="output_norm")
    output_scaler.adapt(y_train)

    def hypermodel_fn(hp):
        tf.keras.backend.clear_session()
        hyperparams = {
            "hidden_layers": [
                hp.Int(f"units_{i}", min_value=64, max_value=256, step=32)
                for i in range(hp.Int("num_layers", 1, 3))
            ],
            "activation": hp.Choice("activation", values=["tanh", "sigmoid"]),
            "batch_norm": hp.Boolean("batch_norm"),
            "dropout_rate": hp.Float("dropout_rate", 0, 0.5, step=0.1),
            "learning_rate": hp.Float("learning_rate", 1e-5, 1e-3, sampling="log"),
            "weight_decay": hp.Float("weight_decay", 1e-5, 1e-3, sampling="log"),
            "batch_size": hp.Int("batch_size", min_value=32, max_value=256, step=32),
        }
        model = build_model(
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            hyperparams=hyperparams,
            n_features=len(features),
        )
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hyperparams["learning_rate"],
                weight_decay=hyperparams["weight_decay"],
            ),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanSquaredError()],
        )
        return model

    os.makedirs(model_dir, exist_ok=True)
    kt_dir = os.path.join(model_dir, "kt_logs")
    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: hypermodel_fn(hp),
        objective="val_loss",
        max_trials=max_trials,
        seed=0,
        directory=kt_dir,
        project_name=f"{target}_hyperopt",
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stop],
        verbose=0,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    with open(os.path.join(model_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_hp.values, f)

    tf.keras.backend.clear_session()
    hidden_layers = [best_hp.get(f"units_{i}") for i in range(best_hp.get("num_layers"))]
    hyperparams = {
        "hidden_layers": hidden_layers,
        "activation": best_hp.get("activation"),
        "batch_norm": best_hp.get("batch_norm"),
        "dropout_rate": best_hp.get("dropout_rate"),
    }
    model = build_model(
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        hyperparams=hyperparams,
        n_features=len(features),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=best_hp.get("learning_rate"),
            weight_decay=best_hp.get("weight_decay"),
        ),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()],
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=best_hp.get("batch_size"),
        epochs=epochs,
        callbacks=[early_stop],
        verbose=0,
    )
    _plot_loss_curves(history, model_dir)

    _save_artifacts(
        model_dir=model_dir,
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        features=features,
        target=target,
    )
    return model


def run_retrain(
    data_path: str,
    model_dir: str,
    load_from: str,
    target: str,
    learning_rate: float = 1e-5,
    epochs: int = 4000,
    patience: int = 100,
):
    """Load model, retrain on new data, recompute uncertainty, save."""
    df = pd.read_csv(data_path)
    target_col = _get_target_col(target)
    features = get_feature_columns(df, include_f_racs=True, exclude_invariant=True)
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=[target_col])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    df_train, df_val = train_test_split(df_train, test_size=0.125, random_state=0)

    X_train = df_train[features].values
    X_val = df_val[features].values
    X_test = df_test[features].values
    y_train = df_train[target_col].values.reshape(-1, 1)
    y_val = df_val[target_col].values.reshape(-1, 1)
    y_test = df_test[target_col].values.reshape(-1, 1)

    model_path = os.path.join(load_from, f"nn_{target}.h5")
    model = keras.models.load_model(
        model_path,
        safe_mode=False,
        custom_objects={"add_ones_to_input": add_ones_to_input},
    )
    hp_path = os.path.join(load_from, "best_hyperparameters.json")
    if os.path.isfile(hp_path):
        with open(hp_path) as f:
            best_hp = json.load(f)
    else:
        # Previous iteration may have been a retrain that didn't write hyperparameters
        best_hp = {"batch_size": 64}
    batch_size = best_hp.get("batch_size", 64)

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()],
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=0,
    )
    os.makedirs(model_dir, exist_ok=True)
    # Persist hyperparameters so next retrain (or predict) can load them; may be missing if load_from was a retrain
    with open(os.path.join(model_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_hp, f)
    _plot_loss_curves(history, model_dir)

    _save_artifacts(
        model_dir=model_dir,
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        features=features,
        target=target,
    )
    return model


def _plot_loss_curves(history, model_dir: str) -> None:
    """Plot training and validation loss vs epoch and save to model_dir."""
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], color="red", linestyle="-", label="training loss")
    ax.plot(history.history["val_loss"], color="blue", linestyle="-", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (kcal/mol)")
    ax.set_title("Train and Validation Loss Curves")
    ax.legend()
    fig.savefig(os.path.join(model_dir, "loss_curves.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_artifacts(
    model_dir: str,
    model: keras.Model,
    X_train, y_train, X_val, y_val, X_test, y_test,
    features: list,
    target: str,
):
    """Save model, FIM, scale factor, metrics, and parity plots."""
    train_pred = model.predict(X_train, verbose=0)
    val_pred = model.predict(X_val, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    inv_hessian = compute_inv_fim_last_layer(model, X_train, y_train)
    val_std = propagate_uncertainty_last_layer(model, X_val, inv_hessian)
    alpha = error_calibration(val_std, val_pred.flatten(), y_val.flatten())
    np.savez(
        os.path.join(model_dir, "inv_hessian_and_scale_factor.npz"),
        inv_hessian=inv_hessian.numpy(),
        scale_factor=alpha,
    )
    train_std = propagate_uncertainty_last_layer(model, X_train, inv_hessian, scale_factor=alpha)
    val_std = propagate_uncertainty_last_layer(model, X_val, inv_hessian, scale_factor=alpha)
    test_std = propagate_uncertainty_last_layer(model, X_test, inv_hessian, scale_factor=alpha)

    model.save(os.path.join(model_dir, f"nn_{target}.h5"))

    metrics = []
    splits = [
        ("train", X_train, y_train, train_pred, train_std, "y_train", "pred_train", "train_std", "train_data_and_errors.csv"),
        ("val", X_val, y_val, val_pred, val_std, "y_val", "pred_val", "val_std", "val_data_and_errors.csv"),
        ("test", X_test, y_test, test_pred, test_std, "y_test", "pred_test", "test_std", "test_data_and_errors.csv"),
    ]
    for name, X, y_true, y_pred, std, y_col, pred_col, std_col, csv_name in splits:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        std = std.flatten()
        metrics.append({
            "split": name,
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        })
        df_ = pd.DataFrame(X, columns=features)
        df_[y_col] = y_true
        df_[pred_col] = y_pred
        df_[std_col] = std
        df_.to_csv(os.path.join(model_dir, csv_name), index=False)

    with open(os.path.join(model_dir, f"{target}_model_performance.txt"), "w") as f:
        for m in metrics:
            f.write(f"{m['split']} MAE = {m['MAE']} kcal/mol, R^2 = {m['R2']}, MAPE = {m['MAPE']} %, RMSE = {m['RMSE']} kcal/mol\n")

    # Parity plot with error bars
    fig, ax = plt.subplots()
    ax.errorbar(
        y_train.flatten(),
        train_pred.flatten(),
        yerr=train_std.flatten(),
        linestyle="None",
        marker="o",
        alpha=0.5,
        label="train",
        elinewidth=0.75,
        color="red",
    )
    ax.errorbar(
        y_val.flatten(),
        val_pred.flatten(),
        yerr=val_std.flatten(),
        linestyle="None",
        marker="o",
        alpha=0.5,
        label="val",
        elinewidth=0.75,
        color="blue",
    )
    ax.errorbar(
        y_test.flatten(),
        test_pred.flatten(),
        yerr=test_std.flatten(),
        linestyle="None",
        marker="o",
        alpha=0.5,
        label="test",
        elinewidth=0.75,
        color="green",
    )
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "0.5")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend()
    ax.set_xlabel("DFT (kcal/mol)")
    ax.set_ylabel("ML prediction (kcal/mol)")
    ax.set_title(f"{target} (kcal/mol)")
    fig.savefig(os.path.join(model_dir, f"{target}_parity_w_error_bars.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["initial", "retrain"], required=True)
    p.add_argument("--target", choices=["HAT", "rebound"], required=True)
    p.add_argument("--data-csv", required=True, help="Labeled feature CSV with target column")
    p.add_argument("--model-dir", required=True, help="Where to save model and artifacts")
    p.add_argument("--load-from", default=None, help="For retrain: directory of model to load")
    p.add_argument("--target-col", default=None, help="Target column name (default: HAT or rebound (kcal/mol))")
    p.add_argument("--max-trials", type=int, default=200)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-5, help="For retrain only")
    p.add_argument("--include-f-racs", action="store_true", help="Initial only: use f_RACs and exclude invariant columns (e.g. after design-space expansion)")
    args = p.parse_args()

    if args.mode == "retrain" and not args.load_from:
        p.error("--load-from required for retrain")

    if args.mode == "initial":
        run_initial(
            args.data_csv,
            args.model_dir,
            args.target,
            max_trials=args.max_trials,
            epochs=args.epochs,
            patience=args.patience,
            include_f_racs=args.include_f_racs,
        )
    else:
        run_retrain(
            args.data_csv,
            args.model_dir,
            args.load_from,
            args.target,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
        )
    print(f"Done. Model and artifacts in {args.model_dir}")


if __name__ == "__main__":
    main()
