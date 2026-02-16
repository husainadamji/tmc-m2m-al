"""MLP surrogate model with optional batch norm and dropout."""

from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow import keras


def add_ones_to_input(x: tf.Tensor) -> tf.Tensor:
    """Append a column of ones for bias (used when use_bias=False in Dense)."""
    return tf.concat([x, tf.ones((tf.shape(x)[0], 1))], axis=1)


def build_mlp(hyperparams: Dict[str, Any]) -> keras.Model:
    """
    Build MLP with optional batch norm and dropout.

    Expects hyperparams: hidden_layers (list of int), activation (str),
    batch_norm (bool), dropout_rate (float).
    """
    model = keras.models.Sequential(name="MultiLayerPerceptron")
    for i, n_units in enumerate(hyperparams["hidden_layers"]):
        model.add(keras.layers.Lambda(add_ones_to_input))
        model.add(
            keras.layers.Dense(
                n_units,
                activation=hyperparams["activation"],
                use_bias=False,
                name=f"hidden_{i}",
            )
        )
        if hyperparams.get("batch_norm", False):
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(hyperparams["dropout_rate"]))
    model.add(keras.layers.Lambda(add_ones_to_input))
    model.add(keras.layers.Dense(1, use_bias=False, name="output"))
    return model


def build_model(
    input_scaler: keras.layers.Layer,
    output_scaler: keras.layers.Layer,
    hyperparams: Optional[Dict[str, Any]] = None,
    n_features: Optional[int] = None,
    name: Optional[str] = None,
) -> keras.Model:
    """
    Full regression model: input -> scale -> MLP -> output scale.

    Parameters
    ----------
    input_scaler : keras Layer (e.g. Normalization)
        Adapted to training data.
    output_scaler : keras Layer (e.g. Normalization(invert=True))
        Adapted to training targets.
    hyperparams : dict, optional
        Passed to build_mlp. If None, a default MLP is built.
    n_features : int, optional
        Input dimension (required if hyperparams is None).
    name : str, optional
        Model name.

    Returns
    -------
    model : keras.Model
    """
    if hyperparams is None:
        hyperparams = {
            "hidden_layers": [128, 128],
            "activation": "tanh",
            "batch_norm": False,
            "dropout_rate": 0.2,
        }
    if n_features is None:
        raise ValueError("n_features required when hyperparams is not None")
    inp = keras.layers.Input(shape=(n_features,), name="input")
    scaled = input_scaler(inp)
    out = output_scaler(build_mlp(hyperparams)(scaled))
    return keras.Model(inputs=inp, outputs=out, name=name)
