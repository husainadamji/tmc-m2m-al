"""
Uncertainty quantification via last-layer Laplace approximation (inverse Fisher
Information Matrix).
"""

import numpy as np
import tensorflow as tf


def compute_inv_fim_last_layer(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    regularization: float = 1e-2,
) -> tf.Tensor:
    """
    Inverse Fisher information matrix for the last (output) layer.

    Used to propagate uncertainty for predictions. Assumes MSE loss.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    X : np.ndarray
        Input features (e.g. training set).
    y : np.ndarray
        Targets, shape (n, 1).
    regularization : float
        Diagonal regularization for numerical stability.

    Returns
    -------
    inv_fim : tf.Tensor
        Inverse FIM for the last layer parameters.
    """
    loss_fn = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables[-1])
        y_pred = model(X)
        loss = loss_fn(y, y_pred)
    gradient = tape.gradient(loss, model.trainable_variables[-1])
    gradient = tf.reshape(gradient, [-1])
    fisher_im = tf.tensordot(gradient, gradient, axes=0)
    fisher_im += tf.eye(tf.shape(fisher_im)[0]) * regularization
    inv_fim = tf.linalg.inv(fisher_im)
    return tf.cast(inv_fim, tf.float32)


def propagate_uncertainty_last_layer(
    model: tf.keras.Model,
    X: np.ndarray,
    inv_hessian: tf.Tensor,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Predictive standard deviation via last-layer Laplace approximation.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    X : np.ndarray
        Input features, shape (n, n_features).
    inv_hessian : tf.Tensor
        Inverse FIM from compute_inv_fim_last_layer.
    scale_factor : float
        Calibration factor (e.g. from error_calibration on validation set).

    Returns
    -------
    std_errors : np.ndarray, shape (n,)
        Predictive standard deviation per sample.
    """
    n = X.shape[0]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables[-1])
        y_pred = model(X)
    jacobian = tape.jacobian(y_pred, model.trainable_variables[-1])
    jacobian = tf.reshape(jacobian, [n, -1])
    jacobian = tf.cast(jacobian, tf.float32)
    del tape

    std_errors = []
    for i in jacobian:
        temp = tf.linalg.matvec(inv_hessian, i)
        var = tf.tensordot(i, temp, axes=1)
        std_errors.append(tf.sqrt(var))
    return scale_factor * np.array(std_errors)


def error_calibration(
    std_errors: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Compute scaling factor so that predictive std matches empirical errors.

    Minimizes squared relative error; alpha such that (y - y_pred)^2 ~ (alpha * std)^2.

    Parameters
    ----------
    std_errors : np.ndarray
        Uncalibrated predictive standard deviations.
    y_pred : np.ndarray
        Predictions.
    y_true : np.ndarray
        Observed values.

    Returns
    -------
    alpha : float
        Scale factor to apply to std_errors.
    """
    std_errors = np.maximum(std_errors, 1e-8)
    r_sq = (y_true - y_pred) ** 2
    var = std_errors ** 2
    return float(np.sqrt(np.sum(r_sq / var) / len(y_true)))
