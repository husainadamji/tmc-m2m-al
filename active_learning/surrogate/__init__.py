"""Neural network surrogates with Laplace approximation uncertainty."""

from .model import add_ones_to_input, build_model, build_mlp
from .uncertainty import (
    compute_inv_fim_last_layer,
    propagate_uncertainty_last_layer,
    error_calibration,
)

__all__ = [
    "add_ones_to_input",
    "build_model",
    "build_mlp",
    "compute_inv_fim_last_layer",
    "propagate_uncertainty_last_layer",
    "error_calibration",
]
