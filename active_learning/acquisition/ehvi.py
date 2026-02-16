"""
2D Expected Hypervolume Improvement (EHVI) and Pareto front indices.

Reference: M. Emmerich, K. Yang, A. Deutz, H. Wang, and C. M. Fonseca,
"A Multicriteria Generalization of Bayesian Global Optimization",
https://doi.org/10.1007/978-3-319-29975-4_12
"""

import numpy as np
from scipy.stats import norm


def get_2D_pareto_indices_min_min(points: np.ndarray) -> np.ndarray:
    """Indices of the 2D Pareto front when minimizing both objectives.

    Parameters
    ----------
    points : array_like, shape (N, 2)
        Two-dimensional array (e.g. HAT, |rebound|).

    Returns
    -------
    pareto_indices : ndarray of int, shape (M,)
        Indices that define the Pareto front.
    """
    indices = np.argsort(points[:, 0])
    pareto_indices = []
    y_min = np.inf
    for ind in indices:
        if points[ind, 1] < y_min:
            pareto_indices.append(ind)
            y_min = points[ind, 1]
    return np.array(pareto_indices)


def get_2D_pareto_indices_min_max(points: np.ndarray) -> np.ndarray:
    """Indices of the 2D Pareto front when minimizing obj0 and maximizing obj1."""
    indices = np.argsort(points[:, 0])
    pareto_indices = []
    y_max = -np.inf
    for ind in indices:
        if points[ind, 1] > y_max:
            pareto_indices.append(ind)
            y_max = points[ind, 1]
    return np.array(pareto_indices)


def get_2D_EHVI_min_min(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    pareto_points: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """2D Expected Hypervolume Improvement (both objectives minimized).

    Parameters
    ----------
    pred_mean : array_like, shape (N, 2)
        Predicted mean for both objectives.
    pred_std : array_like, shape (N, 2)
        Predicted standard deviation for both objectives.
    pareto_points : array_like, shape (M, 2)
        Current Pareto front points; x must be ascending, y descending.
    r : array_like, shape (2,)
        Reference point for hypervolume.

    Returns
    -------
    ehvi : ndarray, shape (N,)
        EHVI value per candidate.
    """
    if np.any(np.diff(pareto_points[:, 0]) < 0.0):
        raise ValueError("Pareto front x values must be ascending")
    if np.any(np.diff(pareto_points[:, 1]) > 0.0):
        raise ValueError("Pareto front y values must be descending")

    mu_1 = pred_mean[:, 0]
    s_1 = pred_std[:, 0]
    mu_2 = pred_mean[:, 1]
    s_2 = pred_std[:, 1]

    P = np.zeros([len(pareto_points) + 2, 2])
    P[0] = (r[0], -np.inf)
    P[1:-1] = pareto_points[::-1]
    P[-1] = (-np.inf, r[1])

    ehvi = np.zeros(len(pred_mean))

    def psi(a, b, mu, s):
        t = (b - mu) / s
        return s * norm.pdf(t) + (a - mu) * norm.cdf(t)

    for i in range(len(pareto_points) + 1):
        if i != len(pareto_points):
            ehvi += (
                (P[i][0] - P[i + 1][0])
                * norm.cdf((P[i + 1][0] - mu_1) / s_1)
                * psi(P[i + 1][1], P[i + 1][1], mu_2, s_2)
            )
        ehvi += (
            psi(P[i][0], P[i][0], mu_1, s_1) - psi(P[i][0], P[i + 1][0], mu_1, s_1)
        ) * psi(P[i + 1][1], P[i + 1][1], mu_2, s_2)
    return ehvi


def get_2D_EHVI_min_max(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    pareto_points: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """2D EHVI when minimizing first objective and maximizing second."""
    if np.any(np.diff(pareto_points[:, 0]) < 0.0):
        raise ValueError("Pareto front x values must be ascending")
    if np.any(np.diff(pareto_points[:, 1]) < 0.0):
        raise ValueError("Pareto front y values must be ascending")

    mu_1 = pred_mean[:, 0]
    s_1 = pred_std[:, 0]
    mu_2 = pred_mean[:, 1]
    s_2 = pred_std[:, 1]

    P = np.zeros([len(pareto_points) + 2, 2])
    P[0] = (r[0], np.inf)
    P[1:-1] = pareto_points[::-1]
    P[-1] = (-np.inf, r[1])

    ehvi = np.zeros(len(pred_mean))

    def psi_1(a, b, mu, s):
        t = (b - mu) / s
        return s * norm.pdf(t) + (mu - a) * (1 - norm.cdf(t))

    def psi_2(a, b, mu, s):
        t = (b - mu) / s
        return s * norm.pdf(t) + (a - mu) * norm.cdf(t)

    for i in range(len(pareto_points) + 1):
        if i != len(pareto_points):
            ehvi += (
                (P[i][0] - P[i + 1][0])
                * norm.cdf((P[i + 1][0] - mu_1) / s_1)
                * psi_1(P[i + 1][1], P[i + 1][1], mu_2, s_2)
            )
        ehvi += (
            psi_2(P[i][0], P[i][0], mu_1, s_1) - psi_2(P[i][0], P[i + 1][0], mu_1, s_1)
        ) * psi_1(P[i + 1][1], P[i + 1][1], mu_2, s_2)
    return ehvi
