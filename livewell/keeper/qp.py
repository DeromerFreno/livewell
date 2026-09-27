from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_factor, cho_solve

Array = npt.NDArray[np.float64]


def solve_box_qp(
    p_mat: Array,
    q_vec: Array,
    a_mat: Array,
    lower: Array,
    upper: Array,
    rho: float = 1.0,
    sigma: float = 1.0e-6,
    alpha: float = 1.6,
    iters: int = 400,
    tol: float = 1.0e-7,
) -> Array:
    dim = p_mat.shape[0]
    kkt = p_mat + sigma * np.eye(dim) + rho * (a_mat.T @ a_mat)
    factor = cho_factor(kkt, lower=True, check_finite=False)
    x = np.zeros(dim, dtype=np.float64)
    z = np.zeros(a_mat.shape[0], dtype=np.float64)
    y = np.zeros(a_mat.shape[0], dtype=np.float64)
    for _ in range(iters):
        rhs = sigma * x - q_vec + a_mat.T @ (rho * z - y)
        x_new = cho_solve(factor, rhs, check_finite=False)
        ax = a_mat @ x_new
        relaxed = alpha * ax + (1.0 - alpha) * z
        z_new = np.clip(relaxed + y / rho, lower, upper)
        y = y + rho * (relaxed - z_new)
        gap = float(np.max(np.abs(x_new - x)))
        x = x_new
        z = z_new
        if gap < tol:
            break
    return x


def project_box(a_mat: Array, x: Array, lower: Array, upper: Array) -> Array:
    return np.clip(a_mat @ x, lower, upper)
