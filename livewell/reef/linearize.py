from __future__ import annotations

import numpy as np
import numpy.typing as npt

from livewell.reef.plant import Plant

Array = npt.NDArray[np.float64]


def jacobians(
    plant: Plant, x: Array, u: Array, resistance: float, eps: float = 1.0e-5
) -> tuple[Array, Array]:
    base = plant.step(x, u, resistance)
    n = x.shape[0]
    m = u.shape[0]
    a_mat = np.zeros((n, n), dtype=np.float64)
    b_mat = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        bump = x.copy()
        bump[i] += eps
        a_mat[:, i] = (plant.step(bump, u, resistance) - base) / eps
    for j in range(m):
        bump = u.copy()
        bump[j] += eps
        b_mat[:, j] = (plant.step(x, bump, resistance) - base) / eps
    return a_mat, b_mat


def nominal_operating_point(plant: Plant) -> tuple[Array, Array]:
    x = np.array([0.4, 0.4, 0.7, 0.3], dtype=np.float64)
    u = np.array([0.5, 0.5, 0.3], dtype=np.float64)
    return x, u
