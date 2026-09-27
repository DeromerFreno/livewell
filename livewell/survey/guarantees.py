from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def observability_matrix(transition: Array, readout: Array, horizon: int) -> Array:
    blocks = []
    power = np.eye(transition.shape[0])
    for _ in range(horizon):
        blocks.append(readout @ power)
        power = power @ transition
    return np.vstack(blocks)


def observability_rank(transition: Array, readout: Array, horizon: int) -> int:
    return int(np.linalg.matrix_rank(observability_matrix(transition, readout, horizon)))


def is_reconstructible(transition: Array, readout: Array, horizon: int) -> bool:
    return bool(observability_rank(transition, readout, horizon) == transition.shape[0])


@dataclass(frozen=True)
class TrackingTube:
    gamma_w: float
    gamma_v: float
    gamma_m: float

    def bound(self, w_bar: float, v_bar: float, mismatch: float) -> float:
        return self.gamma_w * w_bar + self.gamma_v * v_bar + self.gamma_m * mismatch


def tracking_tube(transition: Array, control: Array, readout: Array) -> TrackingTube:
    eig = np.abs(np.linalg.eigvals(transition))
    rho = float(np.max(eig))
    contraction = max(1.0 - rho, 1.0e-3)
    obs_gain = float(np.linalg.norm(readout, 2))
    ctrl_gain = float(np.linalg.norm(control, 2))
    gamma_w = (1.0 + ctrl_gain) / contraction
    gamma_v = obs_gain / contraction
    gamma_m = (1.0 + obs_gain + ctrl_gain) / contraction
    return TrackingTube(gamma_w=gamma_w, gamma_v=gamma_v, gamma_m=gamma_m)
