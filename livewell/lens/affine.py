from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from livewell.gear.schema import SafetySpec
from livewell.reef.plant import Plant

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AffineModel:
    transition: Array
    control: Array
    readout: Array
    select: Array

    def lift(self, x: Array) -> Array:
        return np.append(x.astype(np.float64), 1.0)


def fit_affine_koopman(
    plant: Plant,
    safety: SafetySpec,
    resistance: float,
    gen: np.random.Generator,
    samples: int = 6000,
    ridge: float = 1.0e-4,
) -> AffineModel:
    n = plant.spec.n_state
    m = plant.spec.n_input
    low = np.array([0.0, safety.oxygen_floor, 0.0])
    high = np.array([safety.shear_ceiling, safety.oxygen_ceiling, safety.drug_ceiling])
    xs = gen.uniform(0.0, 1.2, size=(samples, n))
    us = gen.uniform(low, high, size=(samples, m))
    nxt = np.empty((samples, n), dtype=np.float64)
    for i in range(samples):
        nxt[i] = plant.step(xs[i], us[i], resistance)
    feat = np.concatenate([xs, us, np.ones((samples, 1))], axis=1)
    gram = feat.T @ feat + ridge * np.eye(feat.shape[1])
    weight = np.linalg.solve(gram, feat.T @ nxt).T
    a_lin = weight[:, :n]
    b_lin = weight[:, n : n + m]
    offset = weight[:, -1]
    size = n + 1
    a_aug = np.zeros((size, size), dtype=np.float64)
    a_aug[:n, :n] = a_lin
    a_aug[:n, n] = offset
    a_aug[n, n] = 1.0
    b_aug = np.zeros((size, m), dtype=np.float64)
    b_aug[:n, :] = b_lin
    readout = np.zeros((plant.spec.n_output, size), dtype=np.float64)
    readout[:, :n] = plant.output_matrix
    select = np.zeros((n, size), dtype=np.float64)
    select[:, :n] = np.eye(n)
    return AffineModel(transition=a_aug, control=b_aug, readout=readout, select=select)
