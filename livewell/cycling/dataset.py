from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from livewell.gear.schema import ExperimentSpec
from livewell.reef.plant import Plant


def transition_tensors(spec: ExperimentSpec, seed: int, count: int) -> tuple[Tensor, Tensor, Tensor]:
    gen = np.random.default_rng(seed)
    plant = Plant(spec.plant)
    n = spec.plant.n_state
    m = spec.plant.n_input
    low = np.array([0.0, spec.safety.oxygen_floor, 0.0])
    high = np.array([spec.safety.shear_ceiling, spec.safety.oxygen_ceiling, spec.safety.drug_ceiling])
    xs = gen.uniform(0.0, 1.2, size=(count, n))
    us = gen.uniform(low, high, size=(count, m))
    res = gen.uniform(0.2, 0.9, size=count)
    nxt = np.empty((count, n), dtype=np.float64)
    for i in range(count):
        nxt[i] = plant.step(xs[i], us[i], float(res[i]))
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(us, dtype=torch.float32),
        torch.tensor(nxt, dtype=torch.float32),
    )


class TransitionBank(torch.utils.data.Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, x: Tensor, u: Tensor, x_next: Tensor) -> None:
        self.x = x
        self.u = u
        self.x_next = x_next

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.x[index], self.u[index], self.x_next[index]
