from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from livewell.gear.schema import LiftSpec, PlantSpec

_ACT: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "elu": nn.ELU, "gelu": nn.GELU}


class ObservableDictionary(nn.Module):
    net: nn.Sequential

    def __init__(self, lift: LiftSpec, n_state: int) -> None:
        super().__init__()
        self.n_state = n_state
        self.n_extra = lift.n_observable - n_state
        if self.n_extra < 0:
            raise ValueError("n_observable must exceed the state dimension")
        layers: list[nn.Module] = []
        width = n_state
        for size in lift.hidden:
            layers.append(nn.Linear(width, size))
            layers.append(_ACT[lift.activation]())
            width = size
        layers.append(nn.Linear(width, self.n_extra))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        extra = self.net(x)
        return torch.cat([x, extra], dim=-1)


class KoopmanModel(nn.Module):
    dictionary: ObservableDictionary
    transition: nn.Parameter
    control: nn.Parameter
    select: Tensor
    readout: Tensor

    def __init__(self, lift: LiftSpec, plant: PlantSpec, output_map: np.ndarray) -> None:
        super().__init__()
        self.n_state = plant.n_state
        self.n_obs = lift.n_observable
        self.n_input = plant.n_input
        self.dictionary = ObservableDictionary(lift, plant.n_state)
        self.transition = nn.Parameter(torch.eye(self.n_obs) + 0.01 * torch.randn(self.n_obs, self.n_obs))
        self.control = nn.Parameter(0.01 * torch.randn(self.n_obs, self.n_input))
        select = torch.zeros(self.n_state, self.n_obs)
        select[: self.n_state, : self.n_state] = torch.eye(self.n_state)
        self.register_buffer("select", select)
        cz = torch.tensor(output_map, dtype=torch.float32) @ select
        self.register_buffer("readout", cz)

    def lift(self, x: Tensor) -> Tensor:
        out: Tensor = self.dictionary(x)
        return out

    def advance(self, z: Tensor, u: Tensor) -> Tensor:
        return z @ self.transition.t() + u @ self.control.t()

    def recover(self, z: Tensor) -> Tensor:
        out: Tensor = z @ self.select.t()
        return out

    def emit(self, z: Tensor) -> Tensor:
        out: Tensor = z @ self.readout.t()
        return out

    def matrices(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.transition.detach(), self.control.detach(), self.readout.detach()
