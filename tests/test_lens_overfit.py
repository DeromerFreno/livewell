from __future__ import annotations

import torch

from livewell.gear.schema import LiftSpec, PlantSpec
from livewell.lens.identify import koopman_loss
from livewell.lens.observables import KoopmanModel
from livewell.reef.plant import Plant


def test_single_trajectory_overfits() -> None:
    torch.manual_seed(0)
    plant = Plant(PlantSpec())
    lift = LiftSpec(n_observable=10, hidden=(48, 48), physics_weight=0.0)
    model = KoopmanModel(lift, plant.spec, plant.output_matrix)
    gen = torch.Generator().manual_seed(3)
    x = torch.rand(64, 4, generator=gen)
    u = torch.rand(64, 3, generator=gen)
    x_next = torch.stack(
        [torch.tensor(plant.step(xi.numpy(), ui.numpy(), 0.6)) for xi, ui in zip(x, u)]
    ).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    first = float(koopman_loss(model, x, u, x_next, lift).linear.detach())
    for _ in range(600):
        optimizer.zero_grad()
        loss = koopman_loss(model, x, u, x_next, lift)
        loss.total.backward()
        optimizer.step()
    final = float(koopman_loss(model, x, u, x_next, lift).linear.detach())
    assert final < first
    assert final < 1.0e-3
