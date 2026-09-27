from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from livewell.gear.schema import LiftSpec
from livewell.lens.observables import KoopmanModel


@dataclass(frozen=True)
class LiftLosses:
    linear: Tensor
    recon: Tensor
    physics: Tensor
    total: Tensor


def physics_penalty(model: KoopmanModel, x: Tensor, u: Tensor) -> Tensor:
    z = model.lift(x)
    nxt = model.recover(model.advance(z, u))
    delta = nxt - x
    hypoxia = x[:, 0]
    secretion = x[:, 1]
    barrier = x[:, 3]
    early = torch.relu(-delta[:, 1] * (hypoxia - secretion))
    late = torch.relu(delta[:, 2] * (0.45 - secretion))
    guard = torch.relu(delta[:, 3] - 0.0) * (1.0 - barrier)
    return early.mean() + late.mean() + guard.mean()


def koopman_loss(
    model: KoopmanModel,
    x: Tensor,
    u: Tensor,
    x_next: Tensor,
    spec: LiftSpec,
) -> LiftLosses:
    z = model.lift(x)
    z_next = model.lift(x_next)
    pred = model.advance(z, u)
    linear = torch.mean((pred - z_next) ** 2)
    recon = torch.mean((model.recover(z) - x) ** 2)
    physics = physics_penalty(model, x, u)
    total = spec.linear_weight * linear + spec.recon_weight * recon + spec.physics_weight * physics
    return LiftLosses(linear=linear, recon=recon, physics=physics, total=total)


def ridge_identify(
    model: KoopmanModel,
    x: Tensor,
    u: Tensor,
    x_next: Tensor,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        z = model.lift(x)
        z_next = model.lift(x_next)
        feat = torch.cat([z, u], dim=-1)
        gram = feat.t() @ feat + ridge * torch.eye(feat.shape[-1])
        rhs = feat.t() @ z_next
        solution = torch.linalg.solve(gram, rhs)
        n_obs = model.n_obs
        transition = solution[:n_obs].t().contiguous()
        control = solution[n_obs:].t().contiguous()
    return transition, control
