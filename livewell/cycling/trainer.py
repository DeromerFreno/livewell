from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, DistributedSampler

from livewell.cycling.dataset import TransitionBank, transition_tensors
from livewell.gear.anchor import resolve_device, set_seed
from livewell.gear.harbor import init_distributed, is_primary, shutdown_distributed
from livewell.gear.logbook import atomic_save, get_logger
from livewell.gear.schema import ExperimentSpec
from livewell.lens.identify import koopman_loss
from livewell.lens.observables import KoopmanModel
from livewell.reef.plant import Plant

_LOG = get_logger("cycling")


@dataclass(frozen=True)
class CycleRecord:
    steps: int
    losses: list[float]
    checkpoint: str

    @property
    def final(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def _warmup(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup)


def run_cycle(spec: ExperimentSpec, workdir: Path, step_budget: int | None = None) -> CycleRecord:
    rank, world = init_distributed()
    set_seed(spec.train.seed + rank)
    device = resolve_device(spec.device)
    plant = Plant(spec.plant)
    model = KoopmanModel(spec.lift, spec.plant, plant.output_matrix).to(device)
    wrapped: torch.nn.Module = model
    if world > 1:
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
    x, u, x_next = transition_tensors(spec, spec.train.seed, spec.train.trajectories)
    bank = TransitionBank(x, u, x_next)
    sampler: DistributedSampler[tuple[Tensor, Tensor, Tensor]] | None = (
        DistributedSampler(bank, num_replicas=world, rank=rank) if world > 1 else None
    )
    batch = min(spec.train.batch_size, len(bank))
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        bank, batch_size=batch, shuffle=sampler is None, sampler=sampler, drop_last=False
    )
    optimizer = torch.optim.AdamW(
        wrapped.parameters(), lr=spec.train.lr, weight_decay=spec.train.weight_decay
    )
    scaler = GradScaler(device.type, enabled=spec.train.amp and device.type == "cuda")
    ckpt_path = workdir / f"{spec.name}.pt"
    losses: list[float] = []
    step = 0
    accum = max(spec.train.grad_accum, 1)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(spec.train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for xb, ub, xn in loader:
            xb, ub, xn = xb.to(device), ub.to(device), xn.to(device)
            scale = optimizer.param_groups[0]["lr"]
            optimizer.param_groups[0]["lr"] = spec.train.lr * _warmup(step, spec.train.warmup_steps)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                result = koopman_loss(model, xb, ub, xn, spec.lift)
                loss = result.total / accum
            torch.autograd.backward(scaler.scale(loss))
            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), spec.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]["lr"] = scale
            losses.append(float(result.total.detach().cpu()))
            step += 1
            if is_primary(rank) and step % spec.train.log_every == 0:
                _LOG.info("step %d loss %.6f", step, losses[-1])
            if is_primary(rank) and step % spec.train.ckpt_every == 0:
                _save(model, spec, step, ckpt_path)
            if step_budget is not None and step >= step_budget:
                break
        if step_budget is not None and step >= step_budget:
            break
    if is_primary(rank):
        _save(model, spec, step, ckpt_path)
    shutdown_distributed()
    return CycleRecord(steps=step, losses=losses, checkpoint=str(ckpt_path))


def _save(model: KoopmanModel, spec: ExperimentSpec, step: int, path: Path) -> None:
    atomic_save(
        {
            "state": model.state_dict(),
            "step": step,
            "seed": spec.train.seed,
            "experiment": spec.name,
        },
        path,
    )
