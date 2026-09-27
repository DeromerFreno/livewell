from __future__ import annotations

import os

import torch
import torch.distributed as dist


def launched_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed() -> tuple[int, int]:
    if not launched_distributed():
        return 0, 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_primary(rank: int) -> bool:
    return rank == 0
