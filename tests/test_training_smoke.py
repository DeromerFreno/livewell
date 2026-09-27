from __future__ import annotations

from pathlib import Path

from livewell.cycling.trainer import run_cycle
from livewell.gear.settings import build_experiment


def test_smoke_cycle_runs_and_improves(tmp_path: Path) -> None:
    spec = build_experiment(Path("configs/experiment/_smoke.yaml"))
    record = run_cycle(spec, tmp_path, step_budget=None)
    assert record.steps >= 2
    assert record.losses[-1] < record.losses[0]
    assert Path(record.checkpoint).exists()


def test_smoke_two_step_budget(tmp_path: Path) -> None:
    spec = build_experiment(Path("configs/experiment/_smoke.yaml"))
    record = run_cycle(spec, tmp_path, step_budget=2)
    assert record.steps == 2
