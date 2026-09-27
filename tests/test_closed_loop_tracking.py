from __future__ import annotations

import numpy as np
import pytest

from livewell.gear.schema import ExperimentSpec
from livewell.lens.affine import fit_affine_koopman
from livewell.probes.array import SensorArray
from livewell.pumps.loop import ClosedLoop
from livewell.reef.plant import Plant


def _evaluate(seed: int, resistance: float) -> float:
    spec = ExperimentSpec(device="cpu")
    plant = Plant(spec.plant)
    sensors = SensorArray(spec.probe, spec.plant.n_output, spec.plant.dt_hours)
    model = fit_affine_koopman(plant, spec.safety, resistance, np.random.default_rng(100 + seed))
    loop = ClosedLoop(spec, plant, sensors, model)
    return loop.evaluate(resistance=resistance, seed=seed, steps=144).reduction


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closed_loop_beats_open_loop_target(seed: int) -> None:
    assert _evaluate(seed, 0.6) >= 1.0 / 3.0


def test_mean_reduction_has_margin() -> None:
    reductions = [_evaluate(s, r) for s in range(3) for r in (0.4, 0.6, 0.8)]
    assert float(np.mean(reductions)) >= 0.36
