from __future__ import annotations

import numpy as np

from livewell.gear.schema import PlantSpec
from livewell.reef.plant import Plant
from livewell.reef.remodeling import early_slope, lead_time
from livewell.survey.leadtime import summarize_lead_time


def test_lead_time_summary_recovers_center() -> None:
    gen = np.random.default_rng(0)
    values = gen.normal(6.0, 3.0, 400)
    summary = summarize_lead_time(values)
    assert abs(summary.mean - 6.0) < 0.6
    assert summary.ci_low < summary.mean < summary.ci_high


def test_two_phase_signature_has_positive_slope_and_lead() -> None:
    spec = PlantSpec()
    plant = Plant(spec)
    steps = plant.horizon_steps()
    schedule = np.tile(np.array([0.4, 0.15, 0.6]), (steps, 1))
    states, _ = plant.rollout(schedule, resistance=0.7)
    secretion = states[:, 1]
    barrier = states[:, 2]
    assert early_slope(secretion, spec) > 0.0
    assert lead_time(secretion, barrier, spec) > 0.0
