from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from livewell.gear.schema import ExperimentSpec, PlantSpec
from livewell.lens.affine import fit_affine_koopman
from livewell.probes.array import SensorArray
from livewell.pumps.loop import ClosedLoop
from livewell.reef.plant import Plant

_BIOLOGICAL = {
    "stiffness_gradient": {"stiffening_gain": 0.5},
    "hypoxia_chamber": {"secretion_gain": 0.55, "hypoxia_floor": 0.05},
    "tri_culture": {"barrier_decay": 0.5},
}
_COMPUTATIONAL = {"multimodal_fusion", "koopman_mhe", "safety_mpc"}


@dataclass(frozen=True)
class AblationRow:
    name: str
    track_error: float
    reduction: float


def _plant_with(spec: PlantSpec, edits: dict[str, float]) -> PlantSpec:
    return spec.model_copy(update=edits)


def ablation_sweep(spec: ExperimentSpec, seed: int, steps: int) -> list[AblationRow]:
    rows: list[AblationRow] = []
    rows.append(_run(spec, "full", frozenset(), spec.plant, seed, steps))
    for name in ("stiffness_gradient", "hypoxia_chamber", "tri_culture"):
        plant_spec = _plant_with(spec.plant, _BIOLOGICAL[name])
        rows.append(_run(spec, name, frozenset(), plant_spec, seed, steps))
    for name in ("multimodal_fusion", "koopman_mhe", "safety_mpc"):
        rows.append(_run(spec, name, frozenset({name}), spec.plant, seed, steps))
    return rows


def _run(
    spec: ExperimentSpec,
    name: str,
    ablations: frozenset[str],
    plant_spec: PlantSpec,
    seed: int,
    steps: int,
) -> AblationRow:
    plant = Plant(plant_spec)
    sensors = SensorArray(spec.probe, plant_spec.n_output, plant_spec.dt_hours)
    model = fit_affine_koopman(plant, spec.safety, 0.6, np.random.default_rng(seed), samples=4000)
    loop = ClosedLoop(spec, plant, sensors, model, ablations=ablations)
    report = loop.evaluate(resistance=0.6, seed=seed, steps=steps)
    return AblationRow(name=name, track_error=report.closed_track, reduction=report.reduction)
