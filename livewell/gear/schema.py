from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlantSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_state: int = 4
    n_input: int = 3
    n_output: int = 5
    dt_hours: float = 0.5
    horizon_hours: float = 72.0
    early_window_hours: float = 24.0
    hypoxia_floor: float = 0.008
    secretion_gain: float = 1.35
    stiffening_gain: float = 0.42
    barrier_decay: float = 0.27
    drug_efficacy: float = 0.6
    process_noise: float = 0.01


class ProbeSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lod: tuple[float, float, float] = (0.37, 4.1, 20.7)
    teer_baseline: float = 870.0
    teer_resolution: float = 12.5
    oxygen_resolution: float = 0.07
    drift_per_day: float = 0.086
    measurement_noise: float = 0.02
    zone_count: int = 8


class LiftSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_observable: int = 12
    hidden: tuple[int, ...] = (64, 64)
    activation: Literal["tanh", "elu", "gelu"] = "tanh"
    ridge: float = 1.0e-3
    recon_weight: float = 1.0
    linear_weight: float = 1.0
    physics_weight: float = 0.5


class HorizonSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimation: int = 24
    control: int = 16
    arrival_weight: float = 4.0
    process_weight: float = 1.0
    physics_weight: float = 0.25
    noise_forget: float = 0.9


class SafetySpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shear_ceiling: float = 1.0
    oxygen_floor: float = 0.05
    oxygen_ceiling: float = 1.0
    drug_ceiling: float = 1.0
    drug_ramp_cap: float = 0.15
    viability_guard: float = 0.2
    track_weight: float = 1.0
    effort_weight: float = 0.05


class CohortSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Path to the de-identified, access-controlled cohort records consumed by
    # `panel correlate`. The records are not distributed with this repository;
    # supply your access-controlled copy here or with `--records`.
    records: str | None = None
    # Analysis setting for the bootstrap confidence intervals, not a result.
    bootstrap: int = 2000


class TrainSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = 4000
    batch_size: int = 4096
    grad_accum: int = 4
    world_size: int = 8
    lr: float = 3.0e-6
    warmup_steps: int = 8000
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    precision: Literal["fp32", "tf32", "bf16"] = "fp32"
    amp: bool = False
    trajectories: int = 65536
    seed: int = 20240117
    log_every: int = 50
    ckpt_every: int = 500


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "main"
    device: Literal["cuda", "cpu"] = "cuda"
    plant: PlantSpec = Field(default_factory=PlantSpec)
    probe: ProbeSpec = Field(default_factory=ProbeSpec)
    lift: LiftSpec = Field(default_factory=LiftSpec)
    horizon: HorizonSpec = Field(default_factory=HorizonSpec)
    safety: SafetySpec = Field(default_factory=SafetySpec)
    cohort: CohortSpec = Field(default_factory=CohortSpec)
    train: TrainSpec = Field(default_factory=TrainSpec)
    ablate: tuple[str, ...] = ()
