from __future__ import annotations

import numpy as np
import numpy.typing as npt

from livewell.gear.schema import ProbeSpec

Array = npt.NDArray[np.float64]


class SensorArray:
    def __init__(self, spec: ProbeSpec, n_output: int, dt_hours: float) -> None:
        self.spec = spec
        self.n_output = n_output
        self.dt = dt_hours
        self._scale = self._channel_scale()
        self._floor = self._channel_floor()

    def _channel_scale(self) -> Array:
        return np.array([320.0, 410.0, 540.0, spec_teer(self.spec), 100.0], dtype=np.float64)

    def _channel_floor(self) -> Array:
        lod = np.array(self.spec.lod, dtype=np.float64)
        return np.concatenate([lod, np.array([0.0, 0.0], dtype=np.float64)])

    def noise_floor(self) -> Array:
        base = np.full(self.n_output, self.spec.measurement_noise, dtype=np.float64)
        return base * self._scale

    def measure(
        self,
        clean: Array,
        gen: np.random.Generator | None = None,
        drift_fraction: float = 0.0,
    ) -> Array:
        physical = clean * self._scale
        drift = drift_fraction * self.spec.drift_per_day * self._scale
        noise = (
            gen.normal(0.0, 1.0, physical.shape) * self.noise_floor()
            if gen is not None
            else np.zeros_like(physical)
        )
        measured = physical + drift + noise
        measured = np.maximum(measured, self._floor)
        return measured

    def to_state_units(self, measured: Array) -> Array:
        return measured / self._scale


def spec_teer(spec: ProbeSpec) -> float:
    return spec.teer_baseline
