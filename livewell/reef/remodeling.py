from __future__ import annotations

import numpy as np
import numpy.typing as npt

from livewell.gear.schema import PlantSpec

Array = npt.NDArray[np.float64]


def phase_split(spec: PlantSpec) -> int:
    return int(round(spec.early_window_hours / spec.dt_hours))


def early_slope(secretion: Array, spec: PlantSpec) -> float:
    cut = phase_split(spec)
    window = secretion[: cut + 1]
    t = np.arange(window.shape[0], dtype=np.float64) * spec.dt_hours
    t = t - t.mean()
    denom = float(np.dot(t, t))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(t, window - window.mean()) / denom)


def late_collapse(barrier: Array, spec: PlantSpec) -> float:
    cut = phase_split(spec)
    return float(barrier[0] - barrier[-1] if barrier.shape[0] > cut else 0.0)


def lead_time(secretion: Array, barrier: Array, spec: PlantSpec) -> float:
    cut = phase_split(spec)
    dt = spec.dt_hours
    sec_peak = int(np.argmax(secretion[: cut + 1])) if secretion.shape[0] else 0
    base = barrier[0]
    drop_idx = barrier.shape[0] - 1
    for k in range(barrier.shape[0]):
        if barrier[k] < base - 0.1:
            drop_idx = k
            break
    return float(max(drop_idx - sec_peak, 0)) * dt
