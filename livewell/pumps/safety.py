from __future__ import annotations

import numpy as np
import numpy.typing as npt

from livewell.gear.schema import SafetySpec

Array = npt.NDArray[np.float64]
_DRUG = 2


class SafetySet:
    def __init__(self, spec: SafetySpec) -> None:
        self.spec = spec
        self._low = np.array([0.0, spec.oxygen_floor, 0.0], dtype=np.float64)
        self._high = np.array([spec.shear_ceiling, spec.oxygen_ceiling, spec.drug_ceiling], dtype=np.float64)

    def input_box(self, horizon: int) -> tuple[Array, Array]:
        return np.tile(self._low, horizon), np.tile(self._high, horizon)

    def drug_slew(self, horizon: int) -> tuple[Array, Array, Array]:
        rows = max(horizon - 1, 0)
        width = horizon * 3
        mat = np.zeros((rows, width), dtype=np.float64)
        for k in range(rows):
            mat[k, k * 3 + _DRUG] = -1.0
            mat[k, (k + 1) * 3 + _DRUG] = 1.0
        cap = self.spec.drug_ramp_cap
        return mat, np.full(rows, -cap), np.full(rows, cap)

    def enforce(self, u: Array, prev_drug: float) -> Array:
        out = np.clip(u, self._low, self._high)
        cap = self.spec.drug_ramp_cap
        out[_DRUG] = float(np.clip(out[_DRUG], prev_drug - cap, prev_drug + cap))
        out[_DRUG] = float(np.clip(out[_DRUG], 0.0, self.spec.drug_ceiling))
        return out

    def contains(self, u: Array, prev_drug: float, tol: float = 1.0e-6) -> bool:
        if np.any(u < self._low - tol) or np.any(u > self._high + tol):
            return False
        return bool(abs(u[_DRUG] - prev_drug) <= self.spec.drug_ramp_cap + tol)
