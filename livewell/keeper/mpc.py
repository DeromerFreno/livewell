from __future__ import annotations

import numpy as np
import numpy.typing as npt

from livewell.gear.schema import HorizonSpec, SafetySpec
from livewell.keeper.qp import solve_box_qp
from livewell.pumps.safety import SafetySet

Array = npt.NDArray[np.float64]
_BIG = 1.0e6


class SteeringCore:
    def __init__(
        self,
        transition: Array,
        control: Array,
        readout: Array,
        horizon: HorizonSpec,
        safety: SafetySpec,
    ) -> None:
        self.a = transition.astype(np.float64)
        self.b = control.astype(np.float64)
        self.c = readout.astype(np.float64)
        self.hc = horizon.control
        self.safety = SafetySet(safety)
        self.n_obs = self.a.shape[0]
        self.n_input = self.b.shape[1]
        self.n_output = self.c.shape[0]
        self._track_w = safety.track_weight
        self._effort_w = safety.effort_weight
        self._guard = safety.viability_guard
        self._teer_row = 3
        self._phi, self._gamma = self._condense()

    def _condense(self) -> tuple[Array, Array]:
        rows = self.hc * self.n_output
        phi = np.zeros((rows, self.n_obs), dtype=np.float64)
        gamma = np.zeros((rows, self.hc * self.n_input), dtype=np.float64)
        powers = [np.eye(self.n_obs)]
        for _ in range(self.hc):
            powers.append(powers[-1] @ self.a)
        for k in range(1, self.hc + 1):
            r0 = (k - 1) * self.n_output
            phi[r0 : r0 + self.n_output] = self.c @ powers[k]
            for j in range(k):
                block = self.c @ powers[k - 1 - j] @ self.b
                c0 = j * self.n_input
                gamma[r0 : r0 + self.n_output, c0 : c0 + self.n_input] = block
        return phi, gamma

    def _cost(self, z0: Array, target: Array) -> tuple[Array, Array]:
        ref = np.tile(target, self.hc) if target.ndim == 1 else target.reshape(-1)
        q_diag = np.full(self.hc * self.n_output, self._track_w)
        r_diag = np.full(self.hc * self.n_input, self._effort_w)
        offset = self._phi @ z0 - ref
        p_mat = 2.0 * (self._gamma.T * q_diag) @ self._gamma + 2.0 * np.diag(r_diag)
        q_vec = 2.0 * (self._gamma.T * q_diag) @ offset
        return p_mat, q_vec

    def _constraints(self, z0: Array) -> tuple[Array, Array, Array]:
        dim = self.hc * self.n_input
        box = np.eye(dim)
        low, high = self.safety.input_box(self.hc)
        slew, slew_lo, slew_hi = self.safety.drug_slew(self.hc)
        teer_rows = self._gamma[self._teer_row :: self.n_output]
        teer_offset = (self._phi @ z0)[self._teer_row :: self.n_output]
        guard_lo = self._guard - teer_offset
        guard_hi = np.full(teer_rows.shape[0], _BIG)
        a_mat = np.vstack([box, slew, teer_rows])
        lower = np.concatenate([low, slew_lo, guard_lo])
        upper = np.concatenate([high, slew_hi, guard_hi])
        return a_mat, lower, upper

    def solve(self, z0: Array, target: Array, prev_drug: float = 0.0) -> Array:
        p_mat, q_vec = self._cost(z0, target)
        a_mat, lower, upper = self._constraints(z0)
        plan = solve_box_qp(p_mat, q_vec, a_mat, lower, upper)
        first = plan[: self.n_input].copy()
        return self.safety.enforce(first, prev_drug)
