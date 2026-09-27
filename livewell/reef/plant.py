from __future__ import annotations

import numpy as np
import numpy.typing as npt

from livewell.gear.schema import PlantSpec

Array = npt.NDArray[np.float64]


class Plant:
    def __init__(self, spec: PlantSpec) -> None:
        self.spec = spec
        self.dt = spec.dt_hours
        self._c = self._output_map()

    def _output_map(self) -> Array:
        return np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.35, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    @property
    def output_matrix(self) -> Array:
        return self._c

    def initial_state(self) -> Array:
        return np.array([0.85, 0.05, 1.0, 0.02], dtype=np.float64)

    def step(self, x: Array, u: Array, resistance: float = 0.5) -> Array:
        s = self.spec
        hypoxia, secretion, barrier, stiffness = x
        shear, oxygen, drug = u
        target_hypoxia = np.clip(1.0 - 0.95 * oxygen, s.hypoxia_floor, 1.0)
        washout = 0.18 * shear
        d_hyp = 0.9 * (target_hypoxia - hypoxia)
        d_sec = s.secretion_gain * hypoxia * (1.0 - secretion) - (0.25 + washout) * secretion
        drive = max(secretion - 0.45, 0.0)
        d_stf = s.stiffening_gain * drive * (1.0 - stiffness) - 0.04 * stiffness
        kill = s.drug_efficacy * drug * (1.0 - resistance)
        d_bar = -s.barrier_decay * (stiffness + 0.5 * kill) * barrier + 0.02 * (1.0 - barrier)
        nxt = x + self.dt * np.array([d_hyp, d_sec, d_stf, d_bar], dtype=np.float64)
        return np.clip(nxt, 0.0, 1.5)

    def rollout(
        self,
        u_seq: Array,
        resistance: float,
        gen: np.random.Generator | None = None,
        x0: Array | None = None,
    ) -> tuple[Array, Array]:
        steps = u_seq.shape[0]
        x = self.initial_state() if x0 is None else x0.astype(np.float64).copy()
        states = np.empty((steps + 1, self.spec.n_state), dtype=np.float64)
        outputs = np.empty((steps + 1, self.spec.n_output), dtype=np.float64)
        states[0] = x
        outputs[0] = self._c @ x
        for k in range(steps):
            noise = (
                gen.normal(0.0, self.spec.process_noise, self.spec.n_state)
                if gen is not None
                else np.zeros(self.spec.n_state)
            )
            x = self.step(x, u_seq[k], resistance) + self.dt * noise
            x = np.clip(x, 0.0, 1.5)
            states[k + 1] = x
            outputs[k + 1] = self._c @ x
        return states, outputs

    def horizon_steps(self) -> int:
        return int(round(self.spec.horizon_hours / self.dt))
