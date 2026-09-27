from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from livewell.gear.schema import ExperimentSpec
from livewell.keeper.mpc import SteeringCore
from livewell.lens.affine import AffineModel
from livewell.probes.array import SensorArray
from livewell.reef.plant import Plant
from livewell.sounding.mhe import MovingHorizonEstimator

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LoopReport:
    closed_track: float
    open_track: float
    feasible: bool

    @property
    def reduction(self) -> float:
        if self.open_track <= 0.0:
            return 0.0
        return (self.open_track - self.closed_track) / self.open_track


class ClosedLoop:
    def __init__(
        self,
        spec: ExperimentSpec,
        plant: Plant,
        sensors: SensorArray,
        model: AffineModel,
        ablations: frozenset[str] = frozenset(),
    ) -> None:
        self.spec = spec
        self.plant = plant
        self.sensors = sensors
        self.model = model
        self.ablations = ablations
        self._readout_pinv = np.linalg.pinv(model.readout)
        self.steering = SteeringCore(
            model.transition, model.control, model.readout, spec.horizon, spec.safety
        )
        self.estimator = MovingHorizonEstimator(
            model.transition,
            model.control,
            model.readout,
            model.select,
            spec.horizon,
            sensors.noise_floor() / np.array([320.0, 410.0, 540.0, spec.probe.teer_baseline, 100.0]),
        )
        if "multimodal_fusion" in ablations:
            self.estimator.rv[[2, 3]] *= 1.0e6
        self.target_state = np.array([0.3, 0.35, 0.85, 0.15], dtype=np.float64)
        self.target_output = plant.output_matrix @ self.target_state

    def _track_error(self, outputs: Array) -> float:
        ref = self.target_output
        denom = float(np.linalg.norm(ref)) + 1.0e-9
        errs = np.linalg.norm(outputs - ref, axis=1) / denom
        return float(np.mean(errs) * 100.0)

    def run_open(self, resistance: float, gen: np.random.Generator, steps: int) -> Array:
        schedule = np.tile(np.array([0.5, 0.55, 0.2]), (steps, 1))
        _, outputs = self.plant.rollout(schedule, resistance, gen)
        return outputs[1:]

    def run_closed(self, resistance: float, gen: np.random.Generator, steps: int) -> tuple[Array, bool]:
        he = self.spec.horizon.estimation
        x = self.plant.initial_state()
        clean0 = self.plant.output_matrix @ x
        meas_hist = [self.sensors.to_state_units(self.sensors.measure(clean0, gen))]
        u_hist: list[Array] = []
        z_prior = self.model.lift(x)
        outputs = np.empty((steps, self.plant.spec.n_output), dtype=np.float64)
        feasible = True
        prev_drug = 0.0
        for k in range(steps):
            window_y = np.array(meas_hist[-(he + 1) :])
            window_u = np.array(u_hist[-he:]) if u_hist else np.zeros((1, self.plant.spec.n_input))
            if "koopman_mhe" in self.ablations:
                x_hat = self.model.select @ (self._readout_pinv @ window_y[-1])
            elif window_y.shape[0] < 2:
                x_hat = x
            else:
                if window_u.shape[0] < window_y.shape[0] - 1:
                    pad = np.tile(window_u[:1], (window_y.shape[0] - 1 - window_u.shape[0], 1))
                    window_u = np.vstack([pad, window_u]) if window_u.size else pad
                x_hat, _ = self.estimator.estimate(window_y, window_u[-(window_y.shape[0] - 1) :], z_prior)
            z0 = self.model.lift(x_hat)
            z_prior = z0
            if "safety_mpc" in self.ablations:
                u = self.steering.safety.enforce(np.array([0.5, 0.55, 0.2]), prev_drug)
            else:
                u = self.steering.solve(z0, self.target_output, prev_drug)
            if not self.steering.safety.contains(u, prev_drug):
                feasible = False
            prev_drug = float(u[2])
            x = self.plant.step(x, u, resistance) + self.plant.dt * gen.normal(
                0.0, self.plant.spec.process_noise, self.plant.spec.n_state
            )
            x = np.clip(x, 0.0, 1.5)
            clean = self.plant.output_matrix @ x
            day_fraction = (k * self.plant.dt) / 24.0
            meas_hist.append(self.sensors.to_state_units(self.sensors.measure(clean, gen, day_fraction)))
            u_hist.append(u)
            outputs[k] = clean
        return outputs, feasible

    def evaluate(self, resistance: float, seed: int, steps: int | None = None) -> LoopReport:
        horizon = steps if steps is not None else self.plant.horizon_steps()
        gen_open = np.random.default_rng(seed)
        gen_closed = np.random.default_rng(seed)
        open_out = self.run_open(resistance, gen_open, horizon)
        closed_out, feasible = self.run_closed(resistance, gen_closed, horizon)
        return LoopReport(
            closed_track=self._track_error(closed_out),
            open_track=self._track_error(open_out),
            feasible=feasible,
        )
