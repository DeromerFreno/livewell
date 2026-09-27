from __future__ import annotations

import numpy as np
import numpy.typing as npt

from livewell.gear.schema import HorizonSpec

Array = npt.NDArray[np.float64]


class MovingHorizonEstimator:
    def __init__(
        self,
        transition: Array,
        control: Array,
        readout: Array,
        select: Array,
        horizon: HorizonSpec,
        noise_floor: Array,
    ) -> None:
        self.a = transition.astype(np.float64)
        self.b = control.astype(np.float64)
        self.c = readout.astype(np.float64)
        self.select = select.astype(np.float64)
        self.n_obs = self.a.shape[0]
        self.n_state = self.select.shape[0]
        self.n_output = self.c.shape[0]
        self.he = horizon.estimation
        self.arrival = horizon.arrival_weight
        self.process_w = horizon.process_weight
        self.physics_w = horizon.physics_weight
        self.forget = horizon.noise_forget
        self.rv = np.maximum(noise_floor.astype(np.float64), 1.0e-6) ** 2

    def _gradient_residual(self, z_flat: Array, h_mat: Array, g_vec: Array) -> float:
        return float(np.max(np.abs(h_mat @ z_flat - g_vec)))

    def _assemble(self, y_window: Array, u_window: Array, z_prior: Array) -> tuple[Array, Array]:
        steps = u_window.shape[0]
        length = (steps + 1) * self.n_obs
        h_mat = np.zeros((length, length), dtype=np.float64)
        g_vec = np.zeros(length, dtype=np.float64)
        w_meas = 1.0 / self.rv
        ctc = (self.c.T * w_meas) @ self.c
        for k in range(steps + 1):
            sl = slice(k * self.n_obs, (k + 1) * self.n_obs)
            h_mat[sl, sl] += ctc
            g_vec[sl] += (self.c.T * w_meas) @ y_window[k]
        for k in range(steps):
            cur = slice(k * self.n_obs, (k + 1) * self.n_obs)
            nxt = slice((k + 1) * self.n_obs, (k + 2) * self.n_obs)
            bk = self.b @ u_window[k]
            h_mat[nxt, nxt] += self.process_w * np.eye(self.n_obs)
            h_mat[cur, cur] += self.process_w * (self.a.T @ self.a)
            h_mat[cur, nxt] += -self.process_w * self.a.T
            h_mat[nxt, cur] += -self.process_w * self.a
            g_vec[nxt] += self.process_w * bk
            g_vec[cur] += -self.process_w * (self.a.T @ bk)
        head = slice(0, self.n_obs)
        h_mat[head, head] += self.arrival * np.eye(self.n_obs)
        g_vec[head] += self.arrival * z_prior
        self._add_physics(h_mat, steps)
        h_mat += 1.0e-9 * np.eye(length)
        return h_mat, g_vec

    def _add_physics(self, h_mat: Array, steps: int) -> None:
        if self.physics_w <= 0.0 or steps < 2:
            return
        proj = self.select
        smooth = proj.T @ proj
        for k in range(1, steps):
            prev = slice((k - 1) * self.n_obs, k * self.n_obs)
            cur = slice(k * self.n_obs, (k + 1) * self.n_obs)
            nxt = slice((k + 1) * self.n_obs, (k + 2) * self.n_obs)
            for a_sl, b_sl, sign in (
                (cur, cur, 4.0),
                (prev, prev, 1.0),
                (nxt, nxt, 1.0),
                (prev, nxt, 1.0),
                (nxt, prev, 1.0),
                (cur, prev, -2.0),
                (prev, cur, -2.0),
                (cur, nxt, -2.0),
                (nxt, cur, -2.0),
            ):
                h_mat[a_sl, b_sl] += self.physics_w * sign * smooth

    def estimate(self, y_window: Array, u_window: Array, z_prior: Array) -> tuple[Array, Array]:
        h_mat, g_vec = self._assemble(y_window, u_window, z_prior)
        z_flat = np.linalg.solve(h_mat, g_vec).astype(np.float64)
        self._last_residual = self._gradient_residual(z_flat, h_mat, g_vec)
        z_full = z_flat.reshape(-1, self.n_obs)
        self._tune_noise(y_window, z_full)
        x_hat = self.select @ z_full[-1]
        return x_hat, z_full

    def _tune_noise(self, y_window: Array, z_full: Array) -> None:
        pred = z_full @ self.c.T
        residual = np.mean((y_window - pred) ** 2, axis=0)
        self.rv = self.forget * self.rv + (1.0 - self.forget) * np.maximum(residual, 1.0e-8)

    @property
    def optimality_residual(self) -> float:
        return getattr(self, "_last_residual", 0.0)
