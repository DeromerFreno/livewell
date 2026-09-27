from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


@dataclass(frozen=True)
class CoxFit:
    coef: Array
    se: Array

    def hazard_ratio(self) -> Array:
        return np.exp(self.coef)

    def ci(self, index: int = 0, level: float = 0.95) -> tuple[float, float]:
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        lo = self.coef[index] - z * self.se[index]
        hi = self.coef[index] + z * self.se[index]
        return float(np.exp(lo)), float(np.exp(hi))


def cox_ph(time: Array, event: IntArray, covariates: Array, iters: int = 50, tol: float = 1.0e-8) -> CoxFit:
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    n, p = covariates.shape
    order = np.argsort(-time)
    e = event[order]
    x = covariates[order]
    beta = np.zeros(p, dtype=np.float64)
    for _ in range(iters):
        eta = x @ beta
        w = np.exp(eta)
        risk_sum = np.cumsum(w)
        risk_x = np.cumsum(w[:, None] * x, axis=0)
        risk_xx = np.cumsum(w[:, None, None] * (x[:, :, None] * x[:, None, :]), axis=0)
        grad = np.zeros(p, dtype=np.float64)
        hess = np.zeros((p, p), dtype=np.float64)
        for i in range(n):
            if e[i] != 1:
                continue
            mean = risk_x[i] / risk_sum[i]
            grad += x[i] - mean
            cov = risk_xx[i] / risk_sum[i] - np.outer(mean, mean)
            hess -= cov
        step = np.linalg.solve(hess - 1.0e-9 * np.eye(p), grad)
        beta = beta - step
        if float(np.max(np.abs(step))) < tol:
            break
    eta = x @ beta
    w = np.exp(eta)
    risk_sum = np.cumsum(w)
    risk_x = np.cumsum(w[:, None] * x, axis=0)
    risk_xx = np.cumsum(w[:, None, None] * (x[:, :, None] * x[:, None, :]), axis=0)
    info = np.zeros((p, p), dtype=np.float64)
    for i in range(n):
        if e[i] != 1:
            continue
        mean = risk_x[i] / risk_sum[i]
        info += risk_xx[i] / risk_sum[i] - np.outer(mean, mean)
    cov_beta = np.linalg.inv(info + 1.0e-9 * np.eye(p))
    return CoxFit(coef=beta, se=np.sqrt(np.diag(cov_beta)))


def log_rank(time: Array, event: IntArray, group: IntArray) -> tuple[float, float]:
    times = np.unique(time[event == 1])
    obs = 0.0
    exp = 0.0
    var = 0.0
    for tau in times:
        at_risk = time >= tau
        n_total = float(np.sum(at_risk))
        n_one = float(np.sum(at_risk & (group == 1)))
        d_total = float(np.sum((time == tau) & (event == 1)))
        d_one = float(np.sum((time == tau) & (event == 1) & (group == 1)))
        if n_total <= 1:
            continue
        obs += d_one
        exp += d_total * n_one / n_total
        var += d_total * (n_one / n_total) * (1.0 - n_one / n_total) * (n_total - d_total) / (n_total - 1.0)
    if var <= 0.0:
        return 0.0, 1.0
    chi = (obs - exp) ** 2 / var
    return float(chi), float(stats.chi2.sf(chi, 1))
