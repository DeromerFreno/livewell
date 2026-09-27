from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy import stats

Array = npt.NDArray[np.float64]


def mcnemar(discordant_b: int, discordant_c: int) -> tuple[float, float]:
    total = discordant_b + discordant_c
    if total == 0:
        return float("nan"), 1.0
    smaller = min(discordant_b, discordant_c)
    p = 2.0 * stats.binom.cdf(smaller, total, 0.5)
    odds = discordant_b / discordant_c if discordant_c > 0 else float("inf")
    return float(min(p, 1.0)), float(odds)


def bootstrap_ci(
    statistic: Callable[[Array], float],
    sample: Array,
    draws: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    gen = np.random.default_rng(seed)
    n = sample.shape[0]
    values = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        idx = gen.integers(0, n, n)
        values[b] = statistic(sample[idx])
    lo = float(np.quantile(values, (1.0 - level) / 2.0))
    hi = float(np.quantile(values, 0.5 + level / 2.0))
    return lo, hi


def holm(pvalues: Array) -> Array:
    order = np.argsort(pvalues)
    m = pvalues.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(pvalues: Array) -> Array:
    order = np.argsort(pvalues)
    m = pvalues.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = pvalues[idx] * m / (rank + 1)
        running = min(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def one_sample_t(values: Array, popmean: float = 0.0) -> tuple[float, float]:
    result = stats.ttest_1samp(values, popmean)
    return float(result.statistic), float(result.pvalue)
