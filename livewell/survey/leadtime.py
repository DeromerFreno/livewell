from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LeadTimeSummary:
    mean: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float


def summarize_lead_time(values: Array, level: float = 0.95) -> LeadTimeSummary:
    n = values.size
    mean = float(np.mean(values))
    sem = float(stats.sem(values)) if n > 1 else 0.0
    half = sem * float(stats.t.ppf(0.5 + level / 2.0, df=max(n - 1, 1)))
    result = stats.ttest_1samp(values, 0.0)
    return LeadTimeSummary(
        mean=mean,
        ci_low=mean - half,
        ci_high=mean + half,
        t_stat=float(result.statistic),
        p_value=float(result.pvalue),
    )
