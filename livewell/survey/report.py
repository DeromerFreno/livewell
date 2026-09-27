from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from livewell.hatchery.cohort import Cohort
from livewell.survey.inference import mcnemar
from livewell.survey.leadtime import summarize_lead_time
from livewell.survey.metrics import balanced_accuracy, cohen_kappa, delong_test
from livewell.survey.survival import cox_ph, log_rank


@dataclass(frozen=True)
class ClinicalReport:
    dynamics_auroc: float
    endpoint_auroc: float
    delong_p: float
    delong_delta: float
    dynamics_kappa: float
    endpoint_kappa: float
    dynamics_balacc: float
    endpoint_balacc: float
    hazard_ratio: float
    hr_ci_low: float
    hr_ci_high: float
    logrank_p: float
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_p: float
    mcnemar_or: float
    lead_time_mean: float
    lead_time_ci_low: float
    lead_time_ci_high: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def assemble(cohort: Cohort) -> ClinicalReport:
    label = cohort.label
    dyn_auc, end_auc, delong_p = delong_test(cohort.dynamics_score, cohort.endpoint_score, label)
    dyn_kappa = cohen_kappa(cohort.dynamics_call, label)
    end_kappa = cohen_kappa(cohort.endpoint_call, label)
    dyn_bal = balanced_accuracy(label, cohort.dynamics_call)
    end_bal = balanced_accuracy(label, cohort.endpoint_call)
    fit = cox_ph(cohort.survival_time, cohort.survival_event, cohort.survival_group.astype(float))
    hr = float(fit.hazard_ratio()[0])
    lo, hi = fit.ci(0)
    _, lr_p = log_rank(cohort.survival_time, cohort.survival_event, cohort.survival_group)
    discord = cohort.dynamics_call != cohort.endpoint_call
    b = int(np.sum(discord & (cohort.dynamics_call == label)))
    c = int(np.sum(discord & (cohort.endpoint_call == label)))
    mc_p, mc_or = mcnemar(b, c)
    lead = summarize_lead_time(cohort.lead_times)
    return ClinicalReport(
        dynamics_auroc=dyn_auc,
        endpoint_auroc=end_auc,
        delong_p=delong_p,
        delong_delta=dyn_auc - end_auc,
        dynamics_kappa=dyn_kappa,
        endpoint_kappa=end_kappa,
        dynamics_balacc=dyn_bal,
        endpoint_balacc=end_bal,
        hazard_ratio=hr,
        hr_ci_low=lo,
        hr_ci_high=hi,
        logrank_p=lr_p,
        mcnemar_b=b,
        mcnemar_c=c,
        mcnemar_p=mc_p,
        mcnemar_or=mc_or,
        lead_time_mean=lead.mean,
        lead_time_ci_low=lead.ci_low,
        lead_time_ci_high=lead.ci_high,
    )
