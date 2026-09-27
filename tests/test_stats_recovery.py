from __future__ import annotations

import numpy as np

from livewell.survey.inference import benjamini_hochberg, holm, mcnemar
from livewell.survey.metrics import auroc, cohen_kappa, delong_test, lin_ccc
from livewell.survey.survival import cox_ph, log_rank


def test_auroc_matches_known_separation() -> None:
    gen = np.random.default_rng(0)
    labels = (gen.uniform(size=4000) < 0.5).astype(int)
    scores = 1.0 * labels + gen.normal(size=4000)
    value = auroc(scores, labels)
    assert abs(value - 0.760) < 0.02


def test_delong_detects_difference() -> None:
    gen = np.random.default_rng(1)
    labels = (gen.uniform(size=3000) < 0.45).astype(int)
    strong = 1.6 * labels + gen.normal(size=3000)
    weak = 0.7 * labels + gen.normal(size=3000)
    auc_a, auc_b, p = delong_test(strong, weak, labels)
    assert auc_a > auc_b
    assert p < 1.0e-3


def test_cox_recovers_planted_hazard_ratio() -> None:
    gen = np.random.default_rng(4)
    group = (gen.uniform(size=4000) < 0.4).astype(float)
    rate = 0.1 * np.exp(np.log(3.0) * group)
    time = gen.exponential(1.0 / rate)
    event = np.ones_like(time, dtype=int)
    fit = cox_ph(time, event, group)
    assert abs(fit.hazard_ratio()[0] - 3.0) < 0.4
    chi, p = log_rank(time, event.astype(int), group.astype(int))
    assert p < 1.0e-3


def test_mcnemar_and_ccc_and_kappa() -> None:
    p, odds = mcnemar(326, 85)
    assert p < 1.0e-3
    assert abs(odds - 326 / 85) < 1.0e-6
    x = np.linspace(0.0, 1.0, 200)
    assert lin_ccc(x, x) > 0.999
    a = np.array([1, 1, 0, 0, 1, 0])
    assert abs(cohen_kappa(a, a) - 1.0) < 1.0e-9


def test_multiplicity_corrections_are_monotone() -> None:
    raw = np.array([0.001, 0.01, 0.04, 0.2])
    assert np.all(holm(raw) >= raw)
    assert np.all(benjamini_hochberg(raw) >= raw - 1.0e-12)
    assert np.all(np.diff(np.sort(holm(raw))) >= -1.0e-12)
