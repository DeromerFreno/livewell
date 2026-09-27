from __future__ import annotations

import numpy as np

from livewell.gear.schema import HorizonSpec, SafetySpec
from livewell.keeper.mpc import SteeringCore


def _model(seed: int, n: int = 5, m: int = 3, p: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen = np.random.default_rng(seed)
    a = gen.normal(size=(n, n))
    a *= 0.85 / (np.max(np.abs(np.linalg.eigvals(a))) + 1.0e-9)
    b = gen.normal(size=(n, m)) * 0.3
    c = gen.normal(size=(p, n))
    return a, b, c


def test_actions_stay_inside_safety_set() -> None:
    safety = SafetySpec()
    horizon = HorizonSpec(control=12)
    for seed in range(25):
        a, b, c = _model(seed)
        core = SteeringCore(a, b, c, horizon, safety)
        gen = np.random.default_rng(1000 + seed)
        z0 = gen.normal(size=a.shape[0])
        target = gen.normal(size=c.shape[0])
        prev_drug = float(gen.uniform(0.0, 1.0))
        u = core.solve(z0, target, prev_drug)
        assert core.safety.contains(u, prev_drug)
        assert u.shape == (3,)
