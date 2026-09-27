from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from livewell.survey.guarantees import is_reconstructible, observability_rank


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 100000), n=st.integers(2, 6), p=st.integers(1, 3))
def test_generic_pair_is_observable(seed: int, n: int, p: int) -> None:
    gen = np.random.default_rng(seed)
    transition = gen.normal(size=(n, n))
    transition *= 0.8 / (np.max(np.abs(np.linalg.eigvals(transition))) + 1.0e-9)
    readout = gen.normal(size=(p, n))
    horizon = int(np.ceil(n / p))
    assert observability_rank(transition, readout, horizon) == n


def test_blind_mode_is_not_reconstructible() -> None:
    transition = np.diag([0.5, 0.3])
    readout = np.array([[1.0, 0.0]])
    assert not is_reconstructible(transition, readout, 4)
