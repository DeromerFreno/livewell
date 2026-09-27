from __future__ import annotations

import numpy as np

from livewell.gear.schema import ExperimentSpec
from livewell.lens.affine import fit_affine_koopman
from livewell.reef.plant import Plant
from livewell.survey.guarantees import observability_rank, tracking_tube


def test_latent_state_is_reconstructible_over_estimation_horizon() -> None:
    spec = ExperimentSpec(device="cpu")
    plant = Plant(spec.plant)
    model = fit_affine_koopman(plant, spec.safety, 0.6, np.random.default_rng(0))
    rank = observability_rank(model.transition, model.readout, spec.horizon.estimation)
    assert rank >= spec.plant.n_state


def test_tracking_tube_bound_grows_with_disturbance() -> None:
    spec = ExperimentSpec(device="cpu")
    plant = Plant(spec.plant)
    model = fit_affine_koopman(plant, spec.safety, 0.6, np.random.default_rng(1))
    tube = tracking_tube(model.transition, model.control, model.readout)
    low = tube.bound(0.01, 0.01, 0.01)
    high = tube.bound(0.05, 0.05, 0.05)
    assert 0.0 < low < high
