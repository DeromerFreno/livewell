from __future__ import annotations

import numpy as np

from livewell.gear.schema import ExperimentSpec, HorizonSpec
from livewell.lens.affine import fit_affine_koopman
from livewell.reef.plant import Plant
from livewell.sounding.mhe import MovingHorizonEstimator


def test_estimator_is_optimal_and_recovers_state() -> None:
    spec = ExperimentSpec(device="cpu", horizon=HorizonSpec(estimation=14, physics_weight=0.0))
    plant = Plant(spec.plant)
    gen = np.random.default_rng(2)
    model = fit_affine_koopman(plant, spec.safety, 0.6, gen, samples=6000)
    he = spec.horizon.estimation
    x = np.array([0.6, 0.3, 0.8, 0.2])
    inputs = gen.uniform([0.0, 0.05, 0.0], [1.0, 1.0, 1.0], size=(he, 3))
    states = [x]
    for k in range(he):
        states.append(plant.step(states[-1], inputs[k], 0.6))
    truth = np.array(states)
    clean = truth @ plant.output_matrix.T
    measured = clean + gen.normal(0.0, 1.0e-3, clean.shape)
    estimator = MovingHorizonEstimator(
        model.transition,
        model.control,
        model.readout,
        model.select,
        spec.horizon,
        np.full(spec.plant.n_output, 1.0e-3),
    )
    z_prior = model.lift(truth[0])
    x_hat, _ = estimator.estimate(measured, inputs, z_prior)
    assert estimator.optimality_residual < 1.0e-6
    assert np.linalg.norm(x_hat - truth[-1]) < 0.05
