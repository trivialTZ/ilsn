"""Selected-density and common-calibration identities."""

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import trapezoid
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from ilsn.likelihood import (
    SourceSample,
    SourceLikelihood,
    SharedNuisance,
    magnification_log_likelihood,
)
from ilsn.population import MassPopulation
from ilsn.response import ResponseCalibration


def response(means, variance, selection, weights=None):
    means = np.asarray(means, dtype=float)[:, None, :]
    variance = np.broadcast_to(variance, means.shape)
    selection = np.broadcast_to(selection, means.shape)
    return ResponseCalibration(
        [0.0, 2.0],
        [-1.0, 1.0],
        means[0],
        variance[0],
        selection[0],
        means,
        variance,
        selection,
        draw_probability=weights,
    )


def test_shared_calibration_matches_correlated_gaussian():
    nodes, weights = hermgauss(100)
    offset = np.sqrt(2) * 0.2 * nodes
    cal = response(
        np.column_stack((offset, offset)),
        0.12**2,
        (0.5 + 0.2 * np.tanh(offset))[:, None, None],
        weights,
    )
    observed = np.array([0.13, 0.08, 0.16, 0.05])
    sample = SourceSample(observed, np.ones(4), np.zeros((4, 1)))
    pop = MassPopulation([1.0], [[0.0]])
    likelihood = SourceLikelihood(sample, cal)
    actual = likelihood.log_likelihood(pop, [[0.0]])[0]
    expected = multivariate_normal.logpdf(
        observed, mean=np.zeros(4), cov=0.12**2 * np.eye(4) + 0.2**2 * np.ones((4, 4))
    )
    np.testing.assert_allclose(actual, expected, atol=2e-8)
    independent = sum(
        logsumexp(np.log(cal.draw_probability) + norm.logpdf(y, offset, 0.12))
        for y in observed
    )
    assert abs(actual - independent) > 0.2


def test_selected_density_normalizes_and_uses_each_draw_selection():
    cal = response(
        [[-0.3, 0.2], [-0.1, 0.4]],
        0.1**2,
        np.array([[[0.9, 0.1]], [[0.2, 0.8]]]),
        [0.3, 0.7],
    )
    sample = SourceSample([0.15], [1.0], [[-1.0, 1.0]])
    pop = MassPopulation([0.25, 0.75], [[0.0], [0.0]])
    actual = SourceLikelihood(sample, cal).log_likelihood(pop, [[0.0]])[0]
    selected = np.array([[0.25 * 0.9, 0.75 * 0.1], [0.25 * 0.2, 0.75 * 0.8]])
    kernel = norm.pdf(0.15, np.array([[-0.3, 0.2], [-0.1, 0.4]]), 0.1)
    expected = np.log(
        np.array([0.3, 0.7]) @ ((selected * kernel).sum(1) / selected.sum(1))
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    grid = np.linspace(-1.0, 1.2, 501)
    density = [
        np.exp(
            SourceLikelihood(sample.with_residuals([y]), cal).log_likelihood(
                pop, [[0.0]]
            )[0]
        )
        for y in grid
    ]
    np.testing.assert_allclose(trapezoid(density, grid), 1.0, atol=1e-10)


def test_foreground_data_weight_numerator_but_not_selection():
    cal = response(
        [[-0.3, 0.2], [-0.1, 0.4]],
        0.1**2,
        np.array([[[0.9, 0.1]], [[0.2, 0.8]]]),
        [0.3, 0.7],
    )
    foreground = np.log([[0.6, 0.1]])
    sample = SourceSample(
        [0.15], [1.0], [[-1.0, 1.0]], foreground_log_likelihood=foreground
    )
    pop = MassPopulation([0.25, 0.75], [[0.0], [0.0]])
    actual = SourceLikelihood(sample, cal).log_likelihood(pop, [[0.0]])[0]
    selected = np.array([[0.25 * 0.9, 0.75 * 0.1], [0.25 * 0.2, 0.75 * 0.8]])
    kernel = norm.pdf(0.15, np.array([[-0.3, 0.2], [-0.1, 0.4]]), 0.1)
    expected = np.log(
        np.array([0.3, 0.7])
        @ ((selected * kernel * np.exp(foreground)).sum(1) / selected.sum(1))
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    copy = sample.with_residuals([0.15])
    np.testing.assert_array_equal(copy.foreground_log_likelihood, foreground)
    with pytest.raises(ValueError, match="broadcast"):
        SourceSample([0.15], [1.0], [[-1.0, 1.0]], foreground_log_likelihood=[0, 0, 0])
    with pytest.raises(ValueError, match="below"):
        SourceSample([0.15], [1.0], [[-1.0, 1.0]], foreground_log_likelihood=np.inf)


def test_dust_changes_delivery_and_mean_but_extra_scatter_only_variance():
    cal = response([[-1.0, 1.0]], 0.1**2, [0.8, 0.2])
    sample = SourceSample(
        [0.1], [1.0], [[0.0]], extra_variance_mag2=0.03, dust_template_mag=0.2
    )
    likelihood = SourceLikelihood(sample, cal, SharedNuisance([0], [1], [0.5], [0.02]))
    mean, variance, logmass = likelihood.component_response(0, 0)
    np.testing.assert_allclose(mean, 0.12)
    np.testing.assert_allclose(variance, 0.04)
    np.testing.assert_allclose(np.exp(logmass), 0.47)


def test_unsupported_state_does_not_erase_extreme_tail():
    cal = response([[-1.0, 1.0]], 0.02**2, [1.0, 1.0])
    sample = SourceSample([1.0], [1.0], [[0.0, 1.0]], component_weight=[[1.0, 0.0]])
    pop = MassPopulation([1e-290, 1.0], [[0.0], [0.0]])
    value = SourceLikelihood(sample, cal).log_likelihood(pop, [[0.0]])
    np.testing.assert_allclose(value, norm.logpdf(1.0, 0.0, 0.02), atol=1e-10)


def test_fixed_state_magnification_cancels_selection():
    cal = response([[-1.0, 1.0]], 0.15**2, [0.9, 0.1])
    magnification = np.array([1.0, 1.2, 1.4])
    value = magnification_log_likelihood(-0.2, 1.0, magnification, cal)
    np.testing.assert_allclose(
        value, norm.logpdf(-0.2, -2.5 * np.log10(magnification), 0.15)
    )


def test_zero_delivery_cannot_define_a_selected_sample():
    cal = response([[-1.0, 1.0]], 0.1**2, [0.0, 0.0])
    likelihood = SourceLikelihood(SourceSample([0.0], [1.0], [[0.0]]), cal)
    with pytest.raises(ValueError, match="deliverable"):
        likelihood.log_likelihood(MassPopulation([1.0], [[0.0]]), [[0.0]])


def test_tiny_component_and_delivery_weights_retain_selected_support():
    cal = response([[-1.0, 1.0]], 0.1**2, [1e-200, 0.0])
    sample = SourceSample(
        [-0.9], [1.0], [[-1.0, 1.0]], component_weight=[[1e-200, 1.0]]
    )
    pop = MassPopulation([0.5, 0.5], [[0.0], [0.0]])
    likelihood = SourceLikelihood(sample, cal)
    expected = norm.logpdf(-0.9, -1.0, 0.1)
    np.testing.assert_allclose(likelihood.log_likelihood(pop, [[0.0]]), expected)
    np.testing.assert_allclose(
        likelihood.aligned_log_likelihood(pop, [[0.0]], [0]), expected
    )
    simulated, truth = likelihood.simulate(pop, [0.0], np.random.default_rng(4))
    np.testing.assert_array_equal(truth["mass_state"], [0])
    assert np.isfinite(simulated.residual_mag).all()


def test_unit_photon_continuation_is_an_explicit_likelihood_choice():
    cal = response([[-1.0, 1.0]], 0.1**2, [0.9, 0.1])
    sample = SourceSample([1.4], [1.0], [[1.5]])
    pop = MassPopulation([1.0], [[0.0]])
    with pytest.raises(ValueError, match="photon shift"):
        SourceLikelihood(sample, cal).log_likelihood(pop, [[0.0]])
    likelihood = SourceLikelihood(sample, cal, continuation="unit_photon")
    np.testing.assert_allclose(
        likelihood.log_likelihood(pop, [[0.0]]), norm.logpdf(1.4, 1.5, 0.1)
    )
    mean, variance, logmass = likelihood.component_response(0, 0)
    np.testing.assert_allclose(mean, 1.5)
    np.testing.assert_allclose(variance, 0.1**2)
    np.testing.assert_allclose(np.exp(logmass), 0.1)


def test_zero_population_mass_does_not_require_unmeasured_calibration():
    cal = response([[-1.0, 1.0]], 0.1**2, [1.0, 1.0])
    sample = SourceSample([0.1], [1.0], [[0.0, 5.0]])
    value = SourceLikelihood(sample, cal).log_likelihood(
        MassPopulation([1.0, 0.0], [[0.0], [1.0]]), [[0.0]]
    )
    np.testing.assert_allclose(value, norm.logpdf(0.1, 0.0, 0.1))


def test_paired_nodes_preserve_joint_population_calibration_integration():
    cal = response(
        [[-0.3, 0.2], [-0.1, 0.4]],
        0.1**2,
        np.array([[[0.9, 0.1]], [[0.2, 0.8]]]),
        [0.3, 0.7],
    )
    sample = SourceSample([0.15, -0.1], [1.0, 1.0], [[-1.0, 1.0], [-0.5, 0.6]])
    pop = MassPopulation([0.4, 0.6], [[-1.0], [1.0]])
    likelihood = SourceLikelihood(sample, cal)
    eta = np.array([[-0.7], [0.3], [1.0]])
    prior = np.array([0.2, 0.5, 0.3])
    ordinary = likelihood.log_likelihood(pop, eta)
    paired = likelihood.aligned_log_likelihood(
        pop, np.repeat(eta, 2, axis=0), np.tile([0, 1], 3), chunk_size=1
    )
    joint_mass = (prior[:, None] * cal.draw_probability[None, :]).ravel()
    expected_evidence = logsumexp(ordinary + np.log(prior))
    np.testing.assert_allclose(
        logsumexp(paired + np.log(joint_mass)), expected_evidence, atol=1e-14
    )


def test_simulation_uses_selected_state_measure_and_one_common_draw():
    cal = response(
        [[-1.0, 1.0], [-0.8, 1.2]], 0.01**2, np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    )
    sample = SourceSample(np.zeros(5), np.ones(5), np.tile([-1.0, 1.0], (5, 1)))
    pop = MassPopulation([0.5, 0.5], [[-1.0], [1.0]])
    draw, truth = SourceLikelihood(sample, cal).simulate(
        pop, [0.0], np.random.default_rng(4), nuisance_index=1
    )
    np.testing.assert_array_equal(truth["mass_state"], 1)
    assert np.all(abs(draw.residual_mag - 1.2) < 0.05)
