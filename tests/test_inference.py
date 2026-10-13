"""Streaming must preserve the same shared-error statistical model."""

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from ilsn.inference import source_posterior
from ilsn.likelihood import SourceSample, SharedNuisance, SourceLikelihood
from ilsn.population import MassPopulation
from ilsn.response import ResponseCalibration
from ilsn.sampling import gaussian_nodes, receiving_nodes


def test_blocked_joint_posterior_matches_explicit_common_integration():
    pop = MassPopulation([0.4, 0.6], [[-1.0], [1.0]])
    response = ResponseCalibration.gaussian(
        [0.8, 1.6], [-1.0, 0.0, 1.0], zero_point_sigma=0.04, n_draws=2, rng=7
    )
    y, shift = np.array([-0.2, -0.4]), np.array([[-0.1, -0.4], [-0.2, -0.5]])
    offsets = np.array([[0.02, -0.01], [-0.03, 0.04]])
    eta = np.array([[-0.5], [0.8]])
    logp = np.log([0.3, 0.7])
    sample = SourceSample(y, [1.2, 1.2], shift, dust_template_mag=0.03)
    blocks = (
        (
            SourceSample(y[i : i + 1], [1.2], shift[i : i + 1], dust_template_mag=0.03),
            offsets[:, i : i + 1],
        )
        for i in range(2)
    )
    fit = source_posterior(
        blocks,
        pop,
        response,
        eta,
        [0, 1],
        logp,
        dust_amplitude=[0.0, 1.0],
        dust_probability=[0.4, 0.6],
    )
    conditional = np.stack(
        [
            SourceLikelihood(
                sample, response, SharedNuisance([0, 1], [0.5, 0.5], d, offsets)
            ).aligned_log_likelihood(pop, eta, [0, 1])
            for d in [0.0, 1.0]
        ]
    )
    expected = logp + logsumexp(conditional + np.log([0.4, 0.6])[:, None], axis=0)
    np.testing.assert_allclose(
        fit["conditional_log_likelihood"], conditional, atol=1e-14
    )
    np.testing.assert_allclose(
        fit["probability"], np.exp(expected - logsumexp(expected)), atol=1e-14
    )


def test_receiving_rule_keeps_source_distribution_and_common_contrast():
    covariance = np.array([[0.09, 0.02], [0.02, 0.16]])
    nodes, weights = gaussian_nodes([0.3, -0.2], covariance, n=4096)
    np.testing.assert_allclose(weights @ nodes, [0.3, -0.2], atol=0.001)
    np.testing.assert_allclose(
        np.cov(nodes.T, aweights=weights, ddof=0), covariance, atol=0.001
    )
    nodes, weights = receiving_nodes(
        [[-1.0, 2.0], [3.0, 4.0]], [0.25, 0.75], np.zeros((2, 2)), n=1024
    )
    assert np.sum(nodes[:, 0] == -1.0) == 256
    np.testing.assert_allclose(weights @ nodes, [2.0, 3.5])


def test_uninformative_sources_leave_paired_receiving_rule_unchanged():
    pop = MassPopulation([0.4, 0.6], [[-1.0], [1.0]])
    response = ResponseCalibration.gaussian([0.8, 1.6], [-1.0, 0.0, 1.0], n_draws=2)
    # Every mass state predicts the same brightness and delivery, so the SN
    # cannot distinguish population coordinates even after selection.
    sample = SourceSample([-0.3], [1.2], [[-0.2, -0.2]])
    eta, prior = np.array([[-0.5], [0.8]]), np.array([0.3, 0.7])
    fit = source_posterior(
        [(sample, [0.0, 0.0])], pop, response, eta, [0, 1], np.log(prior)
    )
    np.testing.assert_allclose(fit["probability"], prior, atol=1e-14)
    before, _ = receiving_nodes(eta, prior, [[0.09]], n=128, seed=3)
    after, _ = receiving_nodes(eta, fit["probability"], [[0.09]], n=128, seed=3)
    np.testing.assert_array_equal(before, after)


def test_zero_joint_prior_mass_does_not_require_unmeasured_response():
    cal = ResponseCalibration(
        [0.8, 1.6],
        [-1.0, 1.0],
        [[-1.0, 1.0]],
        [[0.01, 0.01]],
        [[1.0, 1.0]],
        [[[-1.0, 1.0]], [[np.nan, np.nan]]],
        [[[0.01, 0.01]], [[np.nan, np.nan]]],
        [[[1.0, 1.0]], [[np.nan, np.nan]]],
    )
    sample = SourceSample([0.1], [1.2], [[0.0]])
    fit = source_posterior(
        [(sample, [0.0, 0.0])],
        MassPopulation([1.0], [[0.0]]),
        cal,
        [[0.0], [0.0]],
        [0, 1],
        [0.0, -np.inf],
        dust_amplitude=[0.0],
        dust_probability=[1.0],
    )
    np.testing.assert_array_equal(fit["probability"], [1.0, 0.0])
    assert np.isneginf(fit["conditional_log_likelihood"][0, 1])
    assert fit["log_likelihood"][0] == pytest.approx(norm.logpdf(0.1, 0.0, 0.1))


def test_zero_dust_prior_mass_does_not_extend_photon_response_support():
    cal = ResponseCalibration.gaussian(
        [0.8, 1.6], [-0.5, 0.5], scatter_mag=0.1, n_draws=1
    )
    sample = SourceSample([0.1], [1.2], [[0.0]], dust_template_mag=2.0)
    fit = source_posterior(
        [(sample, [0.0])],
        MassPopulation([1.0], [[0.0]]),
        cal,
        [[0.0]],
        [0],
        [0.0],
        dust_amplitude=[0.0, 1.0],
        dust_probability=[1.0, 0.0],
    )
    assert fit["log_likelihood"][0] == pytest.approx(norm.logpdf(0.1, 0.0, 0.1))
    assert np.isneginf(fit["conditional_log_likelihood"][1, 0])


def test_streamed_common_offsets_match_correlated_gaussian():
    normal, weight = hermgauss(200)
    offsets = np.sqrt(2) * 0.2 * normal
    prior = weight / np.sqrt(np.pi)
    observed = np.array([0.13, 0.08, 0.16, 0.05])
    cal = ResponseCalibration.gaussian(
        [0.8, 1.6], [-1.0, 1.0], scatter_mag=0.12, n_draws=1
    )
    blocks = ((SourceSample([y], [1.2], [[0.0]]), offsets) for y in observed)
    fit = source_posterior(
        blocks,
        MassPopulation([1.0], [[0.0]]),
        cal,
        np.zeros((200, 1)),
        np.zeros(200, int),
        np.log(prior),
        dust_amplitude=[0.0],
        dust_probability=[1.0],
    )
    actual = logsumexp(fit["log_likelihood"] + np.log(prior))
    expected = multivariate_normal.logpdf(
        observed, cov=0.12**2 * np.eye(4) + 0.2**2 * np.ones((4, 4))
    )
    assert actual == pytest.approx(expected, abs=2e-8)
    independent = np.sum(norm.logpdf(observed, 0, np.hypot(0.12, 0.2)))
    assert abs(actual - independent) > 0.2


def test_postdelivery_offsets_do_not_extend_photon_response_support():
    cal = ResponseCalibration.gaussian(
        [0.8, 1.6], [-0.2, 0.0], scatter_mag=0.1, n_draws=1
    )
    pop = MassPopulation([1.0], [[0.0]])
    sample = SourceSample([0.6], [1.2], [[-0.2]], dust_template_mag=0.3)
    fit = source_posterior(
        [(sample, [0.8])],
        pop,
        cal,
        [[0.0]],
        [0],
        [0.0],
        dust_amplitude=[0.0],
        dust_probability=[1.0],
    )
    assert fit["log_likelihood"][0] == pytest.approx(norm.logpdf(0.6, 0.6, 0.1))
    # Dust changes the photons; an offset back into the observed residual
    # interval cannot make the positive photon shift calibrated.
    with pytest.raises(ValueError, match="photon shift"):
        source_posterior(
            [(sample, [-0.1])],
            pop,
            cal,
            [[0.0]],
            [0],
            [0.0],
            dust_amplitude=[1.0],
            dust_probability=[1.0],
        )
