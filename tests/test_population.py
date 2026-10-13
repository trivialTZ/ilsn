"""Conditional population sharing, target evidence and covariance."""

from itertools import product

import numpy as np
from scipy.special import logsumexp

from ilsn.population import (
    MassPopulation,
    TargetSample,
    population_transfer,
    target_readout,
)


def test_target_covariance_and_imaging_evidence_match_enumeration():
    pop = MassPopulation([0.2, 0.5, 0.3], [[-1.0], [0.0], [1.0]])
    eta, delta = np.array([[-0.8], [0.7]]), np.array([[-0.4], [0.3]])
    pe, pd = np.array([0.4, 0.6]), np.array([0.7, 0.3])
    endpoint = np.array([[0.3, 0.5, 0.8], [0.2, 0.6, 0.7]])
    selection = np.array([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]])
    logimage = -0.5 * ((endpoint - np.array([[0.6], [0.55]])) / 0.2) ** 2
    source_logl = np.log([0.2, 0.8])
    result = population_transfer(
        pop,
        TargetSample(endpoint, selection, logimage),
        eta,
        source_logl,
        pe,
        delta,
        pd,
    )
    for with_source, summary in ((False, result.prior), (True, result.posterior)):
        values, masses = [], []
        for h, c, s1, s2 in product(range(2), range(2), range(3), range(3)):
            p = pop.probabilities(eta[h] + delta[c])[0]
            selected = selection * p[None, :]
            selected /= selected.sum(1, keepdims=True)
            mass = pe[h] * pd[c] * selected[0, s1] * selected[1, s2]
            mass *= np.exp(logimage[0, s1] + logimage[1, s2])
            if with_source:
                mass *= np.exp(source_logl[h])
            masses.append(mass)
            values.append([endpoint[0, s1], endpoint[1, s2]])
        p, values = np.array(masses) / np.sum(masses), np.array(values)
        mean = p @ values
        covariance = ((values - mean).T * p) @ (values - mean)
        np.testing.assert_allclose(summary.mean, mean, atol=1e-14)
        np.testing.assert_allclose(summary.covariance, covariance, atol=1e-14)
        np.testing.assert_allclose(summary.catalogue_mean_sd**2, covariance.sum() / 4)
    assert result.posterior.shared_covariance[0, 1] > 0


def test_source_data_cannot_update_independent_target_contrast_without_target_data():
    pop = MassPopulation([0.5, 0.5], [[-1.0], [1.0]])
    eta, delta = np.array([[-1.0], [1.0]]), np.array([[-2.0], [0.0], [2.0]])
    pd = [0.25, 0.5, 0.25]
    targets = TargetSample([[0.2, 0.8], [0.3, 0.9]], [[1.0, 2.0], [2.0, 1.0]])
    fit = population_transfer(pop, targets, eta, [-5.0, 0.0], [0.5, 0.5], delta, pd)
    np.testing.assert_allclose(fit.posterior_node_probability.reshape(2, 3).sum(0), pd)
    assert fit.posterior.catalogue_mean_sd > 0.1


def test_uninformative_sources_leave_target_distribution_unchanged():
    pop = MassPopulation([0.5, 0.5], [[-1.0], [1.0]])
    target = TargetSample([[0.2, 0.8]], [[1.0, 1.0]], [[-0.1, -1.0]])
    result = population_transfer(
        pop,
        target,
        [[-1.0], [1.0]],
        [-100.0, -100.0],
        [0.5, 0.5],
        [[-0.4], [0.4]],
        [0.5, 0.5],
    )
    np.testing.assert_allclose(result.prior.mean, result.posterior.mean)
    np.testing.assert_allclose(result.prior.covariance, result.posterior.covariance)


def test_mass_population_retains_quadrature_measure_and_zero_support():
    pop = MassPopulation([0.1, 0.9, 0.0], [[-1.0], [1.0], [3.0]])
    np.testing.assert_allclose(pop.probabilities([[0.0]]), [[0.1, 0.9, 0.0]])
    np.testing.assert_allclose(
        logsumexp(pop.log_probabilities([[100.0]])), 0.0, atol=1e-14
    )


def test_paired_receiving_readout_matches_common_contrast_and_all_target_evidence():
    pop = MassPopulation([0.2, 0.5, 0.3], [[-1.0], [0.0], [1.0]])
    targets = TargetSample(
        [[0.3, 0.5, 0.8], [0.2, 0.6, 0.7]],
        [[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]],
        [[-0.3, -0.1, -0.8], [-0.2, -0.6, -0.4]],
    )
    eta, delta = np.array([[-0.8], [0.7]]), np.array([[-0.4], [0.3]])
    prior, contrast = np.array([0.4, 0.6]), np.array([0.7, 0.3])
    source_logl = np.log([0.2, 0.8])
    ordinary = population_transfer(
        pop, targets, eta, source_logl, prior, delta, contrast
    )
    source_post = prior * np.exp(source_logl)
    source_post /= source_post.sum()
    coordinates = np.repeat(eta, 2, axis=0) + np.tile(delta, (2, 1))
    shared_prior = (source_post[:, None] * contrast[None, :]).ravel()
    paired, posterior = target_readout(
        pop, targets, coordinates, shared_prior, chunk_size=1
    )
    np.testing.assert_allclose(
        posterior, ordinary.posterior_node_probability, atol=1e-14
    )
    np.testing.assert_allclose(paired.mean, ordinary.posterior.mean, atol=1e-14)
    np.testing.assert_allclose(
        paired.covariance, ordinary.posterior.covariance, atol=1e-14
    )
    assert paired.shared_covariance[0, 1] > 0
