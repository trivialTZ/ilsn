"""Radial information limits and expected posterior uncertainty."""

import numpy as np
from scipy.stats import norm

from ilsn.likelihood import SourceSample, SourceLikelihood
from ilsn.radial import RadialTransfer
from ilsn.response import ResponseCalibration


def test_outer_null_mode_cannot_constrain_inner_endpoint():
    calibration = ResponseCalibration.gaussian(
        [0.0, 2.0], [-1.0, 1.0], scatter_mag=0.15, n_draws=1
    )
    sample = SourceSample([-0.1], [1.0], [[-0.2, -0.2]])
    radial = RadialTransfer(
        [0.3, 0.7], [0.5, 0.5], SourceLikelihood(sample, calibration)
    )
    np.testing.assert_allclose(
        radial.prior.state_probability, radial.posterior().state_probability
    )
    np.testing.assert_allclose(radial.posterior().standard_deviation, 0.2)


def test_radial_posterior_and_predictive_variance_against_gaussian_mixture():
    calibration = ResponseCalibration.gaussian(
        [0.0, 2.0], [-1.0, 1.0], scatter_mag=0.1, n_draws=1
    )
    sample = SourceSample([-0.15], [1.0], [[-0.3, 0.0]])
    radial = RadialTransfer(
        [0.3, 0.7],
        [0.4, 0.6],
        SourceLikelihood(sample, calibration),
        log_imaging_likelihood=np.log([0.8, 0.2]),
    )
    prior = np.array([0.4 * 0.8, 0.6 * 0.2])
    prior /= prior.sum()
    posterior = prior * norm.pdf(-0.15, [-0.3, 0.0], 0.1)
    posterior /= posterior.sum()
    np.testing.assert_allclose(radial.prior.state_probability, prior)
    np.testing.assert_allclose(radial.posterior().state_probability, posterior)
    coarse = radial.forecast(np.linspace(-1.2, 1.0, 501))
    fine = radial.forecast(np.linspace(-1.2, 1.0, 1001))
    np.testing.assert_allclose(fine["density_integral"], 1.0, atol=1e-12)
    np.testing.assert_allclose(
        coarse["rms_posterior_sd"], fine["rms_posterior_sd"], rtol=1e-5
    )
    assert 0 < fine["rms_posterior_sd"] < fine["prior_sd"]


def test_radial_imaging_evidence_updates_shared_response_consistently():
    means = np.tile([[-0.3, 0.3]], (2, 1, 1))
    variance = np.full_like(means, 0.1**2)
    selection = np.array([[[0.8, 0.2]], [[0.1, 0.9]]])
    cal = ResponseCalibration(
        [0.0, 2.0],
        [-1.0, 1.0],
        means[0],
        variance[0],
        selection[0],
        means,
        variance,
        selection,
        draw_probability=[0.4, 0.6],
    )
    sample = SourceSample([0.1], [1.0], [[-1.0, 1.0]])
    radial = RadialTransfer(
        [0.3, 0.7],
        [0.25, 0.75],
        SourceLikelihood(sample, cal),
        log_imaging_likelihood=np.log([0.9, 0.1]),
    )
    selected = np.array([0.25, 0.75])[None, :] * selection[:, 0, :]
    selected /= selected.sum(1, keepdims=True)
    joint = np.array([0.4, 0.6])[:, None] * selected * [0.9, 0.1]
    np.testing.assert_allclose(
        radial.prior.state_probability, joint.sum(0) / joint.sum()
    )
    joint *= norm.pdf(0.1, [-0.3, 0.3], 0.1)
    np.testing.assert_allclose(
        radial.posterior().state_probability, joint.sum(0) / joint.sum()
    )
