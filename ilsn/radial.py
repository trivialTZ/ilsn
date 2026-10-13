"""Information transfer from an ilSN radius to an inner lens quantity."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import logsumexp

from .population import log_probability, normalized_probability


@dataclass
class RadialSummary:
    """Mass-state distribution and scalar inner-quantity moments."""

    state_probability: np.ndarray
    mean: float
    standard_deviation: float


class RadialTransfer:
    """Marginalize radial model freedom and shared response for one ilSN.

    The mass-state quadrature sets the correlation between outer magnification
    and the inner endpoint. It can include different core scales and signed
    compensated mass distributions. No exact-MST relation between an outer
    magnification measurement and s_E is assumed.

    :param endpoint: Inner quantity per mass state, usually s_E=1-kappa(theta_E).
    :param state_probability: Prior mass-state quadrature weights.
    :param source_likelihood: SourceLikelihood for exactly one source.
    :param log_imaging_likelihood: Optional inner-data log likelihood per state.
        It conditions both the before-SN and after-SN distributions.
    """

    def __init__(
        self,
        endpoint,
        state_probability,
        source_likelihood,
        log_imaging_likelihood=None,
    ):
        self.endpoint = np.asarray(endpoint, dtype=float)
        self.state_probability = normalized_probability(state_probability)
        self.likelihood = source_likelihood
        if (
            len(source_likelihood.sample.residual_mag) != 1
            or self.endpoint.shape != self.state_probability.shape
            or len(self.endpoint) != source_likelihood.sample.n_states
        ):
            raise ValueError(
                "Radial transfer requires one SN and aligned mass states/endpoints."
            )
        logi = (
            np.zeros_like(self.endpoint)
            if log_imaging_likelihood is None
            else np.asarray(log_imaging_likelihood, dtype=float)
        )
        if logi.shape != self.endpoint.shape or np.any(
            np.isnan(logi) | np.isposinf(logi)
        ):
            raise ValueError("Imaging log likelihood must align with the mass states.")
        nq = len(source_likelihood.nuisance.probability)
        self._log_measure = np.full((nq, len(self.endpoint)), -np.inf)
        logw = log_probability(self.state_probability)
        for q, pq in enumerate(source_likelihood.nuisance.probability):
            if pq == 0:
                continue
            _, logs = source_likelihood.state_terms(
                0, q, state_support=self.state_probability > 0
            )
            norm = logsumexp(logw + logs)
            if not np.isfinite(norm):
                raise ValueError(
                    "A positive-prior response has no selected radial support."
                )
            self._log_measure[q] = np.log(pq) + logw - norm + logi
        self._log_prior = self._joint_log_density(None)
        self._prior_evidence = logsumexp(self._log_prior)
        if not np.isfinite(self._prior_evidence):
            raise ValueError("No radial states remain after the imaging likelihood.")
        support = np.isfinite(logsumexp(self._log_prior, axis=0))
        if np.any(support & ~np.isfinite(self.endpoint)):
            raise ValueError("Selected states need finite radial endpoints.")
        self.endpoint = np.where(support, self.endpoint, 0.0)

    def _joint_log_density(self, residual_mag):
        joint = np.full_like(self._log_measure, -np.inf)
        for q, pq in enumerate(self.likelihood.nuisance.probability):
            if pq == 0:
                continue
            logsk, logs = self.likelihood.state_terms(
                0,
                q,
                residual_mag=residual_mag,
                state_support=self.state_probability > 0,
            )
            joint[q] = self._log_measure[q] + (logs if residual_mag is None else logsk)
        return joint

    def _summary(self, joint):
        norm = logsumexp(joint)
        if not np.isfinite(norm):
            raise ValueError("No posterior support for this radial observation.")
        p = np.exp(logsumexp(joint, axis=0) - norm)
        mean = float(p @ self.endpoint)
        sd = float(np.sqrt(p @ (self.endpoint - mean) ** 2))
        return RadialSummary(p, mean, sd)

    @property
    def prior(self):
        """Selected radial prior, including the same inner-data likelihood."""
        return self._summary(self._log_prior)

    def posterior(self, residual_mag=None):
        """Infer the inner endpoint from the observed or supplied SN residual."""
        y = (
            self.likelihood.sample.residual_mag[0]
            if residual_mag is None
            else float(residual_mag)
        )
        return self._summary(self._joint_log_density(y))

    def forecast(self, residual_grid):
        """Average posterior variance over the selected predictive density.

        :param residual_grid: Increasing residual values in magnitudes. Extend
            the range and refine spacing until density_integral and results
            converge. The result is not conditioned on a favorable noise draw.
        :return: Predictive density, its numerical integral, RMS posterior SD
            and the prior SD. No event-rate weighting is introduced.
        """
        grid = np.asarray(residual_grid, dtype=float)
        if (
            grid.ndim != 1
            or len(grid) < 3
            or not np.isfinite(grid).all()
            or np.any(np.diff(grid) <= 0)
        ):
            raise ValueError(
                "Use a finite increasing residual grid with at least three points."
            )
        density, variance = [], []
        for y in grid:
            joint = self._joint_log_density(y)
            density.append(np.exp(logsumexp(joint) - self._prior_evidence))
            variance.append(self._summary(joint).standard_deviation ** 2)
        density, variance = np.asarray(density), np.asarray(variance)
        integral = float(trapezoid(density, grid))
        if integral <= 0 or not np.isfinite(integral):
            raise ValueError("Residual grid has no predictive probability.")
        return dict(
            residual_mag=grid,
            predictive_density=density,
            density_integral=integral,
            rms_posterior_sd=float(
                np.sqrt(trapezoid(density * variance, grid) / integral)
            ),
            prior_sd=self.prior.standard_deviation,
        )
