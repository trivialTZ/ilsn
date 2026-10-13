"""Conditional population sharing between ilSN foregrounds and strong lenses."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


def normalized_probability(values):
    """Normalize nonnegative finite quadrature weights."""
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or np.any(values < 0)
        or not np.isfinite(values).all()
        or not np.any(values > 0)
    ):
        raise ValueError(
            "Weights must be a finite nonnegative vector with positive sum."
        )
    scaled = values / values.max()
    return scaled / scaled.sum()


def log_probability(values):
    """Log weights with exact zero support retained as minus infinity."""
    with np.errstate(divide="ignore"):
        return np.log(values)


class MassPopulation:
    """Exponential family on an explicitly supplied mass-model quadrature.

    ``p(state | eta)`` is proportional to ``base_probability * exp(scores @ eta)``.
    Scores can describe stellar mass, halo structure, core amplitude and their
    correlations. Their choice and base measure are physical population priors.
    Extra radial-model states may share the same scores without being collapsed.

    :param base_probability: Prior quadrature weights, shape (state,).
    :param scores: Dimensionless population scores, shape (state, parameter).
    """

    def __init__(self, base_probability, scores):
        self.base_probability = normalized_probability(base_probability)
        self.scores = np.asarray(scores, dtype=float)
        if (
            self.scores.ndim != 2
            or self.scores.shape[0] != len(self.base_probability)
            or not np.isfinite(self.scores).all()
        ):
            raise ValueError("Scores must be finite with one row per mass state.")

    def log_probabilities(self, coordinates):
        """Return log state probabilities, shape (population node, state).

        :param coordinates: Population coordinates, shape (node, parameter).
        :return: Row-normalized log probabilities.
        """
        coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != self.scores.shape[1]
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError("Population coordinates must align with score columns.")
        logp = (
            log_probability(self.base_probability)[None, :]
            + coordinates @ self.scores.T
        )
        return logp - logsumexp(logp, axis=1, keepdims=True)

    def probabilities(self, coordinates):
        """Return state probabilities, shape (population node, state)."""
        return np.exp(self.log_probabilities(coordinates))


class TargetSample:
    """Independent strong lenses, conditionally sharing one mass population.

    Each row is a lens without an ilSN. Columns refer to the same physical
    state coordinates as the source population, evaluated at that lens's own
    covariates. The selection weight may be a caustic area rather than a
    probability. Selection defines a conditional population; it does not
    supply an event-count likelihood. Imaging or dynamics likelihood evidence
    is retained when updating common population parameters.

    :param endpoint: Inner quantity (usually s_E), shape (target, state).
    :param selection: Nonnegative target selection weights of the same shape.
    :param log_likelihood: Optional per-target imaging/dynamics log likelihood.
        Zero means no additional observation. Minus infinity excludes a state.
    :param physical_mask: Optional boolean physical-admission mask.
    """

    def __init__(self, endpoint, selection, log_likelihood=None, physical_mask=None):
        self.endpoint = np.asarray(endpoint, dtype=float)
        self.selection = np.asarray(selection, dtype=float)
        if (
            self.endpoint.ndim != 2
            or min(self.endpoint.shape) < 1
            or self.selection.shape != self.endpoint.shape
        ):
            raise ValueError(
                "Endpoints and selection need matching (target, state) arrays."
            )
        valid = (
            np.ones(self.endpoint.shape, dtype=bool)
            if physical_mask is None
            else np.asarray(physical_mask, dtype=bool)
        )
        if valid.shape != self.endpoint.shape or np.any(
            valid & (~np.isfinite(self.selection) | (self.selection < 0))
        ):
            raise ValueError(
                "Admitted target states need finite nonnegative selection."
            )
        self.selection = np.where(valid, self.selection, 0.0)
        self.log_likelihood = (
            np.zeros_like(self.endpoint)
            if log_likelihood is None
            else np.asarray(log_likelihood, dtype=float)
        )
        support = self.selection > 0
        if self.log_likelihood.shape != self.endpoint.shape or np.any(
            support
            & (
                ~np.isfinite(self.endpoint)
                | np.isnan(self.log_likelihood)
                | np.isposinf(self.log_likelihood)
            )
        ):
            raise ValueError(
                "Selected target states need finite endpoints and valid log likelihoods."
            )
        self.endpoint = np.where(support, self.endpoint, 0.0)
        self.log_likelihood = np.where(support, self.log_likelihood, -np.inf)

    def moments(self, population, coordinates, chunk_size=256):
        """Integrate each target's mass states at fixed common coordinates.

        :return: Dictionary of conditional means, variances, and the summed
            target log evidence; arrays have axes (node, target) or (node,).
        """
        coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))
        if chunk_size < 1 or self.endpoint.shape[1] != len(population.base_probability):
            raise ValueError(
                "Positive chunk size and matching mass states are required."
            )
        mean = np.empty((len(coordinates), len(self.endpoint)))
        variance = np.empty_like(mean)
        evidence = np.zeros(len(coordinates))
        for start in range(0, len(coordinates), chunk_size):
            stop = min(start + chunk_size, len(coordinates))
            logp = population.log_probabilities(coordinates[start:stop])
            for j, (values, selection, logl) in enumerate(
                zip(self.endpoint, self.selection, self.log_likelihood)
            ):
                selected = logp + log_probability(selection)[None, :]
                norm = logsumexp(selected, axis=1)
                if not np.isfinite(norm).all():
                    raise ValueError("A target has no selected physical support.")
                joint = selected + logl[None, :]
                postnorm = logsumexp(joint, axis=1)
                if not np.isfinite(postnorm).all():
                    raise ValueError(
                        "A target likelihood excludes every supported state."
                    )
                probability = np.exp(joint - postnorm[:, None])
                m = probability @ values
                mean[start:stop, j] = m
                variance[start:stop, j] = np.sum(
                    probability * (values[None, :] - m[:, None]) ** 2, axis=1
                )
                evidence[start:stop] += postnorm - norm
        return dict(mean=mean, variance=variance, log_evidence=evidence)


@dataclass
class TargetSummary:
    """Target means and full covariance after common-parameter integration.

    ``shared_covariance`` includes uncertain source population and common
    source--target contrast. ``individual_variance`` retains each target's
    residual mass-state uncertainty. Targets are independent only at fixed
    common parameters.
    """

    mean: np.ndarray
    shared_covariance: np.ndarray
    individual_variance: np.ndarray

    @property
    def covariance(self):
        """Full target covariance, including individual diagonal variance."""
        return self.shared_covariance + np.diag(self.individual_variance)

    @property
    def standard_deviation(self):
        """Marginal standard deviation of each target."""
        return np.sqrt(np.diag(self.covariance))

    @property
    def catalogue_mean(self):
        """Mean of the finite target catalogue's average endpoint."""
        return float(self.mean.mean())

    @property
    def catalogue_mean_sd(self):
        """Uncertainty of the finite catalogue mean, retaining covariance."""
        return float(np.sqrt(max(self.covariance.sum(), 0.0)) / len(self.mean))


def summarize_targets(moments, probability):
    """Apply the law of total covariance to conditional target moments."""
    probability = normalized_probability(probability)
    means = np.asarray(moments["mean"])
    if len(probability) != len(means):
        raise ValueError("Node weights must align with target moments.")
    mean = probability @ means
    centered = means - mean
    shared = (centered.T * probability) @ centered
    return TargetSummary(mean, shared, probability @ moments["variance"])


def target_readout(
    population, targets, receiving_coordinates, node_probability, chunk_size=256
):
    """Condition a paired receiving rule on all target observations.

    Each row contains one common ``eta + delta`` vector for the whole target
    catalogue. Source and target mass grids may differ, provided their score
    coordinates have the same fixed physical definition. This interface
    avoids a Cartesian product of population and contrast quadratures.

    :param population: Target mass population on its own physical grid.
    :param targets: TargetSample with selection and imaging/dynamics likelihood.
    :param receiving_coordinates: Common receiving nodes, shape (node, parameter).
    :param node_probability: Probability before conditioning on target data.
    :param chunk_size: Number of common nodes evaluated at once.
    :return: TargetSummary and target-conditioned node probabilities.
    """
    probability = normalized_probability(node_probability)
    moments = targets.moments(population, receiving_coordinates, chunk_size)
    if moments["log_evidence"].shape != probability.shape:
        raise ValueError("Receiving nodes and their probabilities must align.")
    logp = log_probability(probability) + moments["log_evidence"]
    probability = np.exp(logp - logsumexp(logp))
    return summarize_targets(moments, probability), probability


@dataclass
class TransferResult:
    """Comparison before/after ilSN data with the same target observations."""

    prior: TargetSummary
    posterior: TargetSummary
    source_coordinates: np.ndarray
    target_coordinates: np.ndarray
    prior_node_probability: np.ndarray
    posterior_node_probability: np.ndarray


def population_transfer(
    population,
    targets,
    source_coordinates,
    source_log_likelihood,
    source_probability,
    contrast_coordinates,
    contrast_probability,
    chunk_size=256,
):
    """Transfer source information while retaining target-population mismatch.

    One common contrast vector delta is drawn for the entire target sample;
    target coordinates are eta + delta. Its input prior is independent of
    the source data. Both reported distributions include all target imaging
    or dynamics likelihoods. The posterior additionally includes the ilSN
    source likelihood, already marginalized over survey-shared calibration.

    :param population: :class:`MassPopulation` on common mass coordinates.
    :param targets: :class:`TargetSample` evaluated at target covariates.
    :param source_coordinates: Eta quadrature, shape (source node, parameter).
    :param source_log_likelihood: Source-survey log likelihood per eta node.
    :param source_probability: Prior quadrature probability per eta node.
    :param contrast_coordinates: Delta quadrature, shape (contrast node, parameter).
    :param contrast_probability: Probability of each common delta node.
    :param chunk_size: Number of shared nodes evaluated at once.
    :return: :class:`TransferResult`, including the joint shared-node weights.
    """
    eta, delta = (
        np.atleast_2d(np.asarray(v, dtype=float))
        for v in (source_coordinates, contrast_coordinates)
    )
    pe, pd = normalized_probability(source_probability), normalized_probability(
        contrast_probability
    )
    logl = np.asarray(source_log_likelihood, dtype=float)
    if (
        eta.ndim != 2
        or delta.ndim != 2
        or eta.shape[1] != delta.shape[1]
        or len(eta) != len(pe)
        or len(delta) != len(pd)
        or logl.shape != pe.shape
        or np.any(np.isnan(logl) | np.isposinf(logl))
    ):
        raise ValueError("Source/contrast quadrature and likelihood arrays must align.")
    source_nodes = np.repeat(eta, len(delta), axis=0)
    receiving_nodes = source_nodes + np.tile(delta, (len(eta), 1))
    prior = (pe[:, None] * pd[None, :]).ravel()
    moments = targets.moments(population, receiving_nodes, chunk_size=chunk_size)
    log_prior = log_probability(prior) + moments["log_evidence"]
    log_posterior = log_prior + np.repeat(logl, len(delta))
    if not np.isfinite(logsumexp(log_posterior)):
        raise ValueError(
            "Source data have zero likelihood on the population quadrature."
        )
    before = np.exp(log_prior - logsumexp(log_prior))
    after = np.exp(log_posterior - logsumexp(log_posterior))
    return TransferResult(
        summarize_targets(moments, before),
        summarize_targets(moments, after),
        source_nodes,
        receiving_nodes,
        before,
        after,
    )
