"""Joint source-population inference with catalogue-shared calibration."""

import numpy as np
from scipy.special import logsumexp

from .likelihood import SharedNuisance, SourceLikelihood
from .population import log_probability, normalized_probability


def source_posterior(
    blocks,
    population,
    calibration,
    coordinates,
    response_index,
    log_prior_weight,
    dust_amplitude=(0.0, 0.5, 1.0),
    dust_probability=(0.25, 0.5, 0.25),
    *,
    continuation="raise",
):
    """Fit the selected source population from a stream of physical blocks.

    A node pairs one population coordinate and one whole response draw.
    Offsets within each block retain that pairing. Sum conditional source
    log likelihoods across *all* blocks before marginalizing common dust.
    This bounds photon-array memory without treating calibration as an
    independent error for each source or block.

    :param blocks: Iterable yielding (SourceSample, residual_offset_mag).
        Offsets have shape (node, source in this block), or (node,).
        Generate each block lazily to avoid building a survey-wide photon cube.
    :param population: Source MassPopulation, identical for all blocks.
    :param calibration: ResponseCalibration for this standardized observable.
    :param coordinates: Joint-node eta values, shape (node, parameter).
    :param response_index: Paired response draw per node, -1 for point response.
    :param log_prior_weight: Log joint quadrature mass, or log prior/proposal
        ratio for equally weighted importance samples. This includes response
        and offset priors once. It excludes the independent dust prior below.
    :param dust_amplitude: Independent shared dust quadrature amplitudes.
    :param dust_probability: Corresponding quadrature probabilities.
    :param continuation: 'raise' or the explicitly assumed 'unit_photon' tail.
    :return: Dictionary with source-only posterior node probability, conditional
        log likelihood (dust, node), marginalized log likelihood, and n_source.
    """
    eta = np.asarray(coordinates, float)
    logw = np.asarray(log_prior_weight, float)
    dust = np.asarray(dust_amplitude, float)
    pd = normalized_probability(dust_probability)
    if (
        eta.ndim != 2
        or logw.shape != (len(eta),)
        or not np.isfinite(eta).all()
        or np.any(np.isnan(logw) | np.isposinf(logw))
        or not np.isfinite(logsumexp(logw))
        or dust.shape != pd.shape
        or not np.isfinite(dust).all()
        or np.any(dust < 0)
    ):
        raise ValueError("Joint coordinates, prior weights and dust rule must align.")
    active = np.flatnonzero(np.isfinite(logw))
    active_dust = np.flatnonzero(pd > 0)
    conditional = np.full((len(dust), len(eta)), -np.inf)
    conditional[np.ix_(active_dust, active)] = 0.0
    n_source = 0
    for sample, offsets in blocks:
        n_source += len(sample.residual_mag)
        for row in active_dust:
            nuisance = SharedNuisance(
                response_index, np.ones(len(eta)), dust[row], offsets
            )
            conditional[row, active] += SourceLikelihood(
                sample,
                calibration,
                nuisance,
                continuation=continuation,
            ).aligned_log_likelihood(population, eta[active], active)
    if n_source == 0:
        raise ValueError("Supply at least one source block.")
    logl = logsumexp(conditional + log_probability(pd)[:, None], axis=0)
    logpost = logl + logw
    norm = logsumexp(logpost)
    if not np.isfinite(norm):
        raise ValueError("Source data have zero likelihood on the joint rule.")
    return dict(
        probability=np.exp(logpost - norm),
        conditional_log_likelihood=conditional,
        log_likelihood=logl,
        n_source=n_source,
    )
