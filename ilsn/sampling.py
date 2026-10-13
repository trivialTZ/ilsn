"""Positive integration rules for common population parameters."""

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .population import normalized_probability


def _normal_rule(covariance, n, seed, extra_dimension=0):
    covariance = np.asarray(covariance, float)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not len(covariance)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T)
    ):
        raise ValueError("A finite symmetric covariance matrix is required.")
    if not isinstance(n, (int, np.integer)) or n < 2 or n & (n - 1):
        raise ValueError("Use a power of two (at least two) Sobol nodes.")
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    if eigenvalue.min() < -1e-12 * max(1.0, abs(eigenvalue).max()):
        raise ValueError("Covariance must be positive semidefinite.")
    factor = eigenvector * np.sqrt(np.maximum(eigenvalue, 0))
    uniform = qmc.Sobol(
        len(covariance) + extra_dimension, scramble=True, seed=seed
    ).random_base2(int(np.log2(n)))
    normal = ndtri(np.clip(uniform[:, extra_dimension:], 1e-15, 1 - 1e-15)) @ factor.T
    return normal, uniform


def gaussian_nodes(mean, covariance, n=128, seed=0):
    """Scrambled Sobol nodes and equal weights for a multivariate Gaussian.

    This is finite numerical quadrature, not a convergence certificate.
    Increase n and repeat with independent seeds when reporting precision.
    Covariance may be singular, including a fixed zero contrast.
    """
    normal, _ = _normal_rule(covariance, n, seed)
    mean = np.asarray(mean, float)
    if mean.shape != (normal.shape[1],) or not np.isfinite(mean).all():
        raise ValueError("Finite mean and covariance dimensions must agree.")
    return mean + normal, np.full(n, 1 / n)


def receiving_nodes(
    source_coordinates, source_probability, contrast_covariance, n=1024, seed=0
):
    """Draw one source-posterior coordinate plus one independent common contrast.

    The first Sobol coordinate samples the discrete source posterior; the
    remaining coordinates draw the zero-mean Gaussian source--target contrast.
    The resulting eta+delta is shared by all target lenses, not redrawn per lens.
    Source posterior weights must already include shared calibration and its
    proposal correction, if an importance proposal was used.

    :return: Receiving coordinates and equal quadrature probabilities.
    """
    probability = normalized_probability(source_probability)
    eta = np.asarray(source_coordinates, float)
    delta, uniform = _normal_rule(contrast_covariance, n, seed, extra_dimension=1)
    if eta.shape != (len(probability), delta.shape[1]) or not np.isfinite(eta).all():
        raise ValueError("Source coordinates, weights and contrast must align.")
    cdf = np.cumsum(probability)
    cdf[-1] = 1.0
    index = np.searchsorted(cdf, uniform[:, 0], side="right")
    return eta[index] + delta, np.full(n, 1 / n)
