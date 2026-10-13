"""Circular source-plane geometry and single-image selection."""

import numpy as np
from scipy.optimize import brentq


def radial_caustic_radius(lens, radius_grid, central_deflection_limit):
    """Find the opposite-side image exclusion radius in the source plane.

    For a circular decreasing surface density this is the maximum of
    alpha(r)-r inside the Einstein ring, including the central limit. Refine
    every bracketed radial critical point, rather than using only a grid
    maximum. Increase the grid range/resolution to check narrow extrema.

    :param lens: Circular lens exposing deflection and radial_eigenvalue.
    :param radius_grid: Increasing positive image radii in arcsec, from inside
        the innermost radial critical point through the Einstein ring.
    :param central_deflection_limit: Explicit alpha(0+) in arcsec. It is zero
        for the finite-cusp stellar/gNFW/core family, theta_E for an SIS, and
        infinite for a point mass. A singular central limit cannot be inferred
        by evaluating the deflection at a small finite radius.
    :return: Source-plane caustic radius in arcsec.
    :rtype: float
    """
    radius = np.asarray(radius_grid, dtype=float)
    if (
        radius.ndim != 1
        or len(radius) < 3
        or not np.isfinite(radius).all()
        or np.any(radius <= 0)
        or np.any(np.diff(radius) <= 0)
    ):
        raise ValueError("Use an increasing positive radial grid.")
    if np.isnan(central_deflection_limit) or central_deflection_limit < 0:
        raise ValueError("Declare a nonnegative central deflection limit.")
    if np.isinf(central_deflection_limit):
        return float("inf")
    values = lens.deflection(radius) - radius
    derivative = -lens.radial_eigenvalue(radius)
    candidates = [float(central_deflection_limit), 0.0, float(values.max())]
    crossings = np.flatnonzero(derivative[:-1] * derivative[1:] < 0)
    for i in crossings:
        critical = brentq(
            lambda r: float(lens.radial_eigenvalue(r)),
            radius[i],
            radius[i + 1],
            xtol=1e-12,
        )
        candidates.append(float(lens.deflection(critical) - critical))
    return max(candidates)


def single_image_mask(lens, radius_arcsec, caustic_radius_arcsec):
    """Require an outer minimum with its source outside the radial caustic.

    Positive magnification alone does not exclude an opposite-side image.
    This criterion assumes the circular, decreasing-density family used to
    compute the supplied caustic; it is not a general elliptical lens solver.

    :param lens: Circular lens.
    :param radius_arcsec: Positive image radii in arcsec.
    :param caustic_radius_arcsec: Opposite-side image exclusion radius.
    :return: Boolean mask, with critical-boundary cases excluded.
    """
    radius = np.asarray(radius_arcsec, dtype=float)
    beta = radius - lens.deflection(radius)
    return (
        (lens.tangential_eigenvalue(radius) > 0)
        & (lens.radial_eigenvalue(radius) > 0)
        & (beta > caustic_radius_arcsec)
    )
