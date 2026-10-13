"""From lens predictions to SN photons and strong-lens observations."""

import numpy as np

from .lens_model import DEFAULT_COSMOLOGY
from .likelihood import SourceSample


def paired_los(kappa_pools, gamma_pools, rows=8, orientations=4):
    """Compress paired empirical convergence/shear samples and shear angles.

    Select actual rows at midpoint ranks of sorted convergence. Taking the
    same row in both pools retains their empirical correlation. Uniform shear
    orientations cover [0, pi). Refine both counts for a convergence check.

    :param kappa_pools: External convergence samples, shape (cell, sample).
    :param gamma_pools: Paired external shear amplitudes, same shape.
    :param rows: Number of paired sample rows retained per cell.
    :param orientations: Number of shear orientations per paired sample.
    :return: Dictionary of kappa, gamma, cos2phi and quadrature probability.
    """
    kappa, gamma = np.asarray(kappa_pools, float), np.asarray(gamma_pools, float)
    if (
        kappa.ndim != 2
        or gamma.shape != kappa.shape
        or not np.isfinite(kappa).all()
        or not np.isfinite(gamma).all()
        or np.any(gamma < 0)
        or not isinstance(rows, (int, np.integer))
        or not 1 <= rows <= kappa.shape[1]
        or not isinstance(orientations, (int, np.integer))
        or orientations < 1
    ):
        raise ValueError(
            "Finite paired LOS pools and positive quadrature sizes required."
        )
    rank = np.floor((np.arange(rows) + 0.5) * kappa.shape[1] / rows).astype(int)
    index = np.argsort(kappa, axis=1, kind="stable")[:, rank]
    angle = (np.arange(orientations) + 0.5) * np.pi / orientations
    return dict(
        kappa=np.repeat(np.take_along_axis(kappa, index, axis=1), orientations, axis=1),
        gamma=np.repeat(np.take_along_axis(gamma, index, axis=1), orientations, axis=1),
        cos2phi=np.tile(np.cos(2 * angle), rows),
        probability=np.full(rows * orientations, 1 / (rows * orientations)),
    )


def source_sample(
    residual_mag,
    redshift,
    physics,
    *,
    radius_arcsec,
    re_arcsec,
    los=None,
    los_cell=None,
    micro_scale=0.2,
):
    """Construct the actual ilSN photon model from a physical lens bank.

    The main-lens single-image condition is supplemented by a positive outer
    Jacobian in the local external convergence/shear approximation. This does
    not solve full multi-plane image topology. Dust shifts photons before
    delivery; the prescribed stellar microlensing variance is added after it.

    :param residual_mag: Fixed-coordinate residuals, shape (source,).
    :param redshift: Response-conditioning redshifts, shape (source,).
    :param physics: Lens-bank arrays kappa, mean_kappa, kappa_star and masks
        physical_valid, single_image, geometry_valid, projection_valid.
    :param radius_arcsec: Fixed observed SN angular radius per source.
    :param re_arcsec: Assigned foreground light half-radius per source.
    :param los: Rule from paired_los; None explicitly assumes no external LOS.
    :param los_cell: LOS cell index for each source, required when los is given.
    :param micro_scale: Prescribed coefficient in variance=micro_scale**2*kappa_star.
    :return: SourceSample, including unit-amplitude dust template
        ``0.01*(2*Re/radius)**2`` in magnitudes.
    """
    kappa = np.asarray(physics["kappa"], float)
    mean = np.asarray(physics["mean_kappa"], float)
    stars = np.asarray(physics["kappa_star"], float)
    if kappa.ndim != 2 or mean.shape != kappa.shape or stars.shape != kappa.shape:
        raise ValueError("Source convergence arrays must have shape (source, state).")
    n = len(kappa)
    radius, re = np.asarray(radius_arcsec, float), np.asarray(re_arcsec, float)
    if (
        radius.shape != (n,)
        or re.shape != (n,)
        or not np.isfinite(radius).all()
        or not np.isfinite(re).all()
        or np.any(radius <= 0)
        or np.any(re <= 0)
        or not np.isfinite(micro_scale)
        or micro_scale < 0
    ):
        raise ValueError(
            "Positive aligned angular radii and nonnegative micro scale required."
        )
    valid = np.ones(kappa.shape, bool)
    for key in ("physical_valid", "single_image", "geometry_valid", "projection_valid"):
        mask = np.asarray(physics[key], bool)
        if mask.shape != kappa.shape:
            raise ValueError("Physical and geometry masks must align with mass states.")
        valid &= mask
    if np.any(
        valid
        & (~np.isfinite(kappa) | ~np.isfinite(mean) | ~np.isfinite(stars) | (stars < 0))
    ):
        raise ValueError(
            "Admitted source states require finite convergence and nonnegative stellar convergence."
        )
    if los is None:
        external_kappa = external_gamma = np.zeros((n, 1))
        cosine, probability = np.ones(1), np.ones(1)
    else:
        cell = np.asarray(los_cell)
        pools = np.asarray(los["kappa"], float)
        gamma = np.asarray(los["gamma"], float)
        cosine, probability = np.asarray(los["cos2phi"], float), np.asarray(
            los["probability"], float
        )
        if (
            pools.ndim != 2
            or gamma.shape != pools.shape
            or cosine.shape != (pools.shape[1],)
            or probability.shape != cosine.shape
            or cell.shape != (n,)
            or not np.issubdtype(cell.dtype, np.integer)
            or np.any(cell < 0)
            or np.any(cell >= len(pools))
            or not np.isfinite(pools).all()
            or not np.isfinite(gamma).all()
            or np.any(gamma < 0)
            or not np.isfinite(cosine).all()
            or np.any(abs(cosine) > 1)
            or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not np.isclose(probability.sum(), 1)
        ):
            raise ValueError("Paired LOS rule and source cell indices must align.")
        external_kappa, external_gamma = pools[cell], gamma[cell]
    internal_shear = (mean - kappa)[..., None]
    a = 1 - kappa[..., None] - external_kappa[:, None, :]
    shear2 = (
        internal_shear**2
        + external_gamma[:, None, :] ** 2
        + 2 * internal_shear * external_gamma[:, None, :] * cosine
    )
    shear = np.sqrt(np.maximum(shear2, 0))
    determinant = (a - shear) * (a + shear)
    admitted = valid[..., None] & (a - shear > 0) & np.isfinite(determinant)
    shift = 2.5 / np.log(10) * np.log(np.where(admitted, determinant, 1))
    variance = micro_scale**2 * np.where(valid, stars, 0)[..., None]
    dust = 0.01 * (2 * re / radius) ** 2
    return SourceSample(
        residual_mag,
        redshift,
        shift,
        component_weight=admitted * probability[None, None, :],
        extra_variance_mag2=variance,
        dust_template_mag=dust[:, None, None],
    )


def td_selection(
    theta_e_arcsec,
    caustic_area_arcsec2,
    z_lens,
    z_source,
    physical_valid=None,
    velocity_range=(150.0, 350.0),
    cosmology=DEFAULT_COSMOLOGY,
):
    """Conditional TD-like caustic-area selection with an SIS velocity proxy.

    The velocity cut is a geometric simulation prescription, not measured
    stellar dynamics or a complete time-delay survey discovery probability.
    Set velocity_range=None for an uncut caustic-area comparison.

    :param theta_e_arcsec: Model Einstein radii, shape (target, state).
    :param caustic_area_arcsec2: Source-plane radial-caustic area, same shape.
    :param z_lens: Foreground redshifts, shape (target,).
    :param z_source: Background redshifts, shape (target,).
    :param physical_valid: Optional physical admission mask.
    :param velocity_range: Inclusive SIS-equivalent velocity range in km/s.
    :param cosmology: Astropy cosmology for angular-diameter distance ratios.
    :return: Nonnegative selection weights, with zero on invalid/cut geometry.
        Missing Einstein/caustic geometry has zero selection mass, without
        changing the supplied physical admission mask or base state measure.
        Check the bank's separate geometry_valid and endpoint_valid flags
        before interpreting results; this function cannot diagnose a failed
        numerical curve calculation from absent geometry alone.
    """
    theta, area = np.asarray(theta_e_arcsec, float), np.asarray(
        caustic_area_arcsec2, float
    )
    zl, zs = np.asarray(z_lens, float), np.asarray(z_source, float)
    valid = (
        np.ones(theta.shape, bool)
        if physical_valid is None
        else np.asarray(physical_valid, bool)
    )
    if (
        theta.ndim != 2
        or area.shape != theta.shape
        or valid.shape != theta.shape
        or zl.shape != (len(theta),)
        or zs.shape != zl.shape
        or not np.isfinite(zl).all()
        or not np.isfinite(zs).all()
        or np.any(zl <= 0)
        or np.any(zs <= zl)
    ):
        raise ValueError(
            "Target geometry and finite foreground/source redshifts must align."
        )
    valid = valid & np.isfinite(theta) & (theta > 0) & np.isfinite(area) & (area >= 0)
    if velocity_range is not None:
        low, high = velocity_range
        if not np.isfinite([low, high]).all() or low < 0 or high < low:
            raise ValueError("Ordered nonnegative velocity limits required.")
        ratio = (
            cosmology.angular_diameter_distance(zs).value
            / cosmology.angular_diameter_distance_z1z2(zl, zs).value
        )
        velocity = 299792.458 * np.sqrt(
            np.where(valid, theta, 0) * ratio[:, None] / (4 * 180 * 3600)
        )
        valid = valid & (velocity >= low) & (velocity <= high)
    return np.where(valid, area, 0.0)


def imaging_log_likelihood(
    theta_e, xi_e, observed_theta_e, observed_xi_e, sigma_theta_e, sigma_xi_e
):
    """Gaussian compressed inner-imaging likelihood, up to common constants.

    Predictions are (target, state), observations/errors are (target,).
    theta_E and its error are in arcsec; xi_E is dimensionless. Errors are
    fixed by the observation, never recomputed from a candidate mass model.
    This summary is not a pixel-level image likelihood. An external image or
    dynamics log likelihood can instead be supplied directly to TargetSample.
    """
    theta, xi = np.asarray(theta_e, float), np.asarray(xi_e, float)
    ot, ox, st, sx = (
        np.asarray(v, float)
        for v in (observed_theta_e, observed_xi_e, sigma_theta_e, sigma_xi_e)
    )
    if (
        theta.ndim != 2
        or xi.shape != theta.shape
        or any(
            v.shape != (len(theta),) or not np.isfinite(v).all()
            for v in (ot, ox, st, sx)
        )
        or np.any(ot <= 0)
        or np.any(st <= 0)
        or np.any(sx <= 0)
    ):
        raise ValueError(
            "Aligned target observations and positive fixed errors required."
        )
    value = -0.5 * ((theta - ot[:, None]) / st[:, None]) ** 2
    value -= 0.5 * ((xi - ox[:, None]) / sx[:, None]) ** 2
    return np.where(np.isfinite(value), value, -np.inf)
