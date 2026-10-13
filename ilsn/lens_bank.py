"""Build physical source and target arrays from covariates and mass states.

Material admission uses an all-radius certificate for the joint gNFW halo
and signed finite cores. It is independent of the source redshift, Einstein
ring, SN image radius, and imaging observations. Geometry is solved only
after admission, and retains separate numerical validity masks.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp2f1

from .geometry import radial_caustic_radius, single_image_mask
from .lens_model import CompositeProfile, FiniteCore, GalaxyProfile, DEFAULT_COSMOLOGY
from .mass_grid import PhysicalRelations, physical_state_parameters


def _ratio_minimum(a, g, derivative=False):
    """Locate the unique halo/core density-ratio minimum on 0 < r < infinity.

    With t=r/rc, a=rc/rs, the density stationary cubic has coefficients
    (a, 4-g, -3a, -g). For negative log-density derivatives the quartic has
    coefficients (3a², (15-3g)a, g(4-g)-15a², -(3+9g)a, -g(g+2)).
    Descartes' rule gives one positive root for 0.3 <= g <= 1.8. Both ratio
    limits diverge, so this root is the global minimum, and its reciprocal
    is the finite-core/halo ratio's global maximum.
    """

    def sign(t):
        x = a * t
        slope = -(g + 3 * x) / (1 + x) + 4 * t * t / (1 + t * t)
        if derivative:
            slope += 3 * x / (g + 3 * x) - x / (1 + x) - 2 / (1 + t * t)
        return slope

    lo, hi = np.full_like(a, 1e-6), np.full_like(a, 4.0)
    for _ in range(48):
        expand = sign(hi) <= 0
        if not np.any(expand):
            break
        hi[expand] *= 2
    if np.any(sign(lo) >= 0) or np.any(sign(hi) <= 0):
        raise ValueError("Cannot bracket the exact material support-ratio minimum")
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        rising = sign(mid) > 0
        hi, lo = np.where(rising, mid, hi), np.where(rising, lo, mid)
    return np.sqrt(lo * hi)


def certify_material_support(
    halo_rs,
    halo_rhos,
    halo_slope,
    radii_kpc,
    sigma_components,
    *,
    initial_nodes=33,
    max_depth=16,
    state_block=2048,
):
    """Certify halo plus signed cores as positive/decreasing for every r > 0.

    Each finite-core/halo density and negative-derivative ratio has a single
    analytic global maximum. Its endpoint values and maximum bound every
    radial interval, including the exact zero limits at r→0 and r→infinity.
    Signed interval bounds apply to the joint sum, permitting compensation;
    ambiguous intervals bisect in log radius. This is an all-radius bound,
    not a sampled positivity test. Coincident kernels combine exactly first.

    ``support_invalid`` requires an evaluated nonpositive density or slope.
    ``support_unresolved`` distinguishes insufficient refinement from physical
    invalidity. Neither status deletes states from the declared base measure.
    Abel projection establishes projected positivity/decrease for admitted
    states; no positive distribution-function certificate follows.
    """
    rs, rho, g = np.broadcast_arrays(
        *(np.asarray(v, float) for v in (halo_rs, halo_rhos, halo_slope))
    )
    rc, sig = np.broadcast_arrays(
        np.asarray(radii_kpc, float), np.asarray(sigma_components, float)
    )
    if rc.ndim < 1 or not rc.shape[-1]:
        raise ValueError("At least one finite-core component required")
    shape = np.broadcast_shapes(rs.shape, rc.shape[:-1])
    rs, rho, g = (np.broadcast_to(a, shape).ravel() for a in (rs, rho, g))
    rc = np.broadcast_to(rc, shape + (rc.shape[-1],)).reshape((-1, rc.shape[-1]))
    sig = np.broadcast_to(sig, shape + (rc.shape[-1],)).reshape(rc.shape).copy()
    for i in range(rc.shape[-1]):
        for j in range(i):
            same = rc[:, i] == rc[:, j]
            sig[:, j] += np.where(same, sig[:, i], 0.0)
            sig[:, i] = np.where(same, 0.0, sig[:, i])
    if (
        not all(np.all(np.isfinite(a)) for a in (rs, rho, g, rc, sig))
        or np.any(rs <= 0)
        or np.any(rho <= 0)
        or np.any(rc <= 0)
        or np.any((g < 0.3) | (g > 1.8))
        or not isinstance(initial_nodes, int)
        or initial_nodes < 9
        or not isinstance(max_depth, int)
        or max_depth < 1
        or not isinstance(state_block, int)
        or state_block < 1
    ):
        raise ValueError(
            "Finite supported gNFW/core parameters and positive certification resolution required"
        )
    invalid, unresolved = np.zeros(len(rs), bool), np.zeros(len(rs), bool)
    margin, depth_used = np.full((len(rs), 2), np.inf), np.zeros((len(rs), 2), int)
    for first in range(0, len(rs), state_block):
        block = slice(first, min(first + state_block, len(rs)))
        sr, rh, gg, cr, sg = rs[block], rho[block], g[block], rc[block], sig[block]
        size, a = len(sr), cr / sr[:, None]
        for derivative in (False, True):
            peak = cr * _ratio_minimum(
                a, np.broadcast_to(gg[:, None], a.shape), derivative
            )

            def ratios(r, ii):
                boundary = (r == 0) | np.isinf(r)
                safe = np.where(boundary, sr[ii], r)
                x, t = safe[:, None] / sr[ii, None], safe[:, None] / cr[ii]
                logv = (
                    np.log(2 / (np.pi * cr[ii] * rh[ii, None]))
                    + gg[ii, None] * np.log(x)
                    + (3 - gg[ii, None]) * np.log1p(x)
                    - 2 * np.log1p(t * t)
                )
                value = np.exp(logv)
                if derivative:
                    value *= (4 * t * t / (1 + t * t)) / (
                        (gg[ii, None] + 3 * x) / (1 + x)
                    )
                return np.where(boundary[:, None], 0.0, value)

            all_ids = np.repeat(np.arange(size), cr.shape[1])
            pv = ratios(peak.ravel(), all_ids).reshape((size, cr.shape[1], cr.shape[1]))
            peak_value = np.diagonal(pv, axis1=1, axis2=2)
            low, high = peak.min(1) * 1e-8, peak.max(1) * 1e8
            grid = low[:, None] * (high / low)[:, None] ** np.linspace(
                0, 1, initial_nodes
            )
            grid = np.sort(
                np.concatenate(
                    (np.zeros((size, 1)), grid, peak, np.full((size, 1), np.inf)),
                    axis=1,
                ),
                axis=1,
            )
            ii = np.repeat(np.arange(size), grid.shape[1] - 1)
            lo, hi = grid[:, :-1].ravel(), grid[:, 1:].ravel()
            for depth in range(max_depth + 1):
                left, right = ratios(lo, ii), ratios(hi, ii)
                minimum = np.minimum(left, right)
                contains = (peak[ii] >= lo[:, None]) & (peak[ii] <= hi[:, None])
                maximum = np.maximum(
                    np.maximum(left, right), np.where(contains, peak_value[ii], 0.0)
                )
                coeff = sg[ii]
                lower = 1 + np.sum(
                    np.where(coeff >= 0, coeff * minimum, coeff * maximum), axis=1
                )
                bad = (1 + np.sum(coeff * left, axis=1) <= 0) | (
                    1 + np.sum(coeff * right, axis=1) <= 0
                )
                invalid[first + np.unique(ii[bad])] = True
                accepted = lower > 2e-13
                np.minimum.at(
                    margin[:, int(derivative)], first + ii[accepted], lower[accepted]
                )
                active = ~accepted & ~invalid[first + ii]
                if not np.any(active):
                    break
                ids, l, h = ii[active], lo[active], hi[active]
                depth_used[first + np.unique(ids), int(derivative)] = depth
                if depth == max_depth:
                    unresolved[first + np.unique(ids)] = True
                    break
                with np.errstate(invalid="ignore"):
                    mid = np.sqrt(l * h)
                mid = np.where(l == 0, h / 4, np.where(np.isinf(h), l * 4, mid))
                ii = np.repeat(ids, 2)
                lo, hi = (
                    np.column_stack((l, mid)).ravel(),
                    np.column_stack((mid, h)).ravel(),
                )
    unresolved &= ~invalid
    valid = ~invalid & ~unresolved
    return dict(
        physical_valid=valid.reshape(shape),
        support_invalid=invalid.reshape(shape),
        support_unresolved=unresolved.reshape(shape),
        support_density_ratio_lower_bound=margin[:, 0].reshape(shape),
        support_derivative_ratio_lower_bound=margin[:, 1].reshape(shape),
        support_refinement_depth=depth_used.reshape(shape + (2,)),
    )


def _halo_scales(design, parameters):
    mass, c = 10 ** parameters["log_halo_mass"], parameters["concentration"]
    rhoc = DEFAULT_COSMOLOGY.critical_density(design.z_lens).to_value("Msun/kpc3")
    rs = (3 * mass / (800 * np.pi * rhoc[:, None])) ** (1 / 3) / c
    g = parameters["halo_slope"]
    a = 3 - g
    norm = c**a * hyp2f1(a, a, a + 1, -c) / a
    return rs, mass / (4 * np.pi * rs**3 * norm)


def profile_at_state(
    design, grid, object_index, state_index, relations=PhysicalRelations()
):
    """Construct one independently normalized public physical profile."""
    p = physical_state_parameters(design, grid, relations, state_slice=[state_index])
    j = object_index
    base = GalaxyProfile(
        design.z_lens[j],
        p["re_arcsec"][j],
        10 ** p["log_stellar_mass"][j, 0],
        10 ** p["log_halo_mass"][j, 0],
        p["concentration"][j, 0],
        design.light_n[j],
        halo_inner_slope=p["halo_slope"][j, 0],
        stellar_ml_gradient=p["stellar_gradient"][j, 0],
    )
    cores = tuple(
        FiniteCore(r, sigma)
        for r, sigma in zip(
            p["core_radii_kpc"][j, 0], p["sigma_core_components_msun_kpc2"][j, 0]
        )
    )
    return CompositeProfile(base, cores)


def build_lens_bank(
    design,
    grid,
    relations=PhysicalRelations(),
    *,
    source=True,
    target=True,
    state_slice=None,
    radial_nodes=129,
    support_options=None,
):
    """Return object by state arrays for absolute-SN and target conditioning.

    ``source=True`` computes kappa, mean_kappa and stellar kappa at the SN
    image radius and a single-image mask. ``target=True`` computes derived
    Einstein-ring normalization/curvature. The caustic area and material and
    geometry masks are computed in both modes. ``source=False`` permits a
    target design without any hypothetical SN radius.

    This transparent scalar implementation suits small surveys and numerical
    controls. The full 72,900-state grid is expensive; use ``state_slice`` to
    build/save blocks externally, and check radial resolution for new designs.
    Excluded source states have zero response placeholders with false masks.
    Unresolved material certificates remain explicit and must be resolved
    before interpreting a selected-population result.
    """
    if not isinstance(radial_nodes, int) or radial_nodes < 33:
        raise ValueError("At least 33 radial nodes required")
    if source and design.radius_arcsec is None:
        raise ValueError("Source bank requires radius_arcsec")
    if not source and not target:
        raise ValueError("Request source or target predictions")
    states = (
        np.arange(len(grid))
        if state_slice is None
        else np.atleast_1d(np.arange(len(grid))[state_slice])
    )
    p = physical_state_parameters(design, grid, relations, state_slice=states)
    rs, rho = _halo_scales(design, p)
    support = certify_material_support(
        rs,
        rho,
        p["halo_slope"],
        p["core_radii_kpc"],
        p["sigma_core_components_msun_kpc2"],
        **(support_options or {}),
    )
    shape = p["log_halo_mass"].shape
    out = {name: np.zeros(shape) for name in ("kappa", "mean_kappa", "kappa_star")}
    out.update(
        {
            name: np.full(shape, np.nan)
            for name in (
                "theta_e_arcsec",
                "xi_E",
                "s_E",
                "theta_e_over_re_design",
                "caustic_area_arcsec2",
                "radial_caustic_arcsec",
                "lnmu",
                "beta_arcsec",
            )
        }
    )
    out.update(
        {
            name: np.zeros(shape, bool)
            for name in (
                "single_image",
                "geometry_valid",
                "projection_valid",
                "endpoint_valid",
            )
        }
    )
    out.update(support)
    for j, k in zip(*np.where(support["physical_valid"])):
        base = GalaxyProfile(
            design.z_lens[j],
            p["re_arcsec"][j],
            10 ** p["log_stellar_mass"][j, k],
            10 ** p["log_halo_mass"][j, k],
            p["concentration"][j, k],
            design.light_n[j],
            halo_inner_slope=p["halo_slope"][j, k],
            stellar_ml_gradient=p["stellar_gradient"][j, k],
        )
        cores = tuple(
            FiniteCore(r, sigma)
            for r, sigma in zip(
                p["core_radii_kpc"][j, k], p["sigma_core_components_msun_kpc2"][j, k]
            )
        )
        lens = CompositeProfile(base, cores).lens(design.z_source[j])
        lower = lens.minimum_radius_arcsec * (1 + 1e-6)
        outer = design.radius_arcsec[j] if source else design.re_arcsec[j]
        upper = max(20 * base.re_arcsec, 2 * outer)
        maximum = lens.maximum_radius_arcsec * (1 - 1e-6)
        while upper < maximum and (
            lens.radial_eigenvalue(upper) <= 0 or lens.mean_kappa(upper) >= 1
        ):
            upper *= 4
        upper = min(upper, maximum)
        radii = np.geomspace(lower, upper, radial_nodes)
        radial = lens.radial_eigenvalue(radii)
        crossings = np.flatnonzero((radial[:-1] < 0) & (radial[1:] >= 0))
        good = (
            radial[0] < 0
            and radial[-1] > 0
            and lens.mean_kappa(upper) < 1
            and crossings.size > 0
        )
        out["geometry_valid"][j, k] = good
        if good:
            caustic = radial_caustic_radius(lens, radii, central_deflection_limit=0.0)
            out["radial_caustic_arcsec"][j, k] = caustic
            out["caustic_area_arcsec2"][j, k] = np.pi * caustic**2
        if source and lower <= outer <= maximum:
            kappa, mean = float(lens.kappa(outer)), float(lens.mean_kappa(outer))
            radius_kpc = outer * base.kpc_per_arcsec
            out["kappa"][j, k], out["mean_kappa"][j, k] = kappa, mean
            out["kappa_star"][j, k] = (
                base._stars(radius_kpc)[0] / lens.sigma_crit_msun_kpc2
            )
            out["projection_valid"][j, k] = True
            determinant = (1 - mean) * (1 - 2 * kappa + mean)
            out["beta_arcsec"][j, k] = outer * (1 - mean)
            if 1 - mean > 0 and 1 - 2 * kappa + mean > 0:
                out["lnmu"][j, k] = -np.log(determinant)
            if good:
                out["single_image"][j, k] = single_image_mask(lens, outer, caustic)
        if target and lens.mean_kappa(lower) > 1 and lens.mean_kappa(upper) < 1:
            theta_e = brentq(
                lambda theta: float(lens.mean_kappa(theta) - 1),
                lower,
                upper,
                xtol=1e-11,
            )
            kappa_e, mean_e = float(lens.kappa(theta_e)), float(
                lens.mean_kappa(theta_e)
            )
            radial_e = 1 - 2 * kappa_e + mean_e
            if 0 <= kappa_e < 1 and radial_e > 1e-10 and abs(mean_e - 1) < 1e-8:
                out["theta_e_arcsec"][j, k] = theta_e
                out["s_E"][j, k] = 1 - kappa_e
                out["xi_E"][j, k] = float(lens.xi_invariant(theta_e))
                out["theta_e_over_re_design"][j, k] = theta_e / design.re_arcsec[j]
                out["endpoint_valid"][j, k] = True
    return out
