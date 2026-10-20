"""Spherical Jeans velocity dispersions of the public mass model.

The tracer is the galaxy's Sersic light without the stellar M/L gradient; the
potential is the full mass profile, including halo and signed finite cores.
Orbits have constant anisotropy beta = 1 - sigma_t^2 / sigma_r^2. For light
density nu and enclosed mass M,

    nu sigma_r^2(r) = r^(-2 beta) int_r^inf s^(2 beta) nu(s) G M(s) / s^2 ds,

and the luminosity-weighted line-of-sight dispersion within a circular
aperture of radius R follows by integrating the projection over the aperture
in closed form, with s = sqrt(max(r^2 - R^2, 0)):

    sigma_ap^2 = int nu sigma_r^2 (r - s) [1 - beta (r - s)(2 r + s) / (3 r^2)] r dr
                 / int nu (r - s) r dr.

Seeing, slit geometry and rotation are not modelled.
"""

import numpy as np
from astropy import constants as const, units as u
from scipy.integrate import cumulative_simpson, simpson

from .lens_model import CompositeProfile, GalaxyProfile, _positive, _stellar_mass_table

_G = const.G.to_value(u.kpc * u.km**2 / (u.s**2 * u.Msun))


def _light(galaxy):
    """Tuples (Re kpc, n, light fraction) of the tracer light."""
    return tuple(
        (re * galaxy.kpc_per_arcsec, n, fraction)
        for re, n, fraction in galaxy.light_components
        or ((galaxy.re_arcsec, galaxy.light_n, 1.0),)
    )


def aperture_dispersion(
    profile, aperture_kpc, anisotropy=0.0, *, inner=1e-4, outer=300.0, samples=2001
):
    """Luminosity-weighted line-of-sight velocity dispersion in an aperture.

    :param profile: GalaxyProfile or CompositeProfile; the light is that of
        the (base) galaxy.
    :param aperture_kpc: Positive circular aperture radius in kpc.
    :param anisotropy: Constant orbital anisotropy beta < 1; scalar or array.
    :param inner: Innermost radius of the integration grid, in units of the
        smallest light-component Re.
    :param outer: Outermost radius, in units of the largest component Re.
    :param samples: Number of logarithmic radius samples.
    :return: sigma_ap in km/s with the shape of ``anisotropy``; NaN where the
        integrated sigma^2 is not positive (unphysical signed profiles).
    """
    galaxy = profile.base if isinstance(profile, CompositeProfile) else profile
    if not isinstance(galaxy, GalaxyProfile):
        raise TypeError("profile must be GalaxyProfile or CompositeProfile")
    aperture = float(_positive(aperture_kpc, "aperture_kpc"))
    beta = np.asarray(anisotropy, dtype=float)
    if not np.all(np.isfinite(beta)) or np.any(beta >= 1):
        raise ValueError("anisotropy must be finite and below one")
    light = _light(galaxy)
    scales = [c[0] for c in light]
    lnr = np.linspace(
        np.log(inner * min(scales)), np.log(outer * max(scales)), int(samples)
    )
    r = np.exp(lnr)
    # nu r^3 = (1 / 4 pi) dL(<r) / dln r, light normalized to unit total.
    nu_r3 = sum(
        fraction
        * _stellar_mass_table(n, 0.0).derivative()(lnr - np.log(re))
        / (4 * np.pi)
        for re, n, fraction in light
    )
    force = nu_r3 * _G * profile.enclosed_mass_3d_msun(r) / r**2  # nu G M / r^2 * r^3
    s = np.sqrt(np.maximum(r**2 - aperture**2, 0.0))
    chord = np.where(r > aperture, aperture**2 / (r + s), r)  # r - s
    # Aperture integrals over dln r carry r^2 nu: (nu r^3) / r.
    denominator = simpson(nu_r3 / r * chord, x=lnr)
    shape = beta.shape
    out = np.empty(beta.size)
    for i, b in enumerate(beta.ravel()):
        # int_r^inf s^(2b) nu G M / s^2 ds over dln s, from the outer edge in.
        integrand = r ** (2 * b - 2) * force
        tail = cumulative_simpson(integrand[::-1], x=-lnr[::-1], initial=0)[::-1]
        nu_sr2_r3 = r ** (3 - 2 * b) * tail
        weight = chord * (1 - b * chord * (2 * r + s) / (3 * r**2))
        out[i] = simpson(nu_sr2_r3 / r * weight, x=lnr) / denominator
    with np.errstate(invalid="ignore"):
        return np.where(out > 0, np.sqrt(out), np.nan).reshape(shape)
