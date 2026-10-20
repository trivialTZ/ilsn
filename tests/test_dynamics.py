"""Jeans aperture dispersions against independent quadrature and the virial limit."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import gamma, gammaincinv

from ilsn.dynamics import _G, aperture_dispersion
from ilsn.lens_model import (
    CompositeProfile,
    FiniteCore,
    GalaxyProfile,
    _stellar_mass_table,
)


@pytest.fixture
def galaxy():
    return GalaxyProfile(
        z_l=0.3,
        re_arcsec=1.0,
        stellar_mass_msun=1.5e11,
        halo_mass_msun=1e13,
        concentration=7.0,
        stellar_ml_gradient=-0.2,
    )


def _nu(galaxy, r):
    """Unit-luminosity Sersic light density without the M/L gradient."""
    table = _stellar_mass_table(galaxy.light_n, 0.0).derivative()
    return table(np.log(r / galaxy.re_kpc)) / (4 * np.pi * r**3)


def test_large_aperture_is_virial_and_independent_of_anisotropy(galaxy):
    """Sum over the whole galaxy: <sigma_los^2> = (1/3) int nu G M / r d^3r."""
    re = galaxy.re_kpc

    def integrand(lnr):
        r = np.exp(lnr)
        mass = galaxy.enclosed_mass_3d_msun(np.array([r]))[0]
        return 4 * np.pi * r**3 * _nu(galaxy, r) * _G * mass / r

    virial = quad(integrand, np.log(1e-4 * re), np.log(300 * re), limit=400)[0] / 3
    sigma = aperture_dispersion(galaxy, 290 * re, [-0.5, 0.0, 0.5])
    np.testing.assert_allclose(sigma, np.sqrt(virial), rtol=1e-5)


@pytest.mark.parametrize("beta", [-0.3, 0.4])
def test_aperture_matches_direct_projection(galaxy, beta):
    """Jeans solution projected along each sightline, then averaged over R."""
    re, aperture = galaxy.re_kpc, 0.7 * galaxy.re_kpc
    profile = CompositeProfile(galaxy, (FiniteCore.from_total_mass(5 * re, 5e10),))

    def nu_g_m(s):
        mass = profile.enclosed_mass_3d_msun(np.array([s]))[0]
        return _nu(galaxy, s) * _G * mass / s**2

    nodes = np.geomspace(1e-4 * re, 200 * re, 241)
    pressure = [
        r ** (-2 * beta)
        * quad(
            lambda u: np.exp((2 * beta + 1) * u) * nu_g_m(np.exp(u)),
            np.log(r),
            np.log(300 * re),
            limit=200,
        )[0]
        for r in nodes
    ]
    log_pressure = CubicSpline(np.log(nodes), np.log(pressure))

    def los(R):
        # sigma_los^2 Sigma(R) = 2 int (1 - beta R^2 / r^2) nu sigma_r^2 dz.
        def f(z):
            r = np.hypot(R, z)
            return (1 - beta * R**2 / r**2) * np.exp(log_pressure(np.log(r)))

        return 2 * quad(f, 0, np.sqrt((200 * re) ** 2 - R**2), limit=200)[0]

    n = galaxy.light_n
    b = gammaincinv(2 * n, 0.5)
    surface = lambda R: (
        b ** (2 * n)
        / (2 * np.pi * n * gamma(2 * n) * re**2)
        * np.exp(-b * (R / re) ** (1 / n))
    )
    numerator = quad(lambda R: 2 * np.pi * R * los(R), 0, aperture, limit=100)[0]
    light = quad(lambda R: 2 * np.pi * R * surface(R), 0, aperture, limit=100)[0]
    expected = np.sqrt(numerator / light)
    np.testing.assert_allclose(
        aperture_dispersion(profile, aperture, beta), expected, rtol=2e-4
    )


class _PointMass(GalaxyProfile):
    """Sersic light orbiting a fixed point mass equal to the stellar mass."""

    def enclosed_mass_3d_msun(self, radius_kpc):
        return np.full_like(np.asarray(radius_kpc, float), self.stellar_mass_msun)


def test_gaussian_light_around_point_mass():
    """n=0.5 light is a 3D Gaussian: <sigma_los^2> = (2GM/3) sqrt(ln 2 / pi) / Re."""
    light = _PointMass(
        z_l=0.3,
        re_arcsec=1.0,
        stellar_mass_msun=1e11,
        halo_mass_msun=1e12,
        concentration=7.0,
        light_n=0.5,
    )
    re = light.re_kpc
    expected = np.sqrt(2 * _G * 1e11 / 3 * np.sqrt(np.log(2) / np.pi) / re)
    sigma = aperture_dispersion(light, 290 * re, [0.0, 0.5])
    np.testing.assert_allclose(sigma, expected, rtol=1e-6)


def test_negative_enclosed_mass_returns_nan(galaxy):
    re = galaxy.re_kpc
    profile = CompositeProfile(galaxy, (FiniteCore.from_total_mass(2 * re, -1.5e12),))
    assert np.isnan(aperture_dispersion(profile, re, 0.0))


def test_radial_orbits_raise_central_dispersion(galaxy):
    sigma = aperture_dispersion(galaxy, 0.5 * galaxy.re_kpc, [-0.3, 0.0, 0.3])
    assert np.all(np.diff(sigma) > 0)


def test_rejects_invalid_anisotropy(galaxy):
    with pytest.raises(ValueError, match="anisotropy"):
        aperture_dispersion(galaxy, galaxy.re_kpc, 1.0)
