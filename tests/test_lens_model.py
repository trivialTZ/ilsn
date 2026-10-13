"""Analytic and physical identities for the public spherical mass model."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf

from ilsn.lens_model import CompositeProfile, FiniteCore, GalaxyProfile, RadialLens


@pytest.fixture
def galaxy():
    return GalaxyProfile(
        z_l=0.3,
        re_arcsec=1.0,
        stellar_mass_msun=1e11,
        halo_mass_msun=1e13,
        concentration=7.0,
    )


def _nfw_projection(x):
    """Analytic NFW Sigma/(2 rho_s rs), M2d/(4 pi rho_s rs^3)."""
    x = np.asarray(x)
    factor = np.empty_like(x)
    lower, upper, equal = x < 1, x > 1, x == 1
    factor[lower] = (
        2
        * np.arctanh(np.sqrt((1 - x[lower]) / (1 + x[lower])))
        / np.sqrt(1 - x[lower] ** 2)
    )
    factor[upper] = (
        2
        * np.arctan(np.sqrt((x[upper] - 1) / (1 + x[upper])))
        / np.sqrt(x[upper] ** 2 - 1)
    )
    factor[equal] = 1
    sigma = np.empty_like(x)
    sigma[~equal] = (1 - factor[~equal]) / (x[~equal] ** 2 - 1)
    sigma[equal] = 1 / 3
    return sigma, np.log(x / 2) + factor


def test_nfw_and_gaussian_stars_match_analytic_projection():
    profile = GalaxyProfile(0.3, 1.0, 1e11, 1e13, 7.0, light_n=0.5)
    x = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0])
    r = x * profile.halo_rs_kpc
    shape, halo_mass = _nfw_projection(x)
    b = np.log(2)
    star_sigma = (
        profile.stellar_mass_msun
        * b
        / (np.pi * profile.re_kpc**2)
        * np.exp(-b * (r / profile.re_kpc) ** 2)
    )
    star_mass = profile.stellar_mass_msun * -np.expm1(-b * (r / profile.re_kpc) ** 2)
    analytic_sigma = 2 * profile.halo_rho_s * profile.halo_rs_kpc * shape + star_sigma
    analytic_mass = (
        4 * np.pi * profile.halo_rho_s * profile.halo_rs_kpc**3 * halo_mass + star_mass
    )
    np.testing.assert_allclose(
        profile.surface_density_msun_kpc2(r), analytic_sigma, rtol=3e-6
    )
    np.testing.assert_allclose(profile.projected_mass_msun(r), analytic_mass, rtol=3e-6)


def test_gaussian_stellar_deprojection_and_nfw_m200():
    profile = GalaxyProfile(0.3, 1.0, 1e11, 1e13, 7.0, light_n=0.5)
    r = profile.re_kpc * np.array([0.1, 0.5, 1, 3, 10])
    b = np.log(2)
    t = np.sqrt(b) * r / profile.re_kpc
    star_rho = (
        profile.stellar_mass_msun
        * b**1.5
        / (np.pi**1.5 * profile.re_kpc**3)
        * np.exp(-t * t)
    )
    star_mass = profile.stellar_mass_msun * (
        erf(t) - 2 / np.sqrt(np.pi) * t * np.exp(-t * t)
    )
    x = r / profile.halo_rs_kpc
    halo_rho = profile.halo_rho_s / (x * (1 + x) ** 2)
    halo_mass = (
        4
        * np.pi
        * profile.halo_rho_s
        * profile.halo_rs_kpc**3
        * (np.log1p(x) - x / (1 + x))
    )
    np.testing.assert_allclose(
        profile.density_3d_msun_kpc3(r), star_rho + halo_rho, rtol=2e-12
    )
    np.testing.assert_allclose(
        profile.enclosed_mass_3d_msun(r), star_mass + halo_mass, rtol=3e-6
    )
    # At r200 virtually all these compact stars are enclosed.
    np.testing.assert_allclose(
        profile.enclosed_mass_3d_msun(profile.halo_r200_kpc),
        profile.halo_mass_msun + profile.stellar_mass_msun,
        rtol=3e-6,
    )


def test_surface_density_and_3d_density_derivatives(galaxy):
    profile = GalaxyProfile(
        0.3,
        1.0,
        1e11,
        1e13,
        7.0,
        stellar_ml_gradient=-0.3,
        halo_inner_slope=1.3,
        light_components=((0.5, 2.0, 0.3), (1.5, 4.0, 0.7)),
    )
    r, epsilon = np.geomspace(0.1, 100, 15), 1e-5
    sigma_derivative = (
        profile.surface_density_msun_kpc2(r * np.exp(epsilon))
        - profile.surface_density_msun_kpc2(r * np.exp(-epsilon))
    ) / (2 * epsilon)
    rho_derivative = (
        profile.density_3d_msun_kpc3(r * np.exp(epsilon))
        - profile.density_3d_msun_kpc3(r * np.exp(-epsilon))
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        profile.surface_density_log_derivative(r), sigma_derivative, rtol=2e-7
    )
    np.testing.assert_allclose(
        profile.density_3d_log_derivative(r), rho_derivative, rtol=2e-7
    )
    # A global gradient shifts component mass fractions while preserving Mstar.
    assert sum(c[2] for c in profile.stellar_mass_components) == pytest.approx(1)
    assert profile.stellar_mass_components[0][2] > 0.3


def test_finite_core_total_mass_and_abel_projection():
    core = FiniteCore(5.0, 2e8)
    np.testing.assert_allclose(
        core.projected_mass_msun(1e8), core.total_mass_msun, rtol=1e-7
    )
    np.testing.assert_allclose(
        core.enclosed_mass_3d_msun(1e8), core.total_mass_msun, rtol=1e-7
    )
    r = 3.0
    projected = (
        2
        * quad(lambda z: float(core.density_3d_msun_kpc3(np.hypot(r, z))), 0, np.inf)[0]
    )
    assert projected == pytest.approx(
        float(core.surface_density_msun_kpc2(r)), rel=2e-10
    )
    mass = (
        4 * np.pi * quad(lambda x: x * x * float(core.density_3d_msun_kpc3(x)), 0, r)[0]
    )
    assert mass == pytest.approx(float(core.enclosed_mass_3d_msun(r)), rel=2e-10)
    # The small-radius series resolves enclosed mass without cancellation.
    tiny_r = core.radius_kpc * 1e-7
    uniform_center_mass = (
        4 * np.pi / 3 * tiny_r**3 * float(core.density_3d_msun_kpc3(tiny_r))
    )
    assert core.enclosed_mass_3d_msun(tiny_r) == pytest.approx(
        uniform_center_mass, rel=1e-12
    )


def test_compensation_and_broad_core_mass_are_source_independent(galaxy):
    pair = FiniteCore.redistribution(2.0, 6.0, 3e10)
    assert sum(c.total_mass_msun for c in pair) == pytest.approx(0, abs=1e-5)
    masses = []
    for q in (3, 5, 10):
        profile = CompositeProfile.from_reference(
            galaxy, 1e11, 4.0, eta=-0.02, q=q, compensation=0.1
        )
        assert sum(c.total_mass_msun for c in profile.cores[1:]) == pytest.approx(
            0, abs=1e-5
        )
        masses.append(profile.total_core_mass_msun)
        assert profile.cores[0].radius_kpc == q * 4
        assert (
            profile.lens(0.8).profile.total_core_mass_msun
            == profile.lens(2.0).profile.total_core_mass_msun
        )
    np.testing.assert_allclose(masses, -1e11, rtol=1e-15)


class _SIS(RadialLens):
    """Analytic SIS used only to check the circular lens identities."""

    def kappa(self, theta):
        return 0.5 / np.asarray(theta)

    def mean_kappa(self, theta):
        return 1 / np.asarray(theta)

    def kappa_log_derivative(self, theta):
        return -self.kappa(theta)

    def einstein_radius_arcsec(self):
        return 1.0


def test_sis_macro_magnification_and_curvature():
    lens = _SIS()
    theta = np.array([0.4, 0.9, 1.1, 2.0, 5.0])
    np.testing.assert_allclose(lens.mu(theta), 1 / (1 - 1 / theta))
    np.testing.assert_allclose(lens.deflection(theta), 1)
    assert lens.s_E() == 0.5
    assert lens.xi_E() == 0
    with pytest.raises(ValueError, match="critical"):
        lens.mu(1.0)


def test_exact_mst_endpoint_and_magnification_identities(galaxy):
    base = galaxy.lens(1.0)
    transformed = base.mass_sheet_transform(0.8)
    theta = base.theta_E() * np.array([0.4, 0.9, 1.1, 2, 5])
    assert transformed.theta_E() == base.theta_E()
    assert transformed.s_E() == pytest.approx(0.8 * base.s_E(), rel=1e-12)
    assert transformed.xi_E() == pytest.approx(base.xi_E(), abs=1e-12)
    np.testing.assert_allclose(
        transformed.mu(theta), base.mu(theta) / 0.8**2, rtol=2e-13
    )
    np.testing.assert_allclose(
        transformed.radial_eigenvalue(theta),
        0.8 * base.radial_eigenvalue(theta),
        rtol=2e-13,
    )
    assert base.mean_kappa(base.theta_E()) == pytest.approx(1, rel=1e-11)
    np.testing.assert_allclose(base.signed_magnification(theta), base.mu(theta))


def test_sampled_physical_support_applies_to_summed_signed_profile(galaxy):
    radius = np.geomspace(0.01, 1000, 100)
    positive = CompositeProfile(galaxy, (FiniteCore(10, 1e7),))
    diagnostic = positive.physical_support(radius)
    assert diagnostic["sampled_valid"]
    assert diagnostic["certified"] is False
    assert diagnostic["radius_min_kpc"] == radius.min()
    assert diagnostic["radius_max_kpc"] == radius.max()
    negative = CompositeProfile(galaxy, (FiniteCore(10, -1e12),))
    assert not negative.physical_support(radius)["sampled_valid"]
    # Opposite equal kernels cancel before checking physical support.
    cancelling = CompositeProfile(galaxy, (FiniteCore(10, -1e12), FiniteCore(10, 1e12)))
    np.testing.assert_allclose(
        cancelling.surface_density_msun_kpc2(radius),
        galaxy.surface_density_msun_kpc2(radius),
        rtol=3e-11,
    )
    assert cancelling.physical_support(radius)["sampled_valid"]


def test_invalid_physical_inputs_and_projection_domain(galaxy):
    with pytest.raises(ValueError, match="gradient"):
        GalaxyProfile(0.3, 1, 1e11, 1e13, 7, stellar_ml_gradient=0.2)
    with pytest.raises(ValueError, match="source|z_s"):
        galaxy.lens(0.2)
    with pytest.raises(ValueError, match="domain"):
        galaxy.surface_density_msun_kpc2(galaxy.halo_rs_kpc * 1e-8)
    with pytest.raises(ValueError, match="positive"):
        FiniteCore(0, 1)
    with pytest.raises(ValueError, match="ordered"):
        CompositeProfile.from_reference(
            galaxy, 1e11, 4, 0.1, 5, compensation_radii=(3, 1)
        )
