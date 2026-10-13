"""Spherical stellar, halo, and finite-core mass models.

Masses use physical solar masses, radii kpc, and angles arcsec. The halo is
an untruncated generalized NFW profile: ``halo_mass_msun`` means M200 within
200 times the critical density at the lens redshift, not its total mass.
Stellar mass follows circular Sersic light multiplied by a projected M/L
gradient. Signed finite cores are additive; they do not re-budget M200.
"""

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Any

import numpy as np
from astropy import constants as const, units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import cumulative_simpson, quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import gamma, gammainc, gammaincinv, hyp2f1

DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)
_ARCSEC_RAD = u.arcsec.to(u.rad)
_C2_OVER_4PI_G = (const.c**2 / (4 * np.pi * const.G)).to_value(u.Msun / u.kpc)
_X_MIN, _X_MAX = 1e-7, 1e4


def _positive(value, name):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return value


@lru_cache(maxsize=32)
def _halo_tables(slope):
    """Tabulate dimensionless gNFW surface density and cylinder mass.

    The line-of-sight substitution z=R sinh(t) uses 256 Gauss nodes. The
    inner cylinder mass is integrated independently so that the projected
    mass has no missing central integration constant. Log interpolation
    uses 4097 radii, with no extrapolation outside [1e-7, 1e4] rs.
    """
    logx = np.linspace(np.log(_X_MIN), np.log(_X_MAX), 4097)
    x = np.exp(logx)
    nodes, weights = np.polynomial.legendre.leggauss(256)
    r = x[:, None] * np.cosh((nodes + 1) * 16)
    sigma = 2 * ((r ** (1 - slope) * (1 + r) ** (slope - 3)) @ (weights * 16))
    xmin = x[0]
    integral = quad(
        lambda v: (np.exp(v) + xmin) ** (slope - 3)
        * np.exp(2 * v)
        / (1 + np.sqrt(1 - np.exp(2 * v))),
        np.log(xmin) - 30,
        0,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=300,
    )[0]
    mass0 = xmin ** (3 - slope) * (
        hyp2f1(3 - slope, 3 - slope, 4 - slope, -xmin) / (3 - slope) + integral
    )
    mass = mass0 + cumulative_simpson(0.5 * x * x * sigma, x=logx, initial=0)
    return (
        PchipInterpolator(logx, np.log(sigma), extrapolate=False),
        PchipInterpolator(logx, np.log(mass), extrapolate=False),
    )


@lru_cache(maxsize=1)
def _abel_quadrature():
    nodes, weights = np.polynomial.legendre.leggauss(256)
    return np.cosh((nodes + 1) * 10), weights * 10


@lru_cache(maxsize=32)
def _stellar_mass_table(n, gradient):
    """Abel-deprojected Sersic mass on [1e-7, 1e4] Re.

    Mass below the lower boundary is omitted (about 1e-6 of the total in
    the steepest allowed case). No density clipping or renormalization is
    applied to this numerical deprojection.
    """
    logx = np.linspace(np.log(_X_MIN), np.log(_X_MAX), 4097)
    x = np.exp(logx)
    cosh_t, weights = _abel_quadrature()
    r = x[:, None] * cosh_t
    b, a = gammaincinv(2 * n, 0.5), (2 + gradient) * n
    t = b * r ** (1 / n)
    sigma = b**a / (2 * np.pi * n * gamma(a)) * r**gradient * np.exp(-t)
    density = (sigma * (t / n - gradient) / r) @ weights / np.pi
    mass = cumulative_simpson(4 * np.pi * x**3 * density, x=logx, initial=0)
    return PchipInterpolator(logx, mass, extrapolate=False)


@dataclass(frozen=True)
class GalaxyProfile:
    """Independently normalized spherical stars and gNFW halo.

    :param z_l: Lens redshift.
    :param re_arcsec: Circular half-light radius in arcsec.
    :param stellar_mass_msun: Total stellar mass in physical solar masses.
    :param halo_mass_msun: Dark M200 in physical solar masses.
    :param concentration: r200/rs.
    :param light_n: Sersic index, between 0.5 and 8.
    :param halo_inner_slope: gNFW inner density slope, between 0.3 and 1.8.
    :param stellar_ml_gradient: Projected dln(M/L)/dlnR, between -0.5 and 0.
    :param light_components: Optional tuples (Re arcsec, n, light fraction).
        Fractions sum to one. The M/L gradient applies globally to their
        sum; component stellar masses are normalized accordingly.
    :param cosmo: Astropy cosmology used for physical sizes and M200.
    """

    z_l: float
    re_arcsec: float
    stellar_mass_msun: float
    halo_mass_msun: float
    concentration: float
    light_n: float = 4.0
    halo_inner_slope: float = 1.0
    stellar_ml_gradient: float = 0.0
    light_components: tuple[tuple[float, float, float], ...] | None = None
    cosmo: Any = field(default_factory=lambda: DEFAULT_COSMOLOGY)

    def __post_init__(self):
        for name in (
            "z_l",
            "re_arcsec",
            "stellar_mass_msun",
            "halo_mass_msun",
            "concentration",
        ):
            _positive(getattr(self, name), name)
        if not np.isfinite(self.light_n) or not 0.5 <= self.light_n <= 8:
            raise ValueError("light_n must be in [0.5, 8]")
        if (
            not np.isfinite(self.halo_inner_slope)
            or not 0.3 <= self.halo_inner_slope <= 1.8
        ):
            raise ValueError("halo_inner_slope must be in [0.3, 1.8]")
        if (
            not np.isfinite(self.stellar_ml_gradient)
            or not -0.5 <= self.stellar_ml_gradient <= 0
        ):
            raise ValueError("stellar_ml_gradient must be in [-0.5, 0]")
        if self.light_components is not None:
            components = tuple(
                tuple(float(v) for v in c) for c in self.light_components
            )
            if not components or any(len(c) != 3 for c in components):
                raise ValueError(
                    "light_components must contain (Re, n, fraction) triples"
                )
            for re, n, fraction in components:
                _positive([re, fraction], "component radius and light fraction")
                if not np.isfinite(n) or not 0.5 <= n <= 8:
                    raise ValueError("component Sersic n must be in [0.5, 8]")
            if not np.isclose(sum(c[2] for c in components), 1, rtol=0, atol=1e-6):
                raise ValueError("component light fractions must sum to one")
            object.__setattr__(self, "light_components", components)

    @cached_property
    def kpc_per_arcsec(self):
        """Physical angular scale in kpc/arcsec."""
        return float(
            self.cosmo.angular_diameter_distance(self.z_l).to_value(u.kpc) * _ARCSEC_RAD
        )

    @cached_property
    def re_kpc(self):
        """Circular half-light radius in kpc."""
        return self.re_arcsec * self.kpc_per_arcsec

    @cached_property
    def halo_r200_kpc(self):
        rhoc = self.cosmo.critical_density(self.z_l).to_value(u.Msun / u.kpc**3)
        return float((3 * self.halo_mass_msun / (800 * np.pi * rhoc)) ** (1 / 3))

    @cached_property
    def halo_rs_kpc(self):
        return self.halo_r200_kpc / self.concentration

    @cached_property
    def halo_rho_s(self):
        """gNFW density normalization in Msun/kpc^3."""
        g, c = self.halo_inner_slope, self.concentration
        mass = c ** (3 - g) * hyp2f1(3 - g, 3 - g, 4 - g, -c) / (3 - g)
        return float(self.halo_mass_msun / (4 * np.pi * self.halo_rs_kpc**3 * mass))

    @cached_property
    def stellar_mass_components(self):
        """Tuples (Re kpc, n, stellar mass fraction) after the M/L gradient."""
        components = []
        g = self.stellar_ml_gradient
        for re, n, fraction in self.light_components or (
            (self.re_arcsec, self.light_n, 1.0),
        ):
            re = re * self.kpc_per_arcsec
            b = gammaincinv(2 * n, 0.5)
            weight = (
                fraction * re**g * b ** (-n * g) * gamma((2 + g) * n) / gamma(2 * n)
            )
            components.append((re, n, float(weight)))
        total = sum(c[2] for c in components)
        return tuple((re, n, weight / total) for re, n, weight in components)

    def _stars(self, radius_kpc):
        r = _positive(radius_kpc, "radius_kpc")
        sigma, mass, derivative = (np.zeros_like(r) for _ in range(3))
        g = self.stellar_ml_gradient
        for re, n, fraction in self.stellar_mass_components:
            b, a = gammaincinv(2 * n, 0.5), (2 + g) * n
            x, t = r / re, b * (r / re) ** (1 / n)
            s = (
                self.stellar_mass_msun
                * fraction
                * b**a
                / (2 * np.pi * n * re**2 * gamma(a))
            )
            s = s * x**g * np.exp(-t)
            sigma += s
            mass += self.stellar_mass_msun * fraction * gammainc(a, t)
            derivative += s * (g - t / n)
        return sigma, mass, derivative

    def _halo(self, radius_kpc):
        r = _positive(radius_kpc, "radius_kpc")
        x = r / self.halo_rs_kpc
        if np.any(x < _X_MIN) or np.any(x > _X_MAX):
            raise ValueError("radius outside halo projection domain [1e-7, 1e4] rs")
        sf, mf = _halo_tables(float(self.halo_inner_slope))
        sigma = self.halo_rho_s * self.halo_rs_kpc * np.exp(sf(np.log(x)))
        mass = 4 * np.pi * self.halo_rho_s * self.halo_rs_kpc**3 * np.exp(mf(np.log(x)))
        return sigma, mass, sigma * sf.derivative()(np.log(x))

    def _stellar_density(self, radius_kpc):
        r = _positive(radius_kpc, "radius_kpc")
        cosh_t, weights = _abel_quadrature()
        rr = r[..., None] * cosh_t
        density, derivative = (np.zeros_like(r) for _ in range(2))
        g = self.stellar_ml_gradient
        for re, n, fraction in self.stellar_mass_components:
            b, a = gammaincinv(2 * n, 0.5), (2 + g) * n
            t = b * (rr / re) ** (1 / n)
            sigma = (
                self.stellar_mass_msun
                * fraction
                * b**a
                / (2 * np.pi * n * re**2 * gamma(a))
            )
            sigma = sigma * (rr / re) ** g * np.exp(-t)
            factor = t / n - g
            density += (sigma * factor / rr) @ weights / np.pi
            derivative += (
                (sigma / rr * (-factor * (factor + 1) + t / n**2)) @ weights / np.pi
            )
        return density, derivative

    def surface_density_msun_kpc2(self, radius_kpc):
        """Projected surface density.

        :param radius_kpc: Positive radial coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^2
        """
        return self._stars(radius_kpc)[0] + self._halo(radius_kpc)[0]

    def surface_density_log_derivative(self, radius_kpc):
        """Return dSigma/dlnR.

        :param radius_kpc: Positive radial coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^2
        """
        return self._stars(radius_kpc)[2] + self._halo(radius_kpc)[2]

    def projected_mass_msun(self, radius_kpc):
        """Return mass within a projected cylinder.

        :param radius_kpc: Positive radial coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun
        """
        return self._stars(radius_kpc)[1] + self._halo(radius_kpc)[1]

    def density_3d_msun_kpc3(self, radius_kpc):
        """Return Abel-deprojected stars plus analytic halo density.

        :param radius_kpc: Positive spherical coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^3
        """
        x = _positive(radius_kpc, "radius_kpc") / self.halo_rs_kpc
        return self._stellar_density(radius_kpc)[0] + self.halo_rho_s * x ** (
            -self.halo_inner_slope
        ) * (1 + x) ** (self.halo_inner_slope - 3)

    def density_3d_log_derivative(self, radius_kpc):
        """Return d(rho)/dlnr.

        :param radius_kpc: Positive spherical coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^3
        """
        x = _positive(radius_kpc, "radius_kpc") / self.halo_rs_kpc
        halo = (
            self.halo_rho_s
            * x ** (-self.halo_inner_slope)
            * (1 + x) ** (self.halo_inner_slope - 3)
        )
        return self._stellar_density(radius_kpc)[1] - halo * (
            self.halo_inner_slope + 3 * x
        ) / (1 + x)

    def enclosed_mass_3d_msun(self, radius_kpc):
        """Return actual mass within a sphere, in Msun.

        Stellar mass is numerically Abel-deprojected on [1e-7, 1e4] times
        each component's Re. The small inner omission is documented above.

        :param radius_kpc: Positive spherical coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun
        """
        r = _positive(radius_kpc, "radius_kpc")
        g, x = self.halo_inner_slope, r / self.halo_rs_kpc
        mass = (
            4
            * np.pi
            * self.halo_rho_s
            * self.halo_rs_kpc**3
            * x ** (3 - g)
            * hyp2f1(3 - g, 3 - g, 4 - g, -x)
            / (3 - g)
        )
        for re, n, fraction in self.stellar_mass_components:
            xr = r / re
            if np.any(xr < _X_MIN) or np.any(xr > _X_MAX):
                raise ValueError("radius outside stellar 3D mass domain [1e-7, 1e4] Re")
            mass += (
                self.stellar_mass_msun
                * fraction
                * _stellar_mass_table(n, self.stellar_ml_gradient)(np.log(xr))
            )
        return mass

    def lens(self, z_s):
        """Return a spherical lens for source redshift z_s.

        :param z_s: Source redshift above z_l.
        :rtype: SphericalLens
        """
        return SphericalLens(self, z_s)

    def physical_support(self, radius_kpc):
        """Return sampled radial support diagnostics; no global certificate.

        :param radius_kpc: Nonempty one-dimensional positive radius grid.
        :rtype: dict
        """
        return _physical_support(self, radius_kpc)


@dataclass(frozen=True)
class FiniteCore:
    """Signed finite-mass kernel Sigma(R)=Sigma0 [1+(R/rc)^2]^-3/2.

    Its total mass is 2 pi Sigma0 rc^2. A signed component need not be
    physical alone; support applies to the sum with stars and halo.

    :param radius_kpc: Positive core scale rc in kpc.
    :param central_surface_density_msun_kpc2: Signed Sigma0 in Msun/kpc^2.
    """

    radius_kpc: float
    central_surface_density_msun_kpc2: float

    def __post_init__(self):
        _positive(self.radius_kpc, "radius_kpc")
        if not np.isfinite(self.central_surface_density_msun_kpc2):
            raise ValueError("central surface density must be finite")

    @property
    def total_mass_msun(self):
        """Return the signed total mass, in physical solar masses."""
        return 2 * np.pi * self.central_surface_density_msun_kpc2 * self.radius_kpc**2

    @classmethod
    def from_total_mass(cls, radius_kpc, mass_msun):
        """Construct a core from its signed finite total mass.

        :param radius_kpc: Positive scale radius in kpc.
        :param mass_msun: Signed total mass in physical solar masses.
        :rtype: FiniteCore
        """
        _positive(radius_kpc, "radius_kpc")
        return cls(radius_kpc, mass_msun / (2 * np.pi * radius_kpc**2))

    @classmethod
    def redistribution(cls, inner_radius_kpc, outer_radius_kpc, mass_msun):
        """Return two signed cores with exactly cancelling total masses.

        :param inner_radius_kpc: Compact core radius in kpc.
        :param outer_radius_kpc: Larger core radius in kpc.
        :param mass_msun: Signed mass assigned to the compact component.
        :rtype: tuple[FiniteCore, FiniteCore]
        """
        _positive([inner_radius_kpc, outer_radius_kpc], "core radii")
        if outer_radius_kpc <= inner_radius_kpc:
            raise ValueError("outer radius must exceed inner radius")
        return (
            cls.from_total_mass(inner_radius_kpc, mass_msun),
            cls.from_total_mass(outer_radius_kpc, -mass_msun),
        )

    def _fields(self, radius_kpc):
        t = _positive(radius_kpc, "radius_kpc") / self.radius_kpc
        x2 = t * t
        sigma = self.central_surface_density_msun_kpc2 * (1 + x2) ** -1.5
        rho = (
            self.central_surface_density_msun_kpc2
            * 2
            / (np.pi * self.radius_kpc)
            * (1 + x2) ** -2
        )
        mass_factor = np.where(
            t < 0.01,
            t**3 * (2 / 3 + x2 * (-4 / 5 + x2 * (6 / 7 - x2 * 8 / 9))),
            np.arctan(t) - t / (1 + x2),
        )
        return (
            sigma,
            -3 * x2 / (1 + x2) * sigma,
            rho,
            -4 * x2 / (1 + x2) * rho,
            mass_factor,
        )

    def surface_density_msun_kpc2(self, radius_kpc):
        """Return signed Sigma at positive radius(s) in kpc.

        :param radius_kpc: Positive radial coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^2
        """
        return self._fields(radius_kpc)[0]

    def surface_density_log_derivative(self, radius_kpc):
        """Return signed dSigma/dlnR at positive radius(s) in kpc.

        :param radius_kpc: Positive radial coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^2
        """
        return self._fields(radius_kpc)[1]

    def projected_mass_msun(self, radius_kpc):
        """Return signed cylinder mass.

        :param radius_kpc: Positive radial coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun
        """
        r = _positive(radius_kpc, "radius_kpc")
        root = np.sqrt(1 + (r / self.radius_kpc) ** 2)
        # This avoids subtracting nearly equal numbers at small r/rc.
        return (
            np.pi
            * r
            * r
            * self.central_surface_density_msun_kpc2
            * 2
            / (root * (1 + root))
        )

    def density_3d_msun_kpc3(self, radius_kpc):
        """Return signed Abel-deprojected density.

        :param radius_kpc: Positive spherical coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^3
        """
        return self._fields(radius_kpc)[2]

    def density_3d_log_derivative(self, radius_kpc):
        """Return signed d(rho)/dlnr.

        :param radius_kpc: Positive spherical coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^3
        """
        return self._fields(radius_kpc)[3]

    def enclosed_mass_3d_msun(self, radius_kpc):
        """Return signed mass within a sphere.

        :param radius_kpc: Positive spherical coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun
        """
        return (
            4
            * self.central_surface_density_msun_kpc2
            * self.radius_kpc**2
            * self._fields(radius_kpc)[4]
        )


@dataclass(frozen=True)
class CompositeProfile:
    """A galaxy plus signed finite cores, summed before physical checks.

    :param base: Stellar and halo GalaxyProfile.
    :param cores: Sequence of FiniteCore components.
    """

    base: GalaxyProfile
    cores: tuple[FiniteCore, ...] = ()

    def __post_init__(self):
        if not isinstance(self.base, GalaxyProfile) or any(
            not isinstance(c, FiniteCore) for c in self.cores
        ):
            raise TypeError(
                "base must be GalaxyProfile and cores must be FiniteCore objects"
            )
        object.__setattr__(self, "cores", tuple(self.cores))

    @classmethod
    def from_reference(
        cls,
        base,
        reference_mass_msun,
        reference_re_kpc,
        eta,
        q,
        compensation=0.0,
        *,
        reference_q=5.0,
        compensation_radii=(1.0, 3.0),
    ):
        """Construct the broad core and zero-total-mass redistribution.

        rc=q Re_ref and Sigma0=eta (q_ref/q)^2 M_ref/(pi Re_ref^2).
        Hence the broad core mass is 2 q_ref^2 eta M_ref at every q. The
        compact pair has amplitudes compensation and -compensation (a/b)^2
        in the same surface-density unit, and total signed mass zero.
        These reference quantities are fixed physical covariates and do
        not depend on source redshift or critical surface density.

        :param base: GalaxyProfile with independent stellar/halo masses.
        :param reference_mass_msun: Positive reference mass in Msun.
        :param reference_re_kpc: Positive reference radius in kpc.
        :param eta: Signed broad-core amplitude at reference_q.
        :param q: Broad core radius divided by the reference radius.
        :param compensation: Signed compact redistribution amplitude.
        :param reference_q: Reference broad-core radius ratio.
        :param compensation_radii: Ordered compact-pair radius ratios (a,b).
        :rtype: CompositeProfile
        """
        _positive(
            [reference_mass_msun, reference_re_kpc, q, reference_q], "reference scales"
        )
        if not np.all(np.isfinite([eta, compensation])):
            raise ValueError("core amplitudes must be finite")
        pair = np.asarray(compensation_radii, dtype=float)
        if (
            pair.shape != (2,)
            or not np.all(np.isfinite(pair))
            or not 0 < pair[0] < pair[1]
        ):
            raise ValueError("compensation_radii must be positive and ordered")
        unit = reference_mass_msun / (np.pi * reference_re_kpc**2)
        broad = FiniteCore(q * reference_re_kpc, eta * (reference_q / q) ** 2 * unit)
        pair_mass = 2 * compensation * pair[0] ** 2 * reference_mass_msun
        compact = FiniteCore.redistribution(*(pair * reference_re_kpc), pair_mass)
        return cls(base, (broad, *compact))

    @property
    def z_l(self):
        return self.base.z_l

    @property
    def cosmo(self):
        return self.base.cosmo

    @property
    def kpc_per_arcsec(self):
        return self.base.kpc_per_arcsec

    @property
    def halo_rs_kpc(self):
        return self.base.halo_rs_kpc

    @property
    def halo_r200_kpc(self):
        return self.base.halo_r200_kpc

    @property
    def total_core_mass_msun(self):
        """Signed finite core mass; the untruncated halo has no total mass."""
        return sum(c.total_mass_msun for c in self.cores)

    def _sum(self, method, radius_kpc):
        core_sum = np.zeros_like(_positive(radius_kpc, "radius_kpc"))
        for core in self.cores:
            core_sum = core_sum + getattr(core, method)(radius_kpc)
        return getattr(self.base, method)(radius_kpc) + core_sum

    def surface_density_msun_kpc2(self, radius_kpc):
        """Return total projected density.

        :param radius_kpc: Positive radial coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^2
        """
        return self._sum("surface_density_msun_kpc2", radius_kpc)

    def surface_density_log_derivative(self, radius_kpc):
        """Return total dSigma/dlnR.

        :param radius_kpc: Positive radial coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^2
        """
        return self._sum("surface_density_log_derivative", radius_kpc)

    def projected_mass_msun(self, radius_kpc):
        """Return total mass within a projected cylinder.

        :param radius_kpc: Positive radial coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun
        """
        return self._sum("projected_mass_msun", radius_kpc)

    def density_3d_msun_kpc3(self, radius_kpc):
        """Return total deprojected density.

        :param radius_kpc: Positive spherical coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^3
        """
        return self._sum("density_3d_msun_kpc3", radius_kpc)

    def density_3d_log_derivative(self, radius_kpc):
        """Return total d(rho)/dlnr.

        :param radius_kpc: Positive spherical coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun/kpc^3
        """
        return self._sum("density_3d_log_derivative", radius_kpc)

    def enclosed_mass_3d_msun(self, radius_kpc):
        """Return total mass within a sphere.

        :param radius_kpc: Positive spherical coordinate(s) in kpc.
        :rtype: numpy.ndarray, in Msun
        """
        return self._sum("enclosed_mass_3d_msun", radius_kpc)

    def lens(self, z_s):
        """Return a spherical lens for the specified source redshift.

        :param z_s: Source redshift above the lens redshift.
        :rtype: SphericalLens
        """
        return SphericalLens(self, z_s)

    def physical_support(self, radius_kpc):
        """Return sampled support diagnostics for the total signed sum.

        :param radius_kpc: Nonempty one-dimensional positive radius grid.
        :rtype: dict
        """
        return _physical_support(self, radius_kpc)


def _physical_support(profile, radius_kpc):
    r = _positive(radius_kpc, "radius_kpc")
    if r.ndim != 1 or not r.size:
        raise ValueError("support radii must be a nonempty one-dimensional grid")
    rho = profile.density_3d_msun_kpc3(r)
    rho_deriv = profile.density_3d_log_derivative(r)
    sigma = profile.surface_density_msun_kpc2(r)
    sigma_deriv = profile.surface_density_log_derivative(r)
    mass = profile.enclosed_mass_3d_msun(r)
    projected_mass = profile.projected_mass_msun(r)
    valid = (
        np.all(rho > 0)
        and np.all(rho_deriv <= 0)
        and np.all(sigma > 0)
        and np.all(sigma_deriv <= 0)
        and np.all(mass >= 0)
        and np.all(projected_mass >= 0)
    )
    return dict(
        sampled_valid=bool(valid),
        certified=False,
        radius_min_kpc=float(r.min()),
        radius_max_kpc=float(r.max()),
        sample_count=int(r.size),
        min_density_3d_msun_kpc3=float(rho.min()),
        max_density_3d_log_derivative=float(rho_deriv.max()),
        min_surface_density_msun_kpc2=float(sigma.min()),
        max_surface_density_log_derivative=float(sigma_deriv.max()),
        min_enclosed_mass_msun=float(mass.min()),
        min_projected_mass_msun=float(projected_mass.min()),
        status="sampled radial positivity and decreasing density; no all-radius or distribution-function certificate",
    )


class RadialLens:
    """Shared circular lens identities for strictly positive radial angles."""

    def deflection(self, theta_arcsec):
        """Return radial deflection.

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray, in arcsec
        """
        return _positive(theta_arcsec, "theta_arcsec") * self.mean_kappa(theta_arcsec)

    def tangential_eigenvalue(self, theta_arcsec):
        """Return 1-mean_kappa at positive radial angle(s) in arcsec.

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        return 1 - self.mean_kappa(theta_arcsec)

    def radial_eigenvalue(self, theta_arcsec):
        """Return 1-2*kappa+mean_kappa at positive radial angle(s).

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        return 1 - 2 * self.kappa(theta_arcsec) + self.mean_kappa(theta_arcsec)

    def mu(self, theta_arcsec):
        """Return signed macro magnification; critical curves are excluded.

        :param theta_arcsec: Positive image radius(s) in arcsec.
        :rtype: numpy.ndarray
        """
        determinant = self.tangential_eigenvalue(theta_arcsec) * self.radial_eigenvalue(
            theta_arcsec
        )
        if np.any(determinant == 0) or not np.all(np.isfinite(determinant)):
            raise ValueError("magnification undefined at critical curve")
        return 1 / determinant

    def signed_magnification(self, theta_arcsec):
        """Alias for mu, including image parity in its sign.

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        return self.mu(theta_arcsec)

    def theta_E(self):
        """Return the outermost circular Einstein radius.

        :rtype: float, in arcsec
        """
        return self.einstein_radius_arcsec()

    def s_E(self):
        """Return the normalization endpoint 1-kappa(theta_E).

        :rtype: float
        """
        return float(1 - self.kappa(self.theta_E()))

    def xi_invariant(self, theta_arcsec):
        """Return theta alpha''/(1-alpha'), an exact MST invariant.

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        theta = _positive(theta_arcsec, "theta_arcsec")
        denominator = self.radial_eigenvalue(theta)
        if np.any(np.abs(denominator) < 1e-10):
            raise ValueError("xi undefined at radial critical curve")
        return (
            2
            * (
                self.kappa_log_derivative(theta)
                - self.kappa(theta)
                + self.mean_kappa(theta)
            )
            / denominator
        )

    def xi_E(self):
        """Return the imaging curvature xi at the derived Einstein ring.

        Uses denominator 1-alpha'=2(1-kappa_E) at the ring. A convention
        using 1-kappa_E instead would give twice this value.

        :rtype: float
        """
        return float(self.xi_invariant(self.theta_E()))

    def mass_sheet_transform(self, lambda_mst):
        """Return an exact mass-sheet transformation for identity checks.

        :param lambda_mst: Positive MST amplitude.
        :rtype: MSTLens
        """
        return MSTLens(self, lambda_mst)


class SphericalLens(RadialLens):
    """Project a physical profile for a single source redshift.

    :param profile: GalaxyProfile or CompositeProfile.
    :param z_s: Source redshift, strictly above the lens redshift.
    """

    def __init__(self, profile, z_s):
        if not np.isfinite(z_s) or not z_s > profile.z_l:
            raise ValueError("require finite z_s > z_l > 0")
        self.profile, self.z_s = profile, float(z_s)
        cosmo = profile.cosmo
        dl = cosmo.angular_diameter_distance(profile.z_l).to_value(u.kpc)
        ds = cosmo.angular_diameter_distance(z_s).to_value(u.kpc)
        dls = cosmo.angular_diameter_distance_z1z2(profile.z_l, z_s).to_value(u.kpc)
        self.sigma_crit_msun_kpc2 = float(_C2_OVER_4PI_G * ds / (dl * dls))
        _positive(self.sigma_crit_msun_kpc2, "critical surface density")
        self.minimum_radius_arcsec = (
            _X_MIN * profile.halo_rs_kpc / profile.kpc_per_arcsec
        )
        self.maximum_radius_arcsec = (
            _X_MAX * profile.halo_rs_kpc / profile.kpc_per_arcsec
        )

    def kappa(self, theta_arcsec):
        """Return convergence at positive image radius(s).

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        r = _positive(theta_arcsec, "theta_arcsec") * self.profile.kpc_per_arcsec
        return self.profile.surface_density_msun_kpc2(r) / self.sigma_crit_msun_kpc2

    def mean_kappa(self, theta_arcsec):
        """Return mean convergence within the projected cylinder.

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        r = _positive(theta_arcsec, "theta_arcsec") * self.profile.kpc_per_arcsec
        return self.profile.projected_mass_msun(r) / (
            np.pi * r * r * self.sigma_crit_msun_kpc2
        )

    def kappa_log_derivative(self, theta_arcsec):
        """Return d(kappa)/dln(theta).

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        r = _positive(theta_arcsec, "theta_arcsec") * self.profile.kpc_per_arcsec
        return (
            self.profile.surface_density_log_derivative(r) / self.sigma_crit_msun_kpc2
        )

    def einstein_radius_arcsec(self):
        """Find the outermost mean-kappa=1 crossing in the projection domain.

        A logarithmic search allows composite signed profiles to have
        multiple rings. The returned ring crosses from above to below one.
        This root is derived from the model, with no imposed imaging anchor.

        :rtype: float, in arcsec
        """
        grid = np.geomspace(
            self.minimum_radius_arcsec * (1 + 1e-6),
            self.maximum_radius_arcsec * (1 - 1e-6),
            1025,
        )
        values = self.mean_kappa(grid) - 1
        brackets = np.flatnonzero((values[:-1] >= 0) & (values[1:] < 0))
        if not brackets.size:
            raise ValueError("no Einstein radius in the halo projection domain")
        i = brackets[-1]
        return float(
            brentq(
                lambda t: float(self.mean_kappa(t) - 1),
                grid[i],
                grid[i + 1],
                xtol=1e-11,
            )
        )


class MSTLens(RadialLens):
    """Exact uniform-sheet transformation used for lensing identity checks.

    A uniform sheet is not an admitted finite physical galaxy component.

    :param base: Radial lens model.
    :param lambda_mst: Positive transformation amplitude.
    """

    def __init__(self, base, lambda_mst):
        _positive(lambda_mst, "lambda_mst")
        self.base, self.lambda_mst = base, float(lambda_mst)

    def kappa(self, theta_arcsec):
        """Return transformed convergence.

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        return self.lambda_mst * self.base.kappa(theta_arcsec) + 1 - self.lambda_mst

    def mean_kappa(self, theta_arcsec):
        """Return transformed mean convergence.

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        return (
            self.lambda_mst * self.base.mean_kappa(theta_arcsec) + 1 - self.lambda_mst
        )

    def kappa_log_derivative(self, theta_arcsec):
        """Return transformed d(kappa)/dln(theta).

        :param theta_arcsec: Positive radial angle(s) in arcsec.
        :rtype: numpy.ndarray
        """
        return self.lambda_mst * self.base.kappa_log_derivative(theta_arcsec)

    def einstein_radius_arcsec(self):
        """Return the Einstein radius preserved by the exact MST.

        :rtype: float, in arcsec
        """
        return self.base.einstein_radius_arcsec()
