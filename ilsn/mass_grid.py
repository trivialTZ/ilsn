"""Physical design covariates and weighted stellar/halo/finite-core states.

Reference masses and sizes are supplied covariates. They are not measured
stellar-population constraints and are never inferred from an Einstein ring.
Population tilts use eight coordinates frozen to the Roman primary measure;
changing quadrature resolution does not redefine those coordinates.
"""

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

from .lens_model import DEFAULT_COSMOLOGY


SCORE_NAMES = (
    "log_mass_multiplier",
    "halo_residual",
    "concentration_residual",
    "stellar_gradient",
    "halo_slope",
    "eta_core_mean",
    "eta_core_even_width",
    "gradient_core_interaction",
)
_PRIMARY_AXES = dict(
    mass_multipliers=(0.7, 0.9, 1.2, 1.6, 2.2),
    halo_nodes=(-1.5, -0.5, 0.5, 1.5),
    concentration_nodes=(-1.5, 0.0, 1.5),
    gradients=(-0.3, -0.15, 0.0),
    halo_slopes=(0.7, 1.0, 1.3),
    eta_nodes=tuple(np.linspace(-0.2, 0.2, 9)),
)


@dataclass(frozen=True)
class LensDesign:
    """Object-aligned fixed covariates for a physical lens bank.

    ``log_design_mass`` is log10 of physical reference mass in Msun;
    ``re_arcsec`` is the fixed reference half-light radius. Scalar inputs
    broadcast to the number of objects. Source banks also require positive
    ``radius_arcsec``, the observed SN image radius.
    """

    log_design_mass: np.ndarray
    re_arcsec: np.ndarray
    z_lens: np.ndarray
    z_source: np.ndarray
    radius_arcsec: np.ndarray | None = None
    light_n: float | np.ndarray = 4.0

    def __post_init__(self):
        names = ("log_design_mass", "re_arcsec", "z_lens", "z_source", "light_n")
        values = [np.asarray(getattr(self, name), float) for name in names]
        if self.radius_arcsec is not None:
            values.append(np.asarray(self.radius_arcsec, float))
        if any(value.ndim > 1 for value in values):
            raise ValueError("Design covariates must be scalars or object vectors")
        values = np.broadcast_arrays(*values)
        if not values[0].size or not all(np.all(np.isfinite(a)) for a in values):
            raise ValueError("Nonempty finite design covariates required")
        for name, value in zip(names, values[: len(names)]):
            object.__setattr__(self, name, np.atleast_1d(value).copy())
        if (
            np.any(self.re_arcsec <= 0)
            or np.any(self.z_lens <= 0)
            or np.any(self.z_source <= self.z_lens)
            or np.any((self.light_n < 0.5) | (self.light_n > 8))
        ):
            raise ValueError(
                "Positive size, foreground redshift, and Sersic n in [.5, 8] required"
            )
        if self.radius_arcsec is not None:
            radius = np.asarray(self.radius_arcsec, float)
            radius = np.broadcast_to(radius, self.log_design_mass.shape)
            if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
                raise ValueError("Positive finite source angular radii required")
            object.__setattr__(self, "radius_arcsec", radius.copy())

    def __len__(self):
        return len(self.log_design_mass)


@dataclass(frozen=True)
class PhysicalRelations:
    """Prescribed stellar--halo mass and concentration relations.

    Halo masses are dark M200 relative to critical density at the lens.
    Residual scatters multiply dimensionless grid residuals; their discrete
    RMS need not equal one. ``size_scale`` changes stellar light only, keeping
    the physical core reference mass and radius fixed.
    """

    mass_pivot: float = 11.5
    halo_intercept: float = 13.93
    halo_mass_slope: float = 1.0
    halo_redshift_slope: float = -1.0
    halo_scatter_dex: float = 0.35
    concentration_pivot: float = 4.1
    concentration_mass_slope: float = -0.08
    concentration_redshift_slope: float = -0.33
    concentration_scatter_dex: float = 0.12
    size_scale: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Relation parameters must be finite")
        if (
            self.halo_scatter_dex < 0
            or self.concentration_scatter_dex < 0
            or self.concentration_pivot <= 0
            or self.size_scale <= 0
        ):
            raise ValueError(
                "Nonnegative scatters and positive concentration/size scales required"
            )


def _axis_measure(axis, gaussian_scale=None):
    axis = np.asarray(axis, float)
    if (
        axis.ndim != 1
        or not axis.size
        or not np.all(np.isfinite(axis))
        or np.any(np.diff(axis) <= 0)
    ):
        raise ValueError("State axes must be increasing finite vectors")
    if axis.size == 1:
        weights = np.ones(1)
    else:
        weights = np.empty(axis.size)
        weights[0], weights[-1] = np.diff(axis)[0] / 2, np.diff(axis)[-1] / 2
        weights[1:-1] = (axis[2:] - axis[:-2]) / 2
    return _gaussian_weight(axis, weights, gaussian_scale)


def _gaussian_weight(axis, weights, scale):
    if scale is not None:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Positive finite Gaussian scale, or None, required")
        weights = weights * np.exp(-0.5 * (axis / scale) ** 2)
    if np.any(weights <= 0):
        raise ValueError("State measure underflows its declared Gaussian density")
    return weights / weights.sum()


def _tensor(axes, measures):
    indices = np.indices(tuple(len(a) for a in axes)).reshape(len(axes), -1).T
    coordinates = np.column_stack([a[indices[:, j]] for j, a in enumerate(axes)])
    weights = np.prod(
        np.column_stack([w[indices[:, j]] for j, w in enumerate(measures)]), axis=1
    )
    return coordinates, weights / weights.sum()


def _raw_scores(coordinates):
    raw = coordinates[:, :6].copy()
    raw[:, 0] = np.log(raw[:, 0])
    return np.column_stack((raw, raw[:, 5] ** 2))


@lru_cache(maxsize=1)
def _score_reference():
    axes = [np.asarray(_PRIMARY_AXES[name]) for name in _PRIMARY_AXES]
    measures = [
        _axis_measure(np.log10(axes[0]) / 0.20, 1.0),
        _axis_measure(axes[1], 1.0),
        _axis_measure(axes[2], 1.0),
        _axis_measure(axes[3]),
        _axis_measure(axes[4]),
        _axis_measure(axes[5], 0.1),
    ]
    coordinates, weights = _tensor(axes, measures)
    raw = _raw_scores(coordinates)
    center = weights @ raw
    scale = np.sqrt(weights @ ((raw - center) ** 2))
    standardized = (raw - center) / scale
    product = standardized[:, 3] * standardized[:, 5]
    cross_center = weights @ product
    cross_scale = np.sqrt(weights @ ((product - cross_center) ** 2))
    return center, scale, cross_center, cross_scale


@dataclass(frozen=True)
class MassGrid:
    """Weighted states with a frozen eight-score population mapping.

    The constructor defaults to 72 states for small calculations. Use
    :meth:`primary` for the exact 4,860-state Roman one-core quadrature, or
    :meth:`roman` for its 72,900-state q/redistribution extension. Base weights
    integrate the declared finite-axis densities: mass multiplier lognormal
    with scale .20 dex, standard-normal halo/concentration residuals, uniform
    gradient/cusp, and truncated Gaussian eta. q is a discrete imposed prior;
    redistribution uses Gauss-Legendre integration unless explicit nodes are
    supplied. Neither q nor redistribution adds a population score.

    All grids use the primary measure's score center and RMS, including eta's
    even-width score and the gradient/core interaction. A reduced or refined
    grid therefore need not have zero-mean/unit-RMS scores under its own base
    weights. These coordinates are exponential tilts, not literal Gaussian
    scatter or Pearson correlation parameters.
    """

    mass_multipliers: tuple = (0.7, 1.2, 2.2)
    halo_nodes: tuple = (-1.5, 1.5)
    concentration_nodes: tuple = (0.0,)
    gradients: tuple = (-0.3, 0.0)
    halo_slopes: tuple = (0.7, 1.3)
    eta_nodes: tuple = (-0.2, 0.0, 0.2)
    eta_sigma: float | None = 0.1
    q_nodes: tuple = (5.0,)
    q_probability: tuple | None = None
    compensation_nodes: tuple | None = (0.0,)
    compensation_sigma: float | None = 0.1
    compensation_bound: float = 0.2
    compensation_order: int = 5
    reference_q: float = 5.0
    compensation_radii: tuple = (1.0, 3.0)
    mass_multiplier: np.ndarray = field(init=False, repr=False)
    halo_residual: np.ndarray = field(init=False, repr=False)
    concentration_residual: np.ndarray = field(init=False, repr=False)
    stellar_gradient: np.ndarray = field(init=False, repr=False)
    halo_slope: np.ndarray = field(init=False, repr=False)
    eta_core: np.ndarray = field(init=False, repr=False)
    core_q: np.ndarray = field(init=False, repr=False)
    compensation: np.ndarray = field(init=False, repr=False)
    base_probability: np.ndarray = field(init=False, repr=False)
    scores: np.ndarray = field(init=False, repr=False)
    score_names: tuple = field(init=False, default=SCORE_NAMES)

    def __post_init__(self):
        axes = [np.asarray(getattr(self, name), float) for name in _PRIMARY_AXES]
        if np.any(axes[0] <= 0):
            raise ValueError("Positive mass multipliers required")
        if np.any((axes[3] < -0.5) | (axes[3] > 0)) or np.any(
            (axes[4] < 0.3) | (axes[4] > 1.8)
        ):
            raise ValueError("Gradient or halo cusp outside verified profile support")
        measures = [
            _axis_measure(np.log10(axes[0]) / 0.20, 1.0),
            _axis_measure(axes[1], 1.0),
            _axis_measure(axes[2], 1.0),
            _axis_measure(axes[3]),
            _axis_measure(axes[4]),
            _axis_measure(axes[5], self.eta_sigma),
        ]
        q = np.asarray(self.q_nodes, float)
        _axis_measure(q)
        if np.any(q <= 0):
            raise ValueError("Positive core scales required")
        qw = (
            np.full(len(q), 1 / len(q))
            if self.q_probability is None
            else np.asarray(self.q_probability, float)
        )
        if (
            qw.shape != q.shape
            or not np.all(np.isfinite(qw))
            or np.any(qw <= 0)
            or not np.isclose(qw.sum(), 1.0)
        ):
            raise ValueError("Positive normalized discrete q probabilities required")
        if self.compensation_nodes is None:
            if (
                not np.isfinite(self.compensation_bound)
                or self.compensation_bound <= 0
                or not isinstance(self.compensation_order, int)
                or self.compensation_order < 1
            ):
                raise ValueError(
                    "Positive compensation bound and quadrature order required"
                )
            xi, xw = np.polynomial.legendre.leggauss(self.compensation_order)
            xi, xw = self.compensation_bound * xi, xw / xw.sum()
            xw = _gaussian_weight(xi, xw, self.compensation_sigma)
        else:
            xi = np.asarray(self.compensation_nodes, float)
            xw = _axis_measure(xi, self.compensation_sigma)
        pair = np.asarray(self.compensation_radii, float)
        if (
            not np.isfinite(self.reference_q)
            or self.reference_q <= 0
            or pair.shape != (2,)
            or not np.all(np.isfinite(pair))
            or not 0 < pair[0] < pair[1]
        ):
            raise ValueError(
                "Positive reference q and ordered compensation radii required"
            )
        coordinates, weights = _tensor(axes + [q, xi], measures + [qw, xw])
        names = (
            "mass_multiplier",
            "halo_residual",
            "concentration_residual",
            "stellar_gradient",
            "halo_slope",
            "eta_core",
            "core_q",
            "compensation",
        )
        for j, name in enumerate(names):
            object.__setattr__(self, name, coordinates[:, j])
        center, scale, cross_center, cross_scale = _score_reference()
        standardized = (_raw_scores(coordinates) - center) / scale
        cross = (standardized[:, 3] * standardized[:, 5] - cross_center) / cross_scale
        object.__setattr__(self, "scores", np.column_stack((standardized, cross)))
        object.__setattr__(self, "base_probability", weights)

    def __len__(self):
        return len(self.base_probability)

    @classmethod
    def primary(cls, **kwargs):
        """Exact 4,860-state primary measure, with optional axis refinements."""
        return cls(**(_PRIMARY_AXES | kwargs))

    @classmethod
    def roman(cls, **kwargs):
        """Exact 72,900-state primary × three q × five redistribution rule."""
        return cls(
            **(
                _PRIMARY_AXES
                | dict(q_nodes=(3.0, 5.0, 10.0), compensation_nodes=None)
                | kwargs
            )
        )

    def refine(self, **kwargs):
        """Replace quadrature axes while preserving population coordinates."""
        return replace(self, **kwargs)

    def probabilities(self, tilts):
        """Return H by state unconditioned exponential-tilt probabilities."""
        tilt = np.atleast_2d(np.asarray(tilts, float))
        if tilt.shape[1] != 8 or not np.all(np.isfinite(tilt)):
            raise ValueError("Finite H by 8 population tilts required")
        logw = np.log(self.base_probability)[None, :] + tilt @ self.scores.T
        return np.exp(logw - logsumexp(logw, axis=1)[:, None])


def physical_state_parameters(
    design, grid, relations=PhysicalRelations(), *, state_slice=None
):
    """Return object by state physical parameters, optionally for a state block.

    The finite cores have fixed reference radii independent of stellar size
    sensitivity and source redshift. Their central amplitudes have units of
    Msun/kpc^2 and the broad component has total mass 2 q_ref^2 eta M_ref.
    """
    selected = (
        np.arange(len(grid))
        if state_slice is None
        else np.arange(len(grid))[state_slice]
    )
    selected = np.atleast_1d(selected)
    z = np.log10((1 + design.z_lens) / 1.5)
    halo_mean = (
        relations.halo_intercept
        + relations.halo_mass_slope * (design.log_design_mass - relations.mass_pivot)
        + relations.halo_redshift_slope * z
    )
    halo = (
        halo_mean[:, None] + relations.halo_scatter_dex * grid.halo_residual[selected]
    )
    logc = (
        np.log10(relations.concentration_pivot)
        + relations.concentration_mass_slope * (halo - relations.halo_intercept)
        + relations.concentration_redshift_slope * z[:, None]
        + relations.concentration_scatter_dex * grid.concentration_residual[selected]
    )
    stars = design.log_design_mass[:, None] + np.log10(grid.mass_multiplier[selected])
    kpc = (
        DEFAULT_COSMOLOGY.angular_diameter_distance(design.z_lens).to_value("kpc")
        * np.pi
        / (180 * 3600)
    )
    re = design.re_arcsec * kpc
    unit = 10**design.log_design_mass / (np.pi * re**2)
    q, eta, xi = (
        grid.core_q[selected],
        grid.eta_core[selected],
        grid.compensation[selected],
    )
    a, b = grid.compensation_radii
    radii = np.column_stack((q, np.full_like(q, a), np.full_like(q, b)))
    amplitude = np.column_stack(
        (eta * (grid.reference_q / q) ** 2, xi, -xi * (a / b) ** 2)
    )
    return dict(
        log_stellar_mass=stars,
        log_halo_mass=halo,
        concentration=10**logc,
        stellar_gradient=np.broadcast_to(grid.stellar_gradient[selected], stars.shape),
        halo_slope=np.broadcast_to(grid.halo_slope[selected], stars.shape),
        re_arcsec=design.re_arcsec * relations.size_scale,
        kpc_per_arcsec=kpc,
        core_radii_kpc=re[:, None, None] * radii[None, ...],
        sigma_core_components_msun_kpc2=unit[:, None, None] * amplitude[None, ...],
    )
