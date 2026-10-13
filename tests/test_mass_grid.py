"""Reference measures, frozen score coordinates and physical normalization."""

import numpy as np
import pytest

from ilsn.mass_grid import (
    LensDesign,
    MassGrid,
    PhysicalRelations,
    physical_state_parameters,
)


def _trapezoid_density(nodes, sigma=None):
    x = np.asarray(nodes, float)
    widths = np.r_[np.diff(x)[0] / 2, (x[2:] - x[:-2]) / 2, np.diff(x)[-1] / 2]
    if sigma is not None:
        widths *= np.exp(-x * x / (2 * sigma * sigma))
    return widths / widths.sum()


def test_exact_primary_axis_measures_and_score_moments():
    grid = MassGrid.primary()
    assert len(grid) == 4860
    assert grid.scores.shape == (4860, 8)
    mass = np.array([0.7, 0.9, 1.2, 1.6, 2.2])
    mass_weights = _trapezoid_density(np.log10(mass), 0.20)
    eta = np.linspace(-0.2, 0.2, 9)
    eta_weights = _trapezoid_density(eta, 0.1)
    for value, expected in zip(mass, mass_weights):
        assert grid.base_probability[
            grid.mass_multiplier == value
        ].sum() == pytest.approx(expected, abs=2e-15)
    for value, expected in zip(eta, eta_weights):
        assert grid.base_probability[grid.eta_core == value].sum() == pytest.approx(
            expected, abs=2e-15
        )
    np.testing.assert_allclose(grid.base_probability @ grid.scores, 0, atol=6e-15)
    np.testing.assert_allclose(grid.base_probability @ (grid.scores**2), 1, atol=6e-15)
    covariance = (grid.scores.T * grid.base_probability) @ grid.scores
    np.testing.assert_allclose(covariance, np.eye(8), atol=6e-15)


def test_refined_grid_maps_into_primary_coordinates():
    primary = MassGrid.primary()
    receiving = MassGrid(
        mass_multipliers=(1.1,),
        halo_nodes=(0.25,),
        concentration_nodes=(-0.4,),
        gradients=(-0.22,),
        halo_slopes=(1.2,),
        eta_nodes=(0.06,),
    )
    raw = np.column_stack(
        (
            np.log(primary.mass_multiplier),
            primary.halo_residual,
            primary.concentration_residual,
            primary.stellar_gradient,
            primary.halo_slope,
            primary.eta_core,
            primary.eta_core**2,
        )
    )
    mean = np.average(raw, axis=0, weights=primary.base_probability)
    sd = np.sqrt(
        np.average((raw - mean) ** 2, axis=0, weights=primary.base_probability)
    )
    point = np.array([np.log(1.1), 0.25, -0.4, -0.22, 1.2, 0.06, 0.06**2])
    mapped = (point - mean) / sd
    np.testing.assert_allclose(receiving.scores[0, :7], mapped, atol=2e-14)
    assert receiving.scores[0, 7] == pytest.approx(mapped[3] * mapped[5], abs=2e-14)
    # A fixed receiving state still uses the nonzero frozen coordinates.
    assert np.linalg.norm(receiving.scores) > 1


def test_family_weights_add_no_population_scores():
    primary, family = MassGrid.primary(), MassGrid.roman()
    assert len(MassGrid()) == 72
    assert len(family) == 72900
    assert family.score_names == primary.score_names
    np.testing.assert_array_equal(family.scores, np.repeat(primary.scores, 15, axis=0))
    for q in (3, 5, 10):
        assert family.base_probability[family.core_q == q].sum() == pytest.approx(1 / 3)
    nodes, weights = np.polynomial.legendre.leggauss(5)
    nodes *= 0.2
    weights *= np.exp(-0.5 * (nodes / 0.1) ** 2)
    weights /= weights.sum()
    for xi, expected in zip(nodes, weights):
        assert family.base_probability[
            family.compensation == xi
        ].sum() == pytest.approx(expected)
    np.testing.assert_allclose(
        family.probabilities(np.zeros(8))[0], family.base_probability, rtol=2e-15
    )
    tilted = family.probabilities(np.array([0.2, 0, 0, -0.1, 0, 0.3, 0, 0.1]))[0]
    assert tilted.sum() == pytest.approx(1)
    assert not np.allclose(tilted, family.base_probability)


def test_relations_and_finite_core_reference_normalizations():
    design = LensDesign(11.5, 1.0, 0.5, 1.5)
    grid = MassGrid(
        mass_multipliers=(1.2,),
        halo_nodes=(0.0,),
        concentration_nodes=(0.0,),
        gradients=(-0.15,),
        halo_slopes=(1.0,),
        eta_nodes=(0.08,),
        q_nodes=(3.0, 5.0, 10.0),
        compensation_nodes=(-0.1, 0.1),
    )
    p = physical_state_parameters(design, grid)
    np.testing.assert_allclose(p["log_stellar_mass"], 11.5 + np.log10(1.2))
    np.testing.assert_allclose(p["log_halo_mass"], 13.93)
    np.testing.assert_allclose(p["concentration"], 4.1)
    masses = 2 * np.pi * p["core_radii_kpc"] ** 2 * p["sigma_core_components_msun_kpc2"]
    np.testing.assert_allclose(masses[..., 0], 50 * 0.08 * 10**11.5, rtol=2e-15)
    np.testing.assert_allclose(masses[..., 1] + masses[..., 2], 0, atol=1e-4)
    altered = physical_state_parameters(
        LensDesign(11.5, 1.0, 0.5, 3.0), grid, PhysicalRelations(size_scale=1.5)
    )
    np.testing.assert_array_equal(altered["core_radii_kpc"], p["core_radii_kpc"])
    np.testing.assert_array_equal(
        altered["sigma_core_components_msun_kpc2"], p["sigma_core_components_msun_kpc2"]
    )
    np.testing.assert_allclose(altered["re_arcsec"], 1.5)


def test_object_alignment_and_state_refinement():
    design = LensDesign([11.2, 11.5], 0.8, [0.3, 0.5], 1.5, radius_arcsec=10.0)
    assert len(design) == 2
    np.testing.assert_array_equal(design.re_arcsec, [0.8, 0.8])
    grid = MassGrid().refine(eta_nodes=(-0.1, 0.0, 0.1))
    p = physical_state_parameters(design, grid, state_slice=slice(1, 4))
    assert p["log_halo_mass"].shape == (2, 3)
    assert p["core_radii_kpc"].shape == (2, 3, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mass_multipliers=(0.0,)),
        dict(q_nodes=(-1.0,)),
        dict(eta_sigma=0.0),
        dict(q_probability=(0.5,)),
        dict(compensation_order=0, compensation_nodes=None),
        dict(compensation_radii=(3.0, 1.0)),
        dict(gradients=(0.1,)),
    ],
)
def test_invalid_state_measures_rejected(kwargs):
    with pytest.raises(ValueError):
        MassGrid(**kwargs)


def test_invalid_design_or_relation_rejected():
    with pytest.raises(ValueError):
        LensDesign(11.0, 1.0, 0.5, 0.4)
    with pytest.raises(ValueError):
        LensDesign(11.0, 0.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        PhysicalRelations(halo_scatter_dex=-0.1)
