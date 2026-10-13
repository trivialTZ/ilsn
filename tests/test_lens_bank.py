"""All-radius admission and independent scalar physical forward checks."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from ilsn.geometry import radial_caustic_radius, single_image_mask
from ilsn.lens_model import CompositeProfile, GalaxyProfile
from ilsn.lens_bank import build_lens_bank, certify_material_support, profile_at_state
from ilsn.mass_grid import LensDesign, MassGrid, PhysicalRelations


def test_certificate_matches_independent_global_one_core_minimum():
    rs, rho, g, rc = 10.0, 1.0, 1.0, 5.0

    def halo_over_unit_core(log_r, derivative):
        r = np.exp(log_r)
        x, t = r / rs, r / rc
        halo = rho * x**-g * (1 + x) ** (g - 3)
        unit_core = 2 / (np.pi * rc) * (1 + t * t) ** -2
        if derivative:
            halo *= (g + 3 * x) / (1 + x)
            unit_core *= 4 * t * t / (1 + t * t)
        return halo / unit_core

    limits = [
        minimize_scalar(
            lambda lr: halo_over_unit_core(lr, derivative),
            bounds=(-15, 15),
            method="bounded",
            options={"xatol": 1e-12},
        ).fun
        for derivative in (False, True)
    ]
    boundary = -min(limits)
    inside = certify_material_support(rs, rho, g, [rc], [boundary * (1 - 1e-4)])
    outside = certify_material_support(rs, rho, g, [rc], [boundary * (1 + 1e-4)])
    assert inside["physical_valid"] and not inside["support_unresolved"]
    assert outside["support_invalid"] and not outside["support_unresolved"]


def test_joint_certificate_keeps_cancellations_and_unresolved_boundaries():
    exact = certify_material_support(
        [10.0, 25.0], [1.0, 0.5], [0.7, 1.3], [5.0, 5.0], [-1e6, 1e6]
    )
    assert np.all(exact["physical_valid"])
    kwargs = dict(
        halo_rs=10.0,
        halo_rhos=1.0,
        halo_slope=1.0,
        radii_kpc=[3.0, 3.01],
        sigma_components=[-30.0, 30.0],
    )
    shallow = certify_material_support(**kwargs, initial_nodes=9, max_depth=1)
    fine = certify_material_support(**kwargs)
    assert shallow["support_unresolved"] and not shallow["support_invalid"]
    assert not shallow["physical_valid"]
    assert fine["physical_valid"]


def _grid():
    return MassGrid(
        mass_multipliers=(1.2,),
        halo_nodes=(0.0,),
        concentration_nodes=(0.0,),
        gradients=(-0.15,),
        halo_slopes=(1.0,),
        eta_nodes=(-0.1, 0.0, 0.1),
        q_nodes=(3.0, 5.0, 10.0),
        compensation_nodes=(-0.2, 0.0, 0.2),
    )


def test_bank_matches_independently_constructed_physical_profiles():
    design, grid = LensDesign(11.5, 1.0, 0.5, 1.5, 10.0), _grid()
    bank = build_lens_bank(design, grid)
    assert np.any(~bank["physical_valid"])
    assert not np.any(bank["support_unresolved"])
    assert len(grid) == 27 and bank["physical_valid"].shape == (1, 27)
    base = GalaxyProfile(
        0.5,
        1.0,
        1.2 * 10**11.5,
        10**13.93,
        4.1,
        halo_inner_slope=1.0,
        stellar_ml_gradient=-0.15,
    )
    for k in np.flatnonzero(bank["physical_valid"][0]):
        profile = CompositeProfile.from_reference(
            base,
            10**11.5,
            base.re_kpc,
            grid.eta_core[k],
            grid.core_q[k],
            grid.compensation[k],
        )
        lens = profile.lens(1.5)
        assert bank["kappa"][0, k] == pytest.approx(float(lens.kappa(10.0)), rel=2e-14)
        assert bank["mean_kappa"][0, k] == pytest.approx(
            float(lens.mean_kappa(10.0)), rel=2e-14
        )
        theta_e = lens.theta_E()
        assert bank["theta_e_arcsec"][0, k] == pytest.approx(theta_e, rel=2e-10)
        assert bank["s_E"][0, k] == pytest.approx(lens.s_E(), rel=2e-10)
        assert bank["xi_E"][0, k] == pytest.approx(lens.xi_E(), abs=2e-10)
        assert bank["theta_e_over_re_design"][0, k] == pytest.approx(theta_e, rel=2e-10)
        radii = np.geomspace(
            lens.minimum_radius_arcsec * (1 + 1e-6), max(20.0, 2 * theta_e), 2049
        )
        caustic = radial_caustic_radius(lens, radii, 0.0)
        assert bank["caustic_area_arcsec2"][0, k] == pytest.approx(
            np.pi * caustic**2, rel=2e-10
        )
        assert bank["single_image"][0, k] == single_image_mask(lens, 10.0, caustic)
    excluded = ~bank["physical_valid"]
    assert np.all(bank["kappa"][excluded] == 0)
    assert not np.any(bank["projection_valid"][excluded])
    assert np.all(grid.base_probability > 0)
    assert grid.base_probability.sum() == pytest.approx(1)


def test_material_support_is_independent_of_source_and_stellar_size():
    grid = _grid()
    source = build_lens_bank(LensDesign(11.5, 1.0, 0.5, 1.5, 10.0), grid)
    target = build_lens_bank(
        LensDesign(11.5, 1.0, 0.5, 3.0),
        grid,
        PhysicalRelations(size_scale=1.5),
        source=False,
    )
    for name in ("physical_valid", "support_invalid", "support_unresolved"):
        np.testing.assert_array_equal(source[name], target[name])
    assert not np.any(target["projection_valid"])
    assert np.any(target["endpoint_valid"])
    assert np.any(abs(source["theta_e_arcsec"] - target["theta_e_arcsec"]) > 1e-3)


def test_state_slices_preserve_complete_predictions():
    design, grid = LensDesign(11.5, 1.0, 0.5, 1.5, 10.0), _grid()
    whole = build_lens_bank(design, grid)
    first = build_lens_bank(design, grid, state_slice=slice(None, 13))
    second = build_lens_bank(design, grid, state_slice=slice(13, None))
    for name in whole:
        np.testing.assert_allclose(
            np.concatenate((first[name], second[name]), axis=1),
            whole[name],
            rtol=1e-13,
            atol=1e-13,
        )
    profile = profile_at_state(design, grid, 0, 13)
    assert profile.base.stellar_mass_msun == pytest.approx(1.2 * 10**11.5)


def test_source_radius_and_numerical_resolution_required():
    with pytest.raises(ValueError, match="radius_arcsec"):
        build_lens_bank(LensDesign(11.5, 1.0, 0.5, 1.5), MassGrid())
    with pytest.raises(ValueError, match="33"):
        build_lens_bank(
            LensDesign(11.5, 1.0, 0.5, 1.5, 10.0), MassGrid(), radial_nodes=8
        )
