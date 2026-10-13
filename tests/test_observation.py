"""Physical photons, paired environments and fixed inner observations."""

import numpy as np

from ilsn.observation import (
    paired_los,
    source_sample,
    td_selection,
    imaging_log_likelihood,
)
from ilsn.likelihood import SourceLikelihood
from ilsn.population import MassPopulation
from ilsn.response import ResponseCalibration


def test_los_preserves_pairs_and_external_jacobian():
    k = np.array([[0.03, 0.01, 0.04, 0.02]])
    los = paired_los(k, 2 * k, rows=2, orientations=2)
    np.testing.assert_allclose(los["gamma"], 2 * los["kappa"])
    physics = dict(
        kappa=np.array([[0.1, 0.2]]),
        mean_kappa=np.array([[0.3, 0.4]]),
        kappa_star=np.array([[0.02, 0.03]]),
    )
    physics.update(
        {
            key: np.ones((1, 2), bool)
            for key in (
                "physical_valid",
                "single_image",
                "geometry_valid",
                "projection_valid",
            )
        }
    )
    sample = source_sample(
        [-0.3],
        [1.2],
        physics,
        radius_arcsec=[3.0],
        re_arcsec=[1.0],
        los=los,
        los_cell=[0],
    )
    a = 1 - physics["kappa"][..., None] - los["kappa"][:, None, :]
    gamma2 = (
        0.2**2
        + los["gamma"][:, None, :] ** 2
        + 0.4 * los["gamma"][:, None, :] * los["cos2phi"]
    )
    np.testing.assert_allclose(sample.photon_shift_mag, 2.5 * np.log10(a * a - gamma2))
    np.testing.assert_allclose(sample.extra_variance_mag2[0, :, 0], [0.0008, 0.0012])
    np.testing.assert_allclose(sample.dust_template_mag, 0.01 * (2 / 3) ** 2)


def test_td_selection_leaves_physical_admission_intact():
    physical = np.ones((1, 3), bool)
    area = np.array([[1.0, 2.0, 3.0]])
    selected = td_selection([[0.01, 1.0, 10.0]], area, [0.3], [1.2], physical)
    np.testing.assert_array_equal(physical, True)
    np.testing.assert_allclose(selected, [[0.0, 2.0, 0.0]])
    np.testing.assert_allclose(
        td_selection(
            [[0.01, 1.0, 10.0]], area, [0.3], [1.2], physical, velocity_range=None
        ),
        area,
    )


def test_compressed_imaging_uses_observed_errors():
    value = imaging_log_likelihood(
        [[1.0, 1.1, np.nan]], [[0.5, 0.7, 0.0]], [1.0], [0.5], [0.05], [0.2]
    )
    np.testing.assert_allclose(value[:, :2], [[0.0, -2.5]])
    assert np.isneginf(value[0, 2])


def test_missing_target_geometry_has_zero_selection_without_changing_material_support():
    material = np.ones((1, 3), bool)
    theta = np.array([[1.0, np.nan, 0.01]])
    area = np.array([[2.0, np.nan, 1.0]])
    selected = td_selection(theta, area, [0.3], [1.2], material)
    np.testing.assert_allclose(selected, [[2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(material, True)
    np.testing.assert_allclose(
        td_selection(theta, area, [0.3], [1.2], material, velocity_range=None),
        [[2.0, 0.0, 1.0]],
    )


def test_four_shear_orientations_equal_their_coalesced_two_component_rule():
    los = paired_los([[0.01, 0.03]], [[0.08, 0.04]], rows=2, orientations=4)
    # Midpoint phi and pi-phi have the same cos(2phi). Combining their
    # empirical mass must preserve the complete selected brightness law.
    compact = {
        key: value.reshape(1, 2, 4)[..., :2].reshape(1, 4)
        for key, value in los.items()
        if key in ("kappa", "gamma")
    }
    compact["cos2phi"] = los["cos2phi"].reshape(2, 4)[:, :2].ravel()
    compact["probability"] = np.full(4, 0.25)
    physics = dict(
        kappa=np.array([[0.1, 0.2]]),
        mean_kappa=np.array([[0.3, 0.4]]),
        kappa_star=np.array([[0.02, 0.03]]),
    )
    physics.update(
        {
            key: np.ones((1, 2), bool)
            for key in (
                "physical_valid",
                "single_image",
                "geometry_valid",
                "projection_valid",
            )
        }
    )
    pop = MassPopulation([0.4, 0.6], [[-1.0], [1.0]])
    cal = ResponseCalibration.gaussian(
        [0.8, 1.6], [-1.0, 0.0, 1.0], delivery_probability=[[0.9, 0.5, 0.2]], n_draws=1
    )
    likelihood = []
    for rule in (los, compact):
        sample = source_sample(
            [-0.3],
            [1.2],
            physics,
            radius_arcsec=[3.0],
            re_arcsec=[1.0],
            los=rule,
            los_cell=[0],
        )
        likelihood.append(
            SourceLikelihood(sample, cal).log_likelihood(pop, [[-0.5], [0.5]])
        )
    np.testing.assert_allclose(*likelihood, atol=2e-14)
