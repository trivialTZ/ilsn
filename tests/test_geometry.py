"""Analytic single-image limits for singular and finite cores."""

import numpy as np

from ilsn.geometry import radial_caustic_radius, single_image_mask
from ilsn.lens_model import RadialLens


class IsothermalSphere(RadialLens):
    def kappa(self, radius):
        return 0.5 / np.asarray(radius)

    def mean_kappa(self, radius):
        return 1.0 / np.asarray(radius)


def test_sis_single_image_boundary_is_twice_einstein_radius():
    lens = IsothermalSphere()
    caustic = radial_caustic_radius(
        lens, np.geomspace(1e-5, 1, 100), central_deflection_limit=1.0
    )
    np.testing.assert_allclose(caustic, 1.0)
    np.testing.assert_array_equal(
        single_image_mask(lens, [1.1, 1.9, 2.0, 2.1, 4.0], caustic),
        [False, False, False, True, True],
    )


def test_point_mass_cannot_be_classified_as_exactly_single_image():
    lens = IsothermalSphere()
    caustic = radial_caustic_radius(
        lens, np.geomspace(1e-5, 1, 100), central_deflection_limit=np.inf
    )
    assert not single_image_mask(lens, 100.0, caustic)


def test_finite_cored_lens_radial_root_is_refined():
    class CoredLens(RadialLens):
        def mean_kappa(self, r):
            return 4.0 / (1 + np.asarray(r) ** 2)

        def kappa(self, r):
            return 4.0 / (1 + np.asarray(r) ** 2) ** 2

    lens = CoredLens()
    critical = np.sqrt(-3 + np.sqrt(12))
    exact = lens.deflection(critical) - critical
    coarse = radial_caustic_radius(lens, np.geomspace(1e-4, np.sqrt(3), 31), 0.0)
    fine = radial_caustic_radius(lens, np.geomspace(1e-6, np.sqrt(3), 301), 0.0)
    np.testing.assert_allclose([coarse, fine], exact, atol=1e-12)
