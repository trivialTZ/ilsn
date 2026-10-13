import numpy as np
import pytest

from ilsn.response import ResponseCalibration


def test_interpolates_variance_and_retains_joint_draw():
    mean = np.array([[-0.2, 0.4], [1.0, 2.0]])
    variance = np.array([[0.01, 0.09], [0.04, 0.16]])
    delivery = np.array([[0.9, 0.3], [0.8, 0.4]])
    calibration = ResponseCalibration(
        [0.0, 1.0, 2.0],
        [-0.5, 0.5],
        mean,
        variance,
        delivery,
        np.stack([mean + 0.05, mean - 0.1]),
        np.stack([variance, variance * 2]),
        np.stack([delivery, delivery / 2]),
        draw_probability=[1, 3],
    )
    value = calibration.evaluate([0.0, 0.0], [0.4, 1.4], draw=1)
    np.testing.assert_allclose(value["mean_mag"], [0.0, 1.4])
    np.testing.assert_allclose(value["variance_mag2"], [0.1, 0.2])
    np.testing.assert_allclose(value["delivery_probability"], [0.3, 0.3])
    np.testing.assert_allclose(calibration.draw_probability, [0.25, 0.75])
    point = calibration.evaluate(0.0, 0.4, draw=None)
    assert point["variance_mag2"] == pytest.approx(0.05)
    assert point["variance_mag2"] != pytest.approx(((0.1 + 0.3) / 2) ** 2)


def test_gaussian_photon_shift_identity_and_common_zero_point_covariance():
    calibration = ResponseCalibration.gaussian(
        [0.2, 0.8, 1.4],
        [-0.4, 0.0, 0.4],
        scatter_mag=0.12,
        zero_point_sigma=0.03,
        n_draws=12000,
        rng=52,
    )
    value = calibration.evaluate([-0.2, 0.1], [0.5, 1.1], draw=None)
    np.testing.assert_allclose(value["mean_mag"], [-0.2, 0.1])
    np.testing.assert_allclose(value["variance_mag2"], 0.12**2)
    np.testing.assert_allclose(value["delivery_probability"], 1.0)
    offsets = calibration.draw_mean_mag - calibration.point_mean_mag
    np.testing.assert_allclose(offsets[:, 0, 0], offsets[:, 1, 2], atol=1e-15)
    covariance = np.cov(
        calibration.draw_mean_mag[:, 0, 1], calibration.draw_mean_mag[:, 1, 1]
    )
    np.testing.assert_allclose(covariance, 0.03**2, rtol=0.025)


def test_fixed_true_redshifts_do_not_authorize_continuous_z_interpolation():
    calibration = ResponseCalibration.gaussian(
        [0.2, 0.8, 1.4],
        [-0.4, 0.4],
        redshift_nodes=[0.5, 1.1],
        redshift_tolerance=1e-6,
    )
    assert calibration.evaluate(0.0, 0.5)["mean_mag"] == pytest.approx(0.0)
    assert calibration.evaluate(0.0, 0.5 + 5e-7)["mean_mag"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="redshift"):
        calibration.evaluate(0.0, 0.6)


@pytest.mark.parametrize(
    "shift,z",
    [(-0.6, 0.5), (0.6, 0.5), (0.0, 1.4), (0.0, 0.1), (np.nan, 0.5), (0.0, np.inf)],
)
def test_required_physical_domain_is_not_extrapolated(shift, z):
    calibration = ResponseCalibration.gaussian([0.2, 1.4], [-0.4, 0.4])
    with pytest.raises(ValueError):
        calibration.evaluate(shift, z)
    value = calibration.evaluate([0.0, shift], [0.5, z], required_mask=[True, False])
    assert value["mean_mag"][1] == 0.0
    assert value["variance_mag2"][1] == 1.0
    assert value["delivery_probability"][1] == 0.0


def test_unmeasured_cells_fail_but_missing_adjacent_knot_is_irrelevant():
    mean, variance, delivery = (
        np.array([x])
        for x in ([0.0, np.nan, 0.4], [0.01, np.nan, 0.04], [1.0, 0.0, 0.5])
    )
    calibration = ResponseCalibration(
        [0.2, 0.8],
        [0.0, 0.2, 0.4],
        mean,
        variance,
        delivery,
        mean[None],
        variance[None],
        delivery[None],
    )
    assert calibration.evaluate(0.0, 0.5)["mean_mag"] == 0.0
    assert calibration.evaluate(0.4, 0.5)["mean_mag"] == 0.4
    with pytest.raises(ValueError, match="unmeasured"):
        calibration.evaluate(0.1, 0.5)
    with pytest.raises(ValueError, match="unmeasured"):
        calibration.evaluate(0.2, 0.5)


def test_generated_selection_keeps_failed_parent_denominators():
    knots = np.array([-0.2, 0.0, 0.2])
    parent = np.repeat(np.arange(4), 3)
    shift = np.tile(knots, 4)
    raw = np.repeat([-19.4, -19.2, -19.0, -18.8], 3) + shift
    selected = (shift != 0.2) | (parent < 2)
    raw[~selected] = np.nan
    calibration = ResponseCalibration.from_generated(
        shift,
        np.full(12, 0.5),
        selected,
        raw,
        parent,
        z_edges=[0.2, 0.8],
        shift_nodes=knots,
        n_draws=200,
        rng=8,
    )
    assert calibration.control_scalar == pytest.approx(-19.1)
    point = calibration.evaluate(knots, 0.5, draw=None)
    np.testing.assert_allclose(point["delivery_probability"], [1.0, 1.0, 0.5])
    np.testing.assert_allclose(point["mean_mag"], [-0.2, 0.0, 0.0], atol=1e-13)
    np.testing.assert_allclose(point["variance_mag2"], [1 / 15, 1 / 15, 0.02])
    assert calibration.draw_delivery_probability[:, 0, 2].std() > 0.1
    assert calibration.draw_mean_mag[:, 0, 1].std() > 0.05


def test_matched_parent_draws_preserve_shift_and_variance_relations():
    knots = np.array([-0.2, 0.0, 0.2])
    parent = np.repeat(np.arange(8), 3)
    shift = np.tile(knots, 8)
    raw = np.repeat(np.linspace(-19.4, -18.8, 8), 3) + shift
    calibration = ResponseCalibration.from_generated(
        shift,
        np.full(24, 0.5),
        np.ones(24, bool),
        raw,
        parent,
        z_edges=[0.2, 0.8],
        shift_nodes=knots,
        n_draws=200,
        rng=9,
    )
    difference = (
        calibration.draw_mean_mag[:, 0] - calibration.draw_mean_mag[:, 0, 1, None]
    )
    np.testing.assert_allclose(
        difference, np.broadcast_to(knots, difference.shape), atol=1e-14
    )
    np.testing.assert_allclose(
        calibration.draw_variance_mag2[:, 0, 0],
        calibration.draw_variance_mag2[:, 0, 2],
        atol=1e-14,
    )
    np.testing.assert_allclose(calibration.draw_delivery_probability, 1.0)
    assert calibration.draw_mean_mag[:, 0, 1].std() > 0.01


def test_fixed_control_averages_redshift_controls_without_population_reweighting():
    parent = np.repeat(np.arange(6), 2)
    z = np.repeat([0.5, 0.5, 0.5, 0.5, 1.1, 1.1], 2)
    shift = np.tile([0.0, 0.2], 6)
    raw = np.repeat([-19.5, -19.3, -19.1, -18.9, -18.2, -18.0], 2) + shift
    calibration = ResponseCalibration.from_generated(
        shift,
        z,
        np.ones(12, bool),
        raw,
        parent,
        z_edges=[0.2, 0.8, 1.4],
        shift_nodes=[0.0, 0.2],
        n_draws=20,
        rng=1,
    )
    assert calibration.control_scalar == pytest.approx((-19.2 - 18.1) / 2)
    assert calibration.control_scalar != pytest.approx(raw[shift == 0.0].mean())
    point = calibration.evaluate(0.0, [0.5, 1.1], draw=None)
    assert point["mean_mag"].mean() == pytest.approx(0.0, abs=1e-13)


def test_saved_calibration_preserves_shared_uncertainty_and_z_support(tmp_path):
    original = ResponseCalibration.gaussian(
        [0.2, 0.8],
        [-0.4, 0.4],
        zero_point_sigma=0.05,
        n_draws=20,
        rng=34,
        control_scalar=-19.3,
        redshift_nodes=[0.5],
    )
    saved = ResponseCalibration.load(original.save(tmp_path / "response"))
    for draw in (None, 0, 19):
        before = original.evaluate([-0.4, -0.1, 0.4], 0.5, draw)
        after = saved.evaluate([-0.4, -0.1, 0.4], 0.5, draw)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
    assert saved.control_scalar == -19.3
    with pytest.raises(ValueError, match="redshift"):
        saved.evaluate(0.0, 0.6)
