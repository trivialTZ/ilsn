import gzip

import numpy as np
import pytest
from scipy.integrate import quad

from ilsn.supernova import (
    fixed_raw_tripp,
    magnification_to_shift,
    read_fitres,
    roman_quality_selection,
    roman_reference_distance_modulus,
    standardized_residual,
)


def test_magnification_shift_has_correct_flux_ratio_and_sign():
    magnification = np.array([0.5, 1.0, 2.0, 10.0])
    shift = magnification_to_shift(magnification)
    np.testing.assert_allclose(10 ** (-0.4 * shift), magnification)
    assert shift[0] > 0 and shift[2] < 0
    assert shift[3] == pytest.approx(-2.5)
    with pytest.raises(ValueError):
        magnification_to_shift(-1.0)


def test_raw_tripp_uses_explicit_reference_and_fixed_control():
    reference = np.array([40.0, 43.0, 45.0])
    magnitude = reference - 19.3 + np.array([-0.2, 0.0, 0.3])
    value = standardized_residual(
        magnitude, 0.0, 0.0, 10.0, reference, control_scalar=-19.3
    )
    np.testing.assert_allclose(value, [-0.2, 0.0, 0.3], atol=1e-14)
    shifted = standardized_residual(
        magnitude, 0.0, 0.0, 10.0, reference + 0.1, control_scalar=-19.3
    )
    np.testing.assert_allclose(shifted, value - 0.1)


def test_host_step_and_fixed_coefficients():
    result = standardized_residual(
        [24.0, 24.0, 24.0],
        1.0,
        0.1,
        [8.0, 10.0, 12.0],
        43.0,
    )
    np.testing.assert_allclose(
        result, 24.0 + 0.1405 - 0.31136 - 43.0 + np.array([-0.005, 0.0, 0.005])
    )


def test_roman_reference_matches_independent_comoving_integration():
    z, hel = np.array([0.05, 0.5, 1.2, 2.475]), np.array([0.051, 0.501, 1.201, 2.474])

    def inverse_expansion(x):
        return 1 / np.sqrt(
            0.315 * (1 + x) ** 3 + 0.685 * (1 + x) ** (-2.4) * np.exp(3 * x / (1 + x))
        )

    integral = np.array([quad(inverse_expansion, 0.0, value)[0] for value in z])
    expected = 5 * np.log10((1 + hel) * 299792.458 / 70 * integral) + 25
    np.testing.assert_allclose(
        roman_reference_distance_modulus(z, hel), expected, atol=1e-12
    )


def test_fixed_raw_tripp_ignores_bbc_and_truth_columns():
    table = {
        "mB": np.array([24.0, 25.0]),
        "x1": np.array([0.2, -0.1]),
        "c": np.array([0.1, -0.05]),
        "HOST_LOGMASS": np.array([9.0, 11.0]),
        "zHD": np.array([0.5, 0.6]),
        "zHEL": np.array([0.501, 0.599]),
        "MURES": np.array([100.0, -100.0]),
        "SIM_DLMAG": np.array([0.0, 0.0]),
    }
    expected = standardized_residual(
        table["mB"],
        table["x1"],
        table["c"],
        table["HOST_LOGMASS"],
        roman_reference_distance_modulus(table["zHD"], table["zHEL"]),
        control_scalar=-19.3,
    )
    np.testing.assert_allclose(fixed_raw_tripp(table, control_scalar=-19.3), expected)
    table["MURES"][:] = np.nan
    table["SIM_DLMAG"][:] = np.nan
    np.testing.assert_allclose(fixed_raw_tripp(table, control_scalar=-19.3), expected)


def test_fitres_preserves_raw_and_bbc_observable_columns_and_string_ids(tmp_path):
    contents = """# observed fits
NVAR: 8
VARNAMES: CID FIELD mB x1 c HOST_LOGMASS zHD MURES
SN: 0001 WIDE 24.1 .2 -.01 10.2 .5 .09
SN: 0002 DEEP 25.2 -.3 .02 9.8 .7 -.13
"""
    path = tmp_path / "example.FITRES.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(contents)
    table = read_fitres(path)
    np.testing.assert_array_equal(table["CID"], ["0001", "0002"])
    np.testing.assert_array_equal(table["FIELD"], ["WIDE", "DEEP"])
    np.testing.assert_allclose(table["mB"], [24.1, 25.2])
    np.testing.assert_allclose(table["MURES"], [0.09, -0.13])
    assert "zHEL" not in table
    with pytest.raises(KeyError):
        fixed_raw_tripp(table)


def test_roman_quality_selection_is_observable_and_has_no_pia_threshold():
    table = {
        "mB": [24.0, 24.0, 24.0],
        "x1": [0.0, 0.0, 0.0],
        "c": [0.0, 0.0, 0.0],
        "HOST_LOGMASS": [10.0, 10.0, 10.0],
        "zHD": [0.5, 0.5, 0.5],
        "zHEL": [0.5, 0.5, 0.5],
        "SNRSUM": [40.0, 40.0, 34.0],
        "FITPROB": [0.5, 0.5, 0.5],
        "PROB_SCONE_PREDICT_ROMAN_DATA": [0.01, 0.99, 0.99],
        "IDSURVEY": [102, 102, 102],
        "FIELD": ["WIDE", "DEEP", "WIDE"],
        "ERRFLAG_FIT": [0, 0, 0],
        "SIM_GENTYPE": [10, 20, 10],
    }
    np.testing.assert_array_equal(roman_quality_selection(table), [True, True, False])
