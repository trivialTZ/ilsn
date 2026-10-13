"""SN magnification shifts, fixed standardization and SNANA table input."""

import gzip
from pathlib import Path

import numpy as np
from scipy.special import expit


def magnification_to_shift(magnification):
    """Convert positive flux magnification to a photon magnitude shift.

    Magnification greater than one brightens the source and gives a negative
    shift. For multiple images, use the flux magnification appropriate to the
    measured observable; signed image parity is not a flux magnification.

    :param magnification: Positive finite flux magnification.
    :return: ``-2.5 log10(magnification)`` in magnitudes.
    :rtype: numpy.ndarray
    """
    magnification = np.asarray(magnification, float)
    if np.any(~np.isfinite(magnification) | (magnification <= 0)):
        raise ValueError("Flux magnification must be positive and finite")
    return -2.5 * np.log10(magnification)


def standardized_residual(
    mB,
    x1,
    color,
    host_logmass,
    reference_distance_modulus,
    *,
    control_scalar=0.0,
    alpha=0.1405,
    beta=3.1136,
    gamma=0.010,
    host_transition_mass=10.0,
    host_transition_width=0.01,
):
    """Compute a raw Tripp residual relative to an explicit distance reference.

    The observable is ``mB + alpha*x1 - beta*color - host_term - mu_ref - C0``,
    where ``host_term = gamma*(0.5 - expit((host_logmass-10)/0.01))`` at the
    default settings. The reference distance is an input to standardization,
    rather than an inferred cosmology. C0 must match the response calibration's
    fixed control origin. No simulation truth or BBC bias correction enters
    this arithmetic; a BBC ``MURES`` column is a different observable.

    :param mB: Observed SALT peak apparent magnitude.
    :param x1: Observed SALT light-curve shape parameter.
    :param color: Observed SALT color parameter ``c``.
    :param host_logmass: Observed host log10 stellar mass in solar masses.
    :param reference_distance_modulus: Explicit reference distance in magnitudes.
    :param control_scalar: Fixed raw-control origin C0 in magnitudes.
    :param alpha: Fixed shape-standardization coefficient.
    :param beta: Fixed color-standardization coefficient.
    :param gamma: Fixed host-step amplitude in magnitudes.
    :param host_transition_mass: Midpoint of the smooth host step.
    :param host_transition_width: Positive width in log10 host mass.
    :return: Standardized residual in the fixed C0 coordinate.
    :rtype: numpy.ndarray
    """
    arrays = np.broadcast_arrays(
        *(
            np.asarray(x, float)
            for x in (
                mB,
                x1,
                color,
                host_logmass,
                reference_distance_modulus,
                control_scalar,
            )
        )
    )
    if any(np.any(~np.isfinite(x)) for x in arrays):
        raise ValueError(
            "Observed standardization inputs and reference distances must be finite"
        )
    if (
        not np.isfinite(host_transition_width)
        or host_transition_width <= 0
        or not np.all(np.isfinite([alpha, beta, gamma, host_transition_mass]))
    ):
        raise ValueError(
            "Finite coefficients and a positive host-transition width are required"
        )
    magnitude, shape, color, mass, distance, origin = arrays
    host = gamma * (0.5 - expit((mass - host_transition_mass) / host_transition_width))
    return magnitude + alpha * shape - beta * color - host - distance - origin


def roman_reference_distance_modulus(zHD, zHEL, nquad=128):
    """Evaluate the frozen reference used by the Roman raw-Tripp calibration.

    This flat reference has Om=0.315, w0=-0.8, wa=-1 and H0=70 km/s/Mpc.
    The comoving integral uses observed ``zHD`` and the luminosity-distance
    prefactor uses observed ``zHEL``. It defines the observable, rather than
    an assumed fitted source or lens cosmology.

    :param zHD: Positive observed Hubble-diagram redshift.
    :param zHEL: Positive observed heliocentric redshift.
    :param nquad: Number of Gauss-Legendre integration points.
    :return: Frozen reference distance modulus in magnitudes.
    :rtype: numpy.ndarray
    """
    redshift, heliocentric = np.broadcast_arrays(
        np.asarray(zHD, float), np.asarray(zHEL, float)
    )
    if np.any(
        ~np.isfinite(redshift)
        | ~np.isfinite(heliocentric)
        | (redshift <= 0)
        | (heliocentric <= 0)
    ):
        raise ValueError("Positive finite observed zHD and zHEL are required")
    if not isinstance(nquad, (int, np.integer)) or nquad < 2:
        raise ValueError("At least two integer quadrature points are required")
    nodes, weights = np.polynomial.legendre.leggauss(nquad)
    integral = np.empty(redshift.size)
    flat = redshift.ravel()
    for start in range(0, len(flat), 1024):
        z = flat[start : start + 1024, None]
        integration_z = z * (nodes + 1) / 2
        expansion = np.sqrt(
            0.315 * (1 + integration_z) ** 3
            + 0.685
            * (1 + integration_z) ** (-2.4)
            * np.exp(3 * integration_z / (1 + integration_z))
        )
        integral[start : start + 1024] = (
            z[:, 0] / 2 * np.sum(weights / expansion, axis=1)
        )
    luminosity_distance = (
        (1 + heliocentric) * 299792.458 / 70 * integral.reshape(redshift.shape)
    )
    return 5 * np.log10(luminosity_distance) + 25


def fixed_raw_tripp(
    records, *, control_scalar=0.0, alpha=0.1405, beta=3.1136, gamma=0.010
):
    """Compute the frozen Roman observable from explicit FITRES columns.

    Required columns are ``mB``, ``x1``, ``c``, ``HOST_LOGMASS``, ``zHD`` and
    ``zHEL``. With C0=0 this returns the raw residual before control centering;
    for inference, pass the calibration's fixed ``control_scalar``. A table
    containing only BBC ``MURES`` cannot supply this observable.

    :param records: Mapping of column names to observed arrays.
    :param control_scalar: Fixed calibration origin, or zero for raw residuals.
    :param alpha: Fixed shape-standardization coefficient.
    :param beta: Fixed color-standardization coefficient.
    :param gamma: Fixed host-step amplitude.
    :return: Raw-Tripp residual, optionally centered by C0.
    :rtype: numpy.ndarray
    """
    return standardized_residual(
        records["mB"],
        records["x1"],
        records["c"],
        records["HOST_LOGMASS"],
        roman_reference_distance_modulus(records["zHD"], records["zHEL"]),
        control_scalar=control_scalar,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def roman_quality_selection(records):
    """Return the frozen observable Roman delivery mask before BBC.

    This mask describes the calibration's light-curve quality/classification
    stage. It imposes no hard pIa threshold or truth-type cut. Restricting a
    simulated calibration population to true Ia is a separate input choice.

    :param records: Observed FITRES columns using the original SNANA names.
    :return: Boolean membership in the frozen selected observable sample.
    :rtype: numpy.ndarray
    """
    numeric = (
        "mB",
        "x1",
        "c",
        "HOST_LOGMASS",
        "zHD",
        "zHEL",
        "SNRSUM",
        "FITPROB",
        "PROB_SCONE_PREDICT_ROMAN_DATA",
    )
    values = {key: np.asarray(records[key], float) for key in numeric}
    finite = np.logical_and.reduce([np.isfinite(values[key]) for key in numeric])

    def between(key, lower, upper):
        return (values[key] >= lower) & (values[key] <= upper)

    return (
        finite
        & (np.asarray(records["IDSURVEY"]) == 102)
        & np.isin(records["FIELD"], ["WIDE", "DEEP"])
        & (np.asarray(records["ERRFLAG_FIT"]) == 0)
        & between("x1", -3, 3)
        & between("c", -0.3, 0.3)
        & between("FITPROB", 0.01, 1.01)
        & between("SNRSUM", 35, 1e8)
        & (values["zHD"] > 0.02)
        & (values["zHD"] < 3)
        & (values["zHEL"] > 0)
        & (values["zHEL"] < 3)
        & between("PROB_SCONE_PREDICT_ROMAN_DATA", 0, 1)
    )


def read_fitres(path):
    """Read a SNANA FITRES text table without changing observable columns.

    A ``VARNAMES:`` header declares the columns. Data rows may have ``SN:``,
    ``ROW:`` or ``LINE:`` prefixes, or no prefix. Comments and metadata lines
    such as ``NVAR:`` are skipped. Numeric columns become float arrays; CID,
    SNID, CCID and FIELD remain strings, preserving leading-zero identifiers.
    Compressed ``.gz`` files are supported. No column aliases, cuts, BBC
    residual conversion or automatic cosmology choice are applied.

    :param path: Explicit FITRES or FITRES.gz path.
    :return: Mapping from the original column names to NumPy arrays.
    :rtype: dict
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    columns = None
    rows = []
    with opener(path, "rt") as stream:
        for number, line in enumerate(stream, 1):
            words = line.partition("#")[0].split()
            if not words:
                continue
            if words[0] == "VARNAMES:":
                header = words[1:]
                if not header or len(set(header)) != len(header):
                    raise ValueError("FITRES column names must be nonempty and unique")
                if columns is not None and columns != header:
                    raise ValueError(
                        "Repeated FITRES headers must use the same columns"
                    )
                columns = header
                continue
            if columns is None:
                continue
            if words[0] in ("SN:", "ROW:", "LINE:"):
                words = words[1:]
            elif words[0].endswith(":"):
                continue
            if len(words) != len(columns):
                raise ValueError(
                    f"FITRES row {number} has {len(words)} values for {len(columns)} columns"
                )
            rows.append(words)
    if columns is None:
        raise ValueError("FITRES table is missing its VARNAMES: header")
    result = {}
    for index, name in enumerate(columns):
        values = np.asarray([row[index] for row in rows], dtype=str)
        if name not in ("CID", "SNID", "CCID", "FIELD"):
            try:
                values = values.astype(float)
            except ValueError:
                pass
        result[name] = values
    return result
