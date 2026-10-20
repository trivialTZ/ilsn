"""Selected SN brightness likelihoods with survey-shared calibration."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .population import log_probability, normalized_probability
from .supernova import magnification_to_shift


class SourceSample:
    """SN residuals and physical predictions on a mass-state grid.

    Components describe object-local line-of-sight or other nuisance states.
    Their weights include physical and single-image admission, before SN
    delivery. A zero weight excludes that state/component. We condition on
    the supplied source covariates and sample size.

    :param residual_mag: Observed standardized residuals, shape (source,).
    :param redshift: Response-conditioning redshifts, shape (source,).
        Use true/design redshifts for a calibration built at true redshift.
    :param photon_shift_mag: Magnification/attenuation shifts before delivery,
        shape (source, mass state, component). A 2-D array adds one component.
    :param component_weight: Nonnegative pre-delivery component weights,
        broadcast to photon shifts; their sum per state must not exceed one.
        Per-source weights need shape (source, 1, 1); a plain vector broadcasts
        over local components. A (source, state) array adds one component axis.
    :param extra_variance_mag2: Additional *post-delivery* variance, broadcast
        to photon shifts. The calibrated SN scatter is already included.
    :param dust_template_mag: Photon attenuation per unit shared dust amplitude.
    :param foreground_log_likelihood: Log likelihood of independent data on
        each source's foreground at every mass state, such as a stellar
        dispersion, broadcast to (source, state). It multiplies the selected
        SN density but not its normalization, since selection acts on SN
        delivery alone. The default zero adds no foreground data.
    """

    def __init__(
        self,
        residual_mag,
        redshift,
        photon_shift_mag,
        component_weight=1.0,
        extra_variance_mag2=0.0,
        dust_template_mag=0.0,
        foreground_log_likelihood=0.0,
    ):
        self.residual_mag = np.asarray(residual_mag, dtype=float)
        self.redshift = np.asarray(redshift, dtype=float)
        shift = np.asarray(photon_shift_mag, dtype=float)
        if shift.ndim == 2:
            shift = shift[..., None]
        if (
            shift.ndim != 3
            or min(shift.shape) < 1
            or self.residual_mag.shape != (len(shift),)
            or self.redshift.shape != self.residual_mag.shape
        ):
            raise ValueError(
                "Observations and (source, state, component) predictions must align."
            )
        self.photon_shift_mag = shift
        for name, value in (
            ("component_weight", component_weight),
            ("extra_variance_mag2", extra_variance_mag2),
            ("dust_template_mag", dust_template_mag),
        ):
            value = np.asarray(value, dtype=float)
            if value.ndim == 2 and value.shape == shift.shape[:2]:
                value = value[..., None]
            setattr(self, name, np.broadcast_to(value, shift.shape))
        w = self.component_weight
        supported = w > 0
        if (
            not np.isfinite(w).all()
            or np.any(w < 0)
            or np.any(w.sum(axis=-1) > 1 + 1e-12)
        ):
            raise ValueError(
                "Local component weights must be nonnegative with sum at most one."
            )
        if (
            not np.isfinite(self.residual_mag).all()
            or not np.isfinite(self.redshift).all()
        ):
            raise ValueError("Residuals and conditioning redshifts must be finite.")
        if np.any(
            supported
            & (
                ~np.isfinite(shift)
                | ~np.isfinite(self.extra_variance_mag2)
                | (self.extra_variance_mag2 < 0)
                | ~np.isfinite(self.dust_template_mag)
            )
        ):
            raise ValueError(
                "Supported states need finite shifts/templates and nonnegative extra variance."
            )
        foreground = np.asarray(foreground_log_likelihood, dtype=float)
        try:
            foreground = np.broadcast_to(foreground, shift.shape[:2])
        except ValueError:
            raise ValueError(
                "Foreground log likelihoods must broadcast to (source, state)."
            ) from None
        if np.any(np.isnan(foreground) | np.isposinf(foreground)):
            raise ValueError("Foreground log likelihoods must be below +inf.")
        self.foreground_log_likelihood = foreground

    @property
    def n_states(self):
        """Number of mass states shared with the population model."""
        return self.photon_shift_mag.shape[1]

    def with_residuals(self, residual_mag):
        """Return the same physical design with a new residual realization."""
        return SourceSample(
            residual_mag,
            self.redshift,
            self.photon_shift_mag,
            self.component_weight,
            self.extra_variance_mag2,
            self.dust_template_mag,
            self.foreground_log_likelihood,
        )


@dataclass
class SharedNuisance:
    """Joint quadrature nodes held common across the source catalogue.

    Each row couples one whole response curve, a dust amplitude and a residual
    offset. Supply correlated rows directly when the nuisance prior is joint.
    Offsets have shape (node,) for a common zero point, or (node, source) for
    common-plus-cell calibration realizations. They are added after delivery.

    :param response_index: Calibration draw indices; -1 means the point curve.
    :param probability: Prior mass per joint node.
    :param dust_amplitude: Nonnegative attenuation amplitude per node.
    :param residual_offset_mag: Post-delivery residual offsets in magnitudes.
    """

    response_index: np.ndarray
    probability: np.ndarray
    dust_amplitude: np.ndarray = 0.0
    residual_offset_mag: np.ndarray = 0.0

    def __post_init__(self):
        indices = np.asarray(self.response_index)
        if (
            indices.ndim != 1
            or np.any(indices != indices.astype(int))
            or np.any(indices < -1)
        ):
            raise ValueError(
                "Response indices must be an integer vector, with -1 for the point curve."
            )
        self.response_index = indices.astype(int)
        self.probability = normalized_probability(self.probability)
        n = len(indices)
        self.dust_amplitude = np.broadcast_to(
            np.asarray(self.dust_amplitude, dtype=float), (n,)
        )
        offsets = np.asarray(self.residual_offset_mag, dtype=float)
        self.residual_offset_mag = (
            np.broadcast_to(offsets, (n,)) if offsets.ndim < 2 else offsets
        )
        if (
            len(self.probability) != n
            or self.residual_offset_mag.shape[0] != n
            or self.residual_offset_mag.ndim > 2
            or not np.isfinite(self.dust_amplitude).all()
            or np.any(self.dust_amplitude < 0)
            or not np.isfinite(self.residual_offset_mag).all()
        ):
            raise ValueError(
                "Shared nuisance values and probabilities must align and be finite."
            )

    @classmethod
    def from_calibration(
        cls, calibration, dust_amplitude=(0.0,), dust_probability=(1.0,)
    ):
        """Product of the saved response prior and an independent dust prior."""
        dust = np.asarray(dust_amplitude, dtype=float)
        pd = normalized_probability(dust_probability)
        if dust.shape != pd.shape:
            raise ValueError("Dust amplitudes and weights must align.")
        return cls(
            np.repeat(np.arange(calibration.n_draws), len(dust)),
            (calibration.draw_probability[:, None] * pd[None, :]).ravel(),
            np.tile(dust, calibration.n_draws),
        )


class SourceLikelihood:
    """Direct log-space integration of a selected ilSN survey.

    At each common response node psi, object i contributes
    ``sum(w * F * component_weight * S * K) / sum(w * component_weight * S)``,
    where F is the sample's foreground data likelihood at each state.
    The full product over objects is formed before integrating psi. Both
    delivery S and selected residual density K use the same photon shift.

    :param continuation: Response behavior beyond measured photon-shift knots.
        ``'raise'`` requires measured support; ``'unit_photon'`` explicitly
        adopts the response calibration's unit photon continuation.
    """

    def __init__(self, sample, calibration, nuisance=None, *, continuation="raise"):
        self.sample = sample
        self.calibration = calibration
        self.continuation = continuation
        self.nuisance = (
            SharedNuisance.from_calibration(calibration)
            if nuisance is None
            else nuisance
        )
        if np.any(self.nuisance.response_index >= calibration.n_draws):
            raise ValueError("A shared node refers to a missing response draw.")
        offsets = self.nuisance.residual_offset_mag
        if offsets.ndim == 2 and offsets.shape[1] != len(sample.residual_mag):
            raise ValueError(
                "Per-source shared offsets must have one column per source."
            )

    def component_response(self, source, node, state_support=None):
        """Evaluate selected moments and delivery weights for one source/node.

        :return: Mean, variance, and log delivered component mass, each with
            shape (mass state, local component).
        """
        sample, nuisance = self.sample, self.nuisance
        weight = sample.component_weight[source]
        required = weight > 0
        if state_support is not None:
            required = required & np.asarray(state_support, dtype=bool)[:, None]
        shift = (
            sample.photon_shift_mag[source]
            + nuisance.dust_amplitude[node] * sample.dust_template_mag[source]
        )
        draw = int(nuisance.response_index[node])
        response = self.calibration.evaluate(
            shift,
            sample.redshift[source],
            draw=None if draw == -1 else draw,
            required_mask=required,
            continuation=self.continuation,
        )
        offset = nuisance.residual_offset_mag[node]
        if np.ndim(offset):
            offset = offset[source]
        mean = np.where(required, response["mean_mag"] + offset, 0.0)
        variance = np.where(
            required,
            response["variance_mag2"] + sample.extra_variance_mag2[source],
            1.0,
        )
        logmass = log_probability(weight) + log_probability(
            response["delivery_probability"]
        )
        return mean, variance, np.where(required, logmass, -np.inf)

    def state_terms(self, source, node, residual_mag=None, state_support=None):
        """Log SK and log S, integrated over local components at each state."""
        y = self.sample.residual_mag[source] if residual_mag is None else residual_mag
        mean, variance, logmass = self.component_response(source, node, state_support)
        logkernel = -0.5 * ((y - mean) ** 2 / variance + np.log(2 * np.pi * variance))
        return logsumexp(logmass + logkernel, axis=-1), logsumexp(logmass, axis=-1)

    def _selected_log_likelihood(self, logw, node, state_support):
        """Sum the selected source densities at one common nuisance node."""
        result = np.zeros(len(logw))
        for source in range(len(self.sample.residual_mag)):
            logsk, logs = self.state_terms(source, node, state_support=state_support)
            normalization = logsumexp(logw + logs[None, :], axis=1)
            if not np.isfinite(normalization).all():
                raise ValueError("A source has no deliverable support at this node.")
            logsk = logsk + self.sample.foreground_log_likelihood[source]
            result += logsumexp(logw + logsk[None, :], axis=1) - normalization
        return result

    def conditional_log_likelihood(self, population, coordinates, chunk_size=256):
        """Survey log likelihood at each eta and shared nuisance node.

        :return: Array of shape (population node, shared nuisance node).
        """
        coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))
        if self.sample.n_states != len(population.base_probability) or chunk_size < 1:
            raise ValueError(
                "Source/population mass states must align; chunk size must be positive."
            )
        result = np.zeros((len(coordinates), len(self.nuisance.probability)))
        state_support = population.base_probability > 0
        for start in range(0, len(coordinates), chunk_size):
            stop = min(start + chunk_size, len(coordinates))
            logw = population.log_probabilities(coordinates[start:stop])
            for q, pq in enumerate(self.nuisance.probability):
                if pq == 0:
                    continue
                result[start:stop, q] = self._selected_log_likelihood(
                    logw, q, state_support
                )
        return result

    def log_likelihood(self, population, coordinates, chunk_size=256):
        """Integrate common calibration after the product of source densities."""
        conditional = self.conditional_log_likelihood(
            population, coordinates, chunk_size
        )
        return logsumexp(
            conditional + log_probability(self.nuisance.probability)[None, :], axis=1
        )

    def aligned_log_likelihood(
        self, population, coordinates, nuisance_index, chunk_size=256
    ):
        """Evaluate paired population/calibration nodes without a Cartesian grid.

        This supports a joint importance or quadrature rule for (eta, psi).
        Each row's psi remains common to all sources. The returned values are
        conditional likelihoods: apply the joint prior/proposal weights once,
        after the survey product, in the subsequent population inference.

        :param population: MassPopulation on the source mass states.
        :param coordinates: Eta coordinates, shape (joint node, parameter).
        :param nuisance_index: Row of this likelihood's SharedNuisance for
            each joint node, shape (joint node,).
        :param chunk_size: Maximum number of population rows per batch.
        :return: Survey log likelihood per paired node, in input order.
        """
        coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))
        indices = np.asarray(nuisance_index)
        if (
            indices.shape != (len(coordinates),)
            or np.any(indices != indices.astype(int))
            or np.any(indices < 0)
            or np.any(indices >= len(self.nuisance.probability))
            or self.sample.n_states != len(population.base_probability)
            or chunk_size < 1
        ):
            raise ValueError("Paired population and nuisance nodes must align.")
        indices = indices.astype(int)
        result = np.zeros(len(coordinates))
        support = population.base_probability > 0
        for q in np.unique(indices):
            rows = np.flatnonzero(indices == q)
            for start in range(0, len(rows), chunk_size):
                chosen = rows[start : start + chunk_size]
                logw = population.log_probabilities(coordinates[chosen])
                result[chosen] = self._selected_log_likelihood(logw, q, support)
        return result

    def simulate(self, population, coordinates, rng, nuisance_index=None):
        """Draw a fixed-size delivered sample with one common calibration node.

        :param coordinates: One source-population parameter vector.
        :param rng: A NumPy Generator.
        :param nuisance_index: Fix a common node, or draw it from its prior.
        :return: New :class:`SourceSample` and a dictionary of injected states.
        """
        logw = population.log_probabilities(coordinates)
        if len(logw) != 1:
            raise ValueError("Simulation takes one population realization.")
        q = (
            int(rng.choice(len(self.nuisance.probability), p=self.nuisance.probability))
            if nuisance_index is None
            else int(nuisance_index)
        )
        if q < 0 or q >= len(self.nuisance.probability):
            raise ValueError("Missing shared nuisance node.")
        y, states, components = [], [], []
        for source in range(len(self.sample.residual_mag)):
            mean, variance, logmass = self.component_response(
                source, q, population.base_probability > 0
            )
            joint = logw[0, :, None] + logmass
            normalization = logsumexp(joint)
            if not np.isfinite(normalization):
                raise ValueError(
                    "Cannot simulate a source with zero delivery probability."
                )
            index = rng.choice(joint.size, p=np.exp(joint.ravel() - normalization))
            state, component = np.unravel_index(index, joint.shape)
            y.append(
                rng.normal(mean[state, component], np.sqrt(variance[state, component]))
            )
            states.append(state)
            components.append(component)
        return self.sample.with_residuals(y), dict(
            nuisance_index=q,
            mass_state=np.array(states),
            component=np.array(components),
        )


def magnification_log_likelihood(residual_mag, redshift, magnification, calibration):
    """Conditional brightness likelihood at fixed magnification and redshift.

    No lens-mass prior or magnification Jacobian is applied. Delivery cancels
    at each fixed physical state. Attenuation, additional microlensing and
    extra calibration offsets are fixed to zero; use SourceLikelihood for
    their explicit integration.
    """
    shift = magnification_to_shift(magnification)
    terms = []
    for q, probability in enumerate(calibration.draw_probability):
        if probability == 0:
            continue
        response = calibration.evaluate(shift, redshift, draw=q)
        if np.any(response["delivery_probability"] <= 0):
            raise ValueError("Fixed magnification has no delivered SN support.")
        variance = response["variance_mag2"]
        terms.append(
            np.log(probability)
            - 0.5
            * (
                (residual_mag - response["mean_mag"]) ** 2 / variance
                + np.log(2 * np.pi * variance)
            )
        )
    return logsumexp(np.stack(terms), axis=0)
