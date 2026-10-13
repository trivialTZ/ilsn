"""Selected SN response moments and jointly sampled calibration uncertainty.

The response is conditioned on delivery: its mean and variance describe the
selected sample, while delivery probability retains all generated parents in
the denominator. One calibration draw is shared by the entire source sample.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .population import normalized_probability


def _array(value):
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ResponseCalibration:
    """Selected mean, variance and delivery on a photon-shift grid.

    Grids have shape ``(redshift bin, photon-shift knot)``; joint draws add
    a leading draw axis. Mean magnitudes use the fixed numerical coordinate
    ``raw_residual - control_scalar`` for every draw. A draw's zero-point
    uncertainty belongs in its mean grid, without recentering the data.

    Redshift bins are half-open. If metadata contains
    ``redshift_conditioning = {"nodes": [...], "tolerance": ...}``, those
    fixed or narrow true-redshift centers constrain the support further.
    Otherwise each grid describes its redshift-bin population average.

    :param z_edges: Strictly increasing redshift lookup edges.
    :param shift_nodes: Strictly increasing photon shifts in magnitudes.
    :param point_mean_mag: Selected point-estimate mean magnitudes.
    :param point_variance_mag2: Selected point-estimate variances in mag squared.
    :param point_delivery_probability: Selected/all-generated probabilities.
    :param draw_mean_mag: Selected mean grids for joint calibration draws.
    :param draw_variance_mag2: Variance grids for the same joint draws.
    :param draw_delivery_probability: Delivery grids for the same joint draws.
    :param control_scalar: Fixed control origin C0 in magnitudes.
    :param draw_probability: Nonnegative weights, normalized on construction.
    :param metadata: Optional scientific conditioning and calibration notes.
    """

    z_edges: np.ndarray
    shift_nodes: np.ndarray
    point_mean_mag: np.ndarray
    point_variance_mag2: np.ndarray
    point_delivery_probability: np.ndarray
    draw_mean_mag: np.ndarray
    draw_variance_mag2: np.ndarray
    draw_delivery_probability: np.ndarray
    control_scalar: float = 0.0
    draw_probability: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        names = (
            "z_edges",
            "shift_nodes",
            "point_mean_mag",
            "point_variance_mag2",
            "point_delivery_probability",
            "draw_mean_mag",
            "draw_variance_mag2",
            "draw_delivery_probability",
        )
        for name in names:
            object.__setattr__(self, name, _array(getattr(self, name)))
        for name in ("z_edges", "shift_nodes"):
            axis = getattr(self, name)
            if (
                axis.ndim != 1
                or axis.size < 2
                or not np.all(np.isfinite(axis))
                or np.any(np.diff(axis) <= 0)
            ):
                raise ValueError(f"{name} needs at least two increasing finite values")
        shape = (len(self.z_edges) - 1, len(self.shift_nodes))
        for name in names[2:5]:
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if (
            self.draw_mean_mag.ndim != 3
            or self.draw_mean_mag.shape[0] < 1
            or self.draw_mean_mag.shape[1:] != shape
        ):
            raise ValueError(
                "Joint calibration draws must have shape (draw, zbin, knot)"
            )
        for name in names[6:]:
            if getattr(self, name).shape != self.draw_mean_mag.shape:
                raise ValueError("Mean, variance and delivery must share the draw axes")
        for mean, variance, delivery in (
            (
                self.point_mean_mag,
                self.point_variance_mag2,
                self.point_delivery_probability,
            ),
            (
                self.draw_mean_mag,
                self.draw_variance_mag2,
                self.draw_delivery_probability,
            ),
        ):
            if any(np.any(np.isinf(x)) for x in (mean, variance, delivery)):
                raise ValueError(
                    "Unmeasured cells use NaN, not infinite response moments"
                )
            if np.any(np.isfinite(mean) != np.isfinite(variance)):
                raise ValueError("Measured mean and variance support must agree")
            if np.any(np.isfinite(variance) & (variance <= 0)):
                raise ValueError("Measured selected variances must be positive")
            if np.any(np.isfinite(delivery) & ((delivery < 0) | (delivery > 1))):
                raise ValueError("Delivery probabilities must lie between zero and one")
        if not np.isfinite(self.control_scalar):
            raise ValueError("The fixed control origin must be finite")
        probability = (
            np.ones(self.n_draws)
            if self.draw_probability is None
            else np.asarray(self.draw_probability, float)
        )
        if (
            probability.shape != (self.n_draws,)
            or np.any(~np.isfinite(probability))
            or np.any(probability < 0)
            or not np.any(probability > 0)
        ):
            raise ValueError("Each joint draw needs a finite nonnegative probability")
        object.__setattr__(
            self, "draw_probability", _array(normalized_probability(probability))
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        conditioning = self.metadata.get("redshift_conditioning")
        if conditioning is not None:
            nodes = np.asarray(conditioning["nodes"], float)
            tolerance = float(conditioning["tolerance"])
            if (
                nodes.shape != (shape[0],)
                or not np.all(np.isfinite(nodes))
                or np.any(np.diff(nodes) <= 0)
                or not np.isfinite(tolerance)
                or tolerance < 0
                or np.any(nodes < self.z_edges[:-1])
                or np.any(nodes >= self.z_edges[1:])
            ):
                raise ValueError(
                    "Fixed true-redshift centers must match the lookup bins"
                )

    @property
    def n_draws(self):
        """Number of joint calibration draws.

        :return: Number of response realizations in the ensemble.
        :rtype: int
        """
        return self.draw_mean_mag.shape[0]

    def evaluate(
        self, photon_shift, z, draw=0, required_mask=None, continuation="raise"
    ):
        """Interpolate a single shared response realization.

        Linear interpolation acts on variance, rather than standard deviation.
        Redshift selects a calibrated population cell; it is not interpolated.
        Required inputs outside the measured domain or using unmeasured
        endpoints raise ``ValueError``. At an exact knot, a missing adjacent
        knot does not matter. Masked entries return mean 0, variance 1 and
        delivery 0, suitable for physical components with zero weight.

        :param photon_shift: Broadcastable absolute photon shift in magnitudes.
        :param z: Broadcastable true source redshift.
        :param draw: Joint draw index; ``None`` explicitly selects point grids.
        :param required_mask: Boolean mask identifying components with support.
        :param continuation: ``raise`` requires measured photon-shift support.
            Explicit ``unit_photon`` adds the extra photon shift to the boundary
            mean, holding its variance and delivery fixed. This is an unmeasured
            tail assumption; it never extends redshift or missing-cell support.
        :return: Arrays keyed by ``mean_mag``, ``variance_mag2`` and
            ``delivery_probability``.
        :rtype: dict
        """
        inputs = [np.asarray(photon_shift, float), np.asarray(z, float)]
        if continuation not in ("raise", "unit_photon"):
            raise ValueError("Continuation must be 'raise' or 'unit_photon'.")
        inputs.append(
            True if required_mask is None else np.asarray(required_mask, bool)
        )
        shift, redshift, required = np.broadcast_arrays(*inputs)
        required = required.astype(bool)
        bad = ~np.isfinite(shift) | ~np.isfinite(redshift)
        bins = np.searchsorted(self.z_edges, redshift, side="right") - 1
        zbad = (bins < 0) | (bins >= len(self.z_edges) - 1)
        bins = np.clip(bins, 0, len(self.z_edges) - 2)
        conditioning = self.metadata.get("redshift_conditioning")
        if conditioning is not None:
            centers = np.asarray(conditioning["nodes"])[bins]
            zbad |= np.abs(redshift - centers) > conditioning["tolerance"]
        if np.any(required & (bad | zbad)):
            raise ValueError(
                "Required true redshift is outside measured calibration support, or input is nonfinite"
            )
        if continuation == "raise" and np.any(
            required & ((shift < self.shift_nodes[0]) | (shift > self.shift_nodes[-1]))
        ):
            raise ValueError(
                "Required photon shift is outside measured calibration knots"
            )
        safe_shift = np.clip(
            np.where(np.isfinite(shift), shift, self.shift_nodes[0]),
            self.shift_nodes[0],
            self.shift_nodes[-1],
        )
        hi = np.clip(
            np.searchsorted(self.shift_nodes, safe_shift, side="right"),
            1,
            len(self.shift_nodes) - 1,
        )
        lo = hi - 1
        fraction = (safe_shift - self.shift_nodes[lo]) / (
            self.shift_nodes[hi] - self.shift_nodes[lo]
        )
        if draw is None:
            grids = (
                self.point_mean_mag,
                self.point_variance_mag2,
                self.point_delivery_probability,
            )
        else:
            if not isinstance(draw, (int, np.integer)) or not 0 <= draw < self.n_draws:
                raise IndexError(
                    "Joint calibration draw index is outside the saved ensemble"
                )
            grids = (
                self.draw_mean_mag[draw],
                self.draw_variance_mag2[draw],
                self.draw_delivery_probability[draw],
            )
        result = {}
        for name, grid, fallback in zip(
            ("mean_mag", "variance_mag2", "delivery_probability"),
            grids,
            (0.0, 1.0, 0.0),
        ):
            low, high = grid[bins, lo], grid[bins, hi]
            low_needed, high_needed = fraction < 1, fraction > 0
            missing = (low_needed & ~np.isfinite(low)) | (
                high_needed & ~np.isfinite(high)
            )
            if np.any(required & missing):
                raise ValueError(
                    "Required response uses an unmeasured selected calibration cell"
                )
            interpolated = (1 - fraction) * np.where(
                low_needed, low, 0.0
            ) + fraction * np.where(high_needed, high, 0.0)
            result[name] = np.where(required, interpolated, fallback)
        if continuation == "unit_photon":
            result["mean_mag"] += np.where(required, shift - safe_shift, 0.0)
        return result

    def save(self, path):
        """Save scientific arrays and metadata in one NPZ archive.

        :param path: Output NPZ path; a missing suffix is appended.
        :return: The saved path.
        :rtype: pathlib.Path
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = Path(str(path) + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "metadata"
        }
        arrays["metadata_json"] = json.dumps(self.metadata, allow_nan=False)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path, metadata_path=None):
        """Read a public archive or an existing saved Roman calibration.

        Existing archives use the same numerical array names. Additional
        empirical samples and bookkeeping arrays are unnecessary for moment
        evaluation and are ignored. Metadata is read from the archive and,
        if present, the adjacent ``.metadata.json`` file. An explicit metadata
        path takes precedence. Retaining fixed true-redshift metadata is
        necessary to interpret a fixed-redshift calibration correctly.

        :param path: Explicit NPZ calibration path, or its suffix-free prefix.
        :param metadata_path: Optional explicit JSON metadata path.
        :return: Selected response with joint draw probabilities and fixed C0.
        :rtype: ResponseCalibration
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = Path(str(path) + ".npz")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: archive[name]
                for name in cls.__dataclass_fields__
                if name != "metadata" and name in archive.files
            }
            metadata = (
                json.loads(str(archive["metadata_json"].item()))
                if "metadata_json" in archive.files
                else {}
            )
        sidecar = (
            path.with_suffix(".metadata.json")
            if metadata_path is None
            else Path(metadata_path)
        )
        if metadata_path is not None or sidecar.exists():
            metadata.update(json.loads(sidecar.read_text()))
        return cls(**arrays, metadata=metadata)

    @classmethod
    def gaussian(
        cls,
        z_edges,
        shift_nodes,
        scatter_mag=0.1,
        *,
        delivery_probability=1.0,
        zero_point_sigma=0.0,
        n_draws=32,
        rng=None,
        control_scalar=0.0,
        redshift_nodes=None,
        redshift_tolerance=1e-6,
    ):
        """Construct an explicit Gaussian demonstration response.

        The selected mean equals the photon shift. Each draw adds one common
        Gaussian zero point to all cells. This prescribed model does not
        infer survey selection from magnification; supply a measured response
        or explicit delivery grid when that coupling matters.

        :param z_edges: Redshift lookup edges.
        :param shift_nodes: Photon-shift knots in magnitudes.
        :param scatter_mag: Selected scatter, broadcastable to the grid.
        :param delivery_probability: Delivery probability grid or scalar.
        :param zero_point_sigma: Standard deviation of the common zero point.
        :param n_draws: Number of shared zero-point draws.
        :param rng: NumPy generator or seed.
        :param control_scalar: Fixed numerical raw-control origin.
        :param redshift_nodes: Optional fixed true-redshift centers.
        :param redshift_tolerance: Allowed distance from a declared center.
        :return: A synthetic response calibration.
        :rtype: ResponseCalibration
        """
        if n_draws < 1 or not isinstance(n_draws, (int, np.integer)):
            raise ValueError("At least one integer calibration draw is required")
        if not np.isfinite(zero_point_sigma) or zero_point_sigma < 0:
            raise ValueError("Zero-point uncertainty must be finite and nonnegative")
        shape = (len(z_edges) - 1, len(shift_nodes))
        scatter = np.broadcast_to(np.asarray(scatter_mag, float), shape)
        if np.any(~np.isfinite(scatter) | (scatter <= 0)):
            raise ValueError("Selected Gaussian scatter must be positive and finite")
        mean = np.broadcast_to(np.asarray(shift_nodes, float), shape)
        variance = scatter**2
        delivery = np.broadcast_to(np.asarray(delivery_probability, float), shape)
        if not np.all(np.isfinite(delivery)):
            raise ValueError("Demonstration delivery probabilities must be finite")
        offsets = np.random.default_rng(rng).normal(0.0, zero_point_sigma, n_draws)
        metadata = {
            "model": "Prescribed Gaussian selected response with common zero point"
        }
        if redshift_nodes is not None:
            metadata["redshift_conditioning"] = {
                "nodes": np.asarray(redshift_nodes, float).tolist(),
                "tolerance": float(redshift_tolerance),
            }
        return cls(
            z_edges,
            shift_nodes,
            mean,
            variance,
            delivery,
            mean[None] + offsets[:, None, None],
            np.broadcast_to(variance, (n_draws,) + shape),
            np.broadcast_to(delivery, (n_draws,) + shape),
            control_scalar=control_scalar,
            metadata=metadata,
        )

    @classmethod
    def from_generated(
        cls,
        photon_shift,
        z,
        selected,
        raw_residual,
        parent_index,
        *,
        z_edges,
        shift_nodes,
        n_draws=200,
        rng=None,
        parent_stratum=None,
        population_weight=None,
        redshift_nodes=None,
        redshift_tolerance=1e-6,
    ):
        """Calibrate from generated outcomes with matched-parent multipliers.

        Supply every generated outcome, including failed selections with NaN
        residuals. A parent's zero control and injected outcomes share one
        multiplier in each draw, preserving covariance across response knots,
        selected moments, and delivery. Point variance uses the effective-count
        correction; draws use the same fixed correction. C0 is the equal-bin
        mean of selected zero controls and remains fixed for every draw.

        :param photon_shift: One nominal calibration shift per generated row.
        :param z: One true source redshift per generated row.
        :param selected: Boolean delivery status for every generated row.
        :param raw_residual: Raw selected residual, NaN permitted for failures.
        :param parent_index: Integer label shared by all outcomes of one parent.
        :param z_edges: Declared true-redshift lookup edges.
        :param shift_nodes: Declared knots, including exactly one zero control.
        :param n_draws: Number of matched-parent multiplier draws.
        :param rng: NumPy generator or seed.
        :param parent_stratum: Optional integer stratum per generated row.
        :param population_weight: Optional generated-parent weight per row.
        :param redshift_nodes: Optional fixed true-redshift centers.
        :param redshift_tolerance: Allowed distance from a declared center.
        :return: Calibration preserving finite-parent joint uncertainty.
        :rtype: ResponseCalibration
        """
        shift, redshift, residual = (
            np.asarray(x, float) for x in (photon_shift, z, raw_residual)
        )
        selected = np.asarray(selected, bool)
        parent_labels = np.asarray(parent_index)
        if (
            shift.ndim != 1
            or not len(shift)
            or any(
                x.shape != shift.shape
                for x in (redshift, residual, selected, parent_labels)
            )
        ):
            raise ValueError("Generated outcome arrays must be aligned and nonempty")
        if not np.issubdtype(parent_labels.dtype, np.integer):
            raise ValueError("Matched parent labels must be integers")
        if not isinstance(n_draws, (int, np.integer)) or n_draws < 1:
            raise ValueError("At least one integer multiplier draw is required")
        edges, knots = np.asarray(z_edges, float), np.asarray(shift_nodes, float)
        if (
            edges.ndim != 1
            or knots.ndim != 1
            or min(len(edges), len(knots)) < 2
            or not np.all(np.isfinite(edges))
            or not np.all(np.isfinite(knots))
            or np.any(np.diff(edges) <= 0)
            or np.any(np.diff(knots) <= 0)
        ):
            raise ValueError(
                "Increasing finite redshift edges and shift knots are required"
            )
        zero = np.flatnonzero(knots == 0)
        if len(zero) != 1:
            raise ValueError("One nominal zero-control knot is required")
        if np.any(~np.isfinite(shift) | ~np.isfinite(redshift)):
            raise ValueError("Generated shifts and true redshifts must be finite")
        bins = np.searchsorted(edges, redshift, side="right") - 1
        nodes = np.searchsorted(knots, shift)
        if (
            np.any((bins < 0) | (bins >= len(edges) - 1))
            or np.any(nodes >= len(knots))
            or not np.array_equal(knots[nodes], shift)
        ):
            raise ValueError(
                "Generated rows must use declared redshift cells and exact shift knots"
            )
        parents, first, parent = np.unique(
            parent_labels, return_index=True, return_inverse=True
        )
        strata = bins if parent_stratum is None else np.asarray(parent_stratum)
        weights = (
            np.ones(len(shift))
            if population_weight is None
            else np.asarray(population_weight, float)
        )
        if (
            strata.shape != shift.shape
            or weights.shape != shift.shape
            or np.any(~np.isfinite(weights) | (weights < 0))
        ):
            raise ValueError(
                "Generated strata and nonnegative population weights must align"
            )
        if not np.issubdtype(strata.dtype, np.integer):
            raise ValueError("Generated parent strata must be integer labels")
        for values in (redshift, strata, weights):
            if not np.array_equal(values, values[first][parent]):
                raise ValueError(
                    "Matched outcomes must share true redshift, stratum and parent weight"
                )
        identity = parent * len(knots) + nodes
        if len(np.unique(identity)) != len(shift):
            raise ValueError("Each parent has at most one generated outcome at a knot")
        if len(np.unique(parent[nodes == zero[0]])) != len(parents):
            raise ValueError("Every generated parent needs its matched zero control")
        if np.any(selected & ~np.isfinite(residual)):
            raise ValueError("Every delivered outcome needs a finite observed residual")
        shape = (len(edges) - 1, len(knots))
        cells = bins * len(knots) + nodes
        size = np.prod(shape)
        selected_cells, selected_parent = cells[selected], parent[selected]
        base = weights[first]
        parent_strata = strata[first]

        def moments(parent_weight, origin):
            weight = parent_weight[selected_parent]
            denominator = np.bincount(
                cells, weights=parent_weight[parent], minlength=size
            )
            count = np.bincount(selected_cells, weights=weight, minlength=size)
            total = np.bincount(
                selected_cells,
                weights=weight * (residual[selected] - origin),
                minlength=size,
            )
            second = np.bincount(
                selected_cells,
                weights=weight * (residual[selected] - origin) ** 2,
                minlength=size,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = total / count
                variance = np.maximum(second / count - mean**2, 0.0)
                delivery = count / denominator
            return (x.reshape(shape) for x in (mean, variance, delivery, count))

        raw_mean, _, _, _ = moments(base, 0.0)
        if not np.all(np.isfinite(raw_mean[:, zero[0]])):
            raise ValueError("Each redshift cell needs delivered zero controls for C0")
        control = float(raw_mean[:, zero[0]].mean())
        mean, variance, delivery, count = moments(base, control)
        sum_square = np.bincount(
            selected_cells, weights=base[selected_parent] ** 2, minlength=size
        ).reshape(shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            effective_count = count**2 / sum_square
            correction = effective_count / (effective_count - 1)
        support = (effective_count > 1) & np.isfinite(variance) & (variance > 0)
        point_mean = np.where(support, mean, np.nan)
        point_variance = np.where(support, variance * correction, np.nan)
        draw_mean, draw_variance, draw_delivery = (
            np.empty((n_draws,) + shape) for _ in range(3)
        )
        generator = np.random.default_rng(rng)
        for draw in range(n_draws):
            multiplier = generator.exponential(size=len(parents))
            for stratum in np.unique(parent_strata):
                at = parent_strata == stratum
                multiplier[at] *= at.sum() / multiplier[at].sum()
            m, v, s, _ = moments(base * multiplier, control)
            measured = support & np.isfinite(v) & (v > 0)
            draw_mean[draw] = np.where(measured, m, np.nan)
            draw_variance[draw] = np.where(measured, v * correction, np.nan)
            draw_delivery[draw] = s
        metadata = {
            "control_rule": "Equal-bin selected zero mean; fixed C0 for all draws",
            "uncertainty": "Finite matched-parent sampling under the supplied generation and selection law",
            "variance_rule": "Population variance times fixed point effective-count correction",
        }
        if redshift_nodes is not None:
            centers = np.asarray(redshift_nodes, float)
            if centers.shape != (shape[0],):
                raise ValueError(
                    "Fixed true-redshift centers must match the lookup bins"
                )
            if np.any(np.abs(redshift - centers[bins]) > redshift_tolerance):
                raise ValueError(
                    "Generated redshifts differ from declared calibration centers"
                )
            metadata["redshift_conditioning"] = {
                "nodes": centers.tolist(),
                "tolerance": float(redshift_tolerance),
            }
        return cls(
            edges,
            knots,
            point_mean,
            point_variance,
            delivery,
            draw_mean,
            draw_variance,
            draw_delivery,
            control,
            metadata=metadata,
        )
