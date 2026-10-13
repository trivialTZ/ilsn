Using the package
=================

Run examples from the repository root with ``PYTHONPATH=.``. Install the
dependencies in ``requirements.txt``; see the README for development setup
and tests.

Read the simulated Roman response
---------------------------------

.. code-block:: python

    import numpy as np
    from ilsn.response import ResponseCalibration
    from ilsn.likelihood import magnification_log_likelihood

    response = ResponseCalibration.load("data/roman_response.npz")
    mu = np.linspace(0.8, 1.8, 301)
    log_likelihood = magnification_log_likelihood(
        residual_mag=-0.3513132671710686,
        redshift=1.2,
        magnification=mu,
        calibration=response,
    )

The saved training event has injected magnification about 1.20 and a noisy
brightness likelihood peaking near 1.35. The response includes fitting scatter.
See ``data/README.rst`` for its calibration domain.

Build the standardized observable
---------------------------------

.. code-block:: python

    from ilsn.supernova import read_fitres, fixed_raw_tripp, roman_quality_selection

    table = read_fitres("simulation.FITRES.gz")
    keep = roman_quality_selection(table)
    selected = {key: value[keep] for key, value in table.items()}
    y = fixed_raw_tripp(selected) - response.control_scalar

For a different reference distance or standardization coefficients, use
``standardized_residual`` and rebuild the response for that observable.
BBC-corrected ``MURES`` and the raw Tripp residual define different observables.
The response conditions on true redshift; standardization uses fitted
redshift. Lens geometry uses fixed redshifts.

Calibrate a new response
------------------------

``ResponseCalibration.from_generated`` requires all generated parents at the
photon-shift nodes, including failed/deselected outcomes. The delivered mask
defines selected moments; generated rows supply the delivery denominator.
Matched parent indices preserve bootstrap correlations across photon shifts.
Use the inference observable and quality/classification selection. The API
specifies row arguments and conditioning metadata.

Construct source and target banks
---------------------------------

.. code-block:: python

    import numpy as np
    from ilsn.mass_grid import LensDesign, MassGrid, PhysicalRelations
    from ilsn.lens_bank import build_lens_bank
    from ilsn.population import MassPopulation

    design = LensDesign(
        log_design_mass=np.log10([2e11]), re_arcsec=[0.9],
        z_lens=[0.3], z_source=[1.2], radius_arcsec=[7.0],
    )
    grid = MassGrid.primary(eta_nodes=(-0.2, 0, 0.2))
    bank = build_lens_bank(design, grid, PhysicalRelations())
    population = MassPopulation(grid.base_probability, grid.scores)

This grid has 1,620 states; ``MassGrid.primary()`` uses 4,860 and
``MassGrid.roman()`` uses 72,900, retaining core scale and redistribution
freedom. Inspect ``support_invalid``, ``support_unresolved`` and the
geometry/endpoint masks. Use ``state_slice`` to build independent physical
blocks for large jobs.

``observation.paired_los`` compresses paired empirical convergence/shear pools.
``observation.source_sample`` turns a physical source bank into photon shifts,
local LOS weights, a dust template and prescribed stellar microlensing variance.
``los=None`` sets external convergence/shear to zero. ``td_selection`` and
``imaging_log_likelihood`` construct target weights and compressed inner
observations while preserving the mass-state axes.

Inference accepts NumPy arrays, including predictions from other lens engines:

========================== ===============================================
Input                      Shape / meaning
========================== ===============================================
source residual            (N,) fixed-coordinate magnitudes
source redshift            (N,) response-conditioning redshifts
photon shift               (N, K, L), or (N, K) for one local component
component weight           (N, K, L), includes physical/geometry admission
base mass probability      (K,), prior quadrature mass
population scores          (K, D), dimensionless physical-population scores
source coordinates         (H, D), eta integration nodes
source prior probability   (H,), quadrature mass, not just a prior density
contrast coordinates       (C, D), common target-only delta nodes
contrast probability       (C,), prior quadrature mass
target endpoint            (J, K), evaluated at each target's own covariates
target selection           (J, K), probability or nonnegative caustic weight
target log likelihood      (J, K), supplied imaging/dynamics data
========================== ===============================================

N is the SN count, J the target count, K the mass-state count, L the local
nuisance count, D the population dimension, and H/C quadrature node counts.
Source and target grids share parameter coordinates. Each object's predictions
use its own foreground radius, source/lens redshifts, stellar size and mass
proxy.

At each mass state, pre-delivery local component weights sum to at most one,
with zero weight for invalid or multi-image sources. The likelihood applies
calibrated delivery selection once. A vector of local weights broadcasts
over L; per-source weights require shape ``(N,1,1)``.

Conditional transfer
--------------------

.. code-block:: python

    from ilsn.population import MassPopulation, TargetSample, population_transfer
    from ilsn.likelihood import SourceSample, SourceLikelihood, SharedNuisance

    population = MassPopulation(base_probability, scores)
    sources = SourceSample(
        y, z_source, photon_shift,
        component_weight=eligible_component_weights,
        extra_variance_mag2=additional_micro_variance,
        dust_template_mag=dust_template,
    )
    nuisance = SharedNuisance.from_calibration(
        response, dust_amplitude=[0, 0.5, 1], dust_probability=[0.25, 0.5, 0.25],
    )
    source_logl = SourceLikelihood(sources, response, nuisance).log_likelihood(
        population, eta,
    )
    targets = TargetSample(
        target_s_E, target_selection,
        log_likelihood=target_log_imaging, physical_mask=target_physical_mask,
    )
    result = population_transfer(
        population, targets, eta, source_logl, eta_probability,
        delta, delta_probability,
    )
    print(result.prior.catalogue_mean_sd, result.posterior.catalogue_mean_sd)
    covariance = result.posterior.covariance

Supply the named arrays as inputs. ``examples/quickstart.py`` constructs them
from physical lenses and a small synthetic dataset.

For correlated common/cell calibration or dust priors, construct
``SharedNuisance`` directly: each row is a joint realization. An offset array
of shape ``(Q,N)`` assigns the correlated residual shift to each source.
The factory above uses an independent Cartesian prior. Response index ``-1``
selects a fixed point calibration.

For a large joint population/calibration rule, use
``SourceLikelihood.aligned_log_likelihood(population, eta, nuisance_index)``.
Each eta row is paired with one shared nuisance row, avoiding the full
eta-by-calibration Cartesian product. Pass those eta rows, the returned
conditional likelihoods, and their *joint* prior quadrature weights to
``population_transfer``. For importance samples, those weights include the
joint prior divided by the proposal density, counting the response prior
once. Save nuisance indices alongside the node arrays.

Streaming a joint calculation
-----------------------------

``examples/roman_transfer.py`` runs the eight-coordinate model from saved
arrays. Its source step is:

.. code-block:: python

    from ilsn.inference import source_posterior

    fit = source_posterior(
        blocks, source_population, response, eta, response_index,
        log_prior_weight, dust_amplitude, dust_probability,
        continuation="unit_photon",
    )

Here ``blocks`` is a lazy iterable of ``(SourceSample, offsets)`` pairs. Each
sample contains only its source block; offsets have shape ``(H,N_block)`` or
``(H,)``. Every block uses the same eta and response-index rows. The function
accumulates all conditional log likelihoods before integrating the independent
shared dust prior. ``log_prior_weight`` is the joint prior/proposal correction
of the supplied rule, including the paired response and offset priors once.
A Gaussian eta rule with independently prior-sampled response draws has equal
prior weights. Construct the proposal and check its convergence separately.

Read out an independent receiving mass grid without constructing an eta-by-delta
Cartesian product:

.. code-block:: python

    from ilsn.sampling import gaussian_nodes, receiving_nodes
    from ilsn.population import target_readout

    before_nodes, before_weights = gaussian_nodes(
        np.zeros(eta.shape[1]), source_covariance + contrast_covariance,
        n=1024, seed=31,
    )
    after_nodes, after_weights = receiving_nodes(
        eta, fit["probability"], contrast_covariance, n=1024, seed=31,
    )
    before, _ = target_readout(target_population, targets, before_nodes, before_weights)
    after, _ = target_readout(target_population, targets, after_nodes, after_weights)

The Gaussian sum for ``before_nodes`` assumes independent, zero-mean Gaussian
source and contrast priors. Every receiving node is shared by all targets,
whose imaging evidence updates its weight. Source and target state counts
may differ, but their frozen score coordinates must agree.

Check receiving node count, seed and the supplied source rule. For SN-only
improvement, use the example's matched before-SN distribution from that same
finite source rule and prior/proposal weights: an uninformative source
likelihood leaves this comparison unchanged. A difference from the Gaussian
reference baseline diagnoses coarse prior representation, without bounding
integration error. If appreciable, it also prevents interpreting the Gaussian
reference width ratio as SN-only improvement.

Response archives and reproducibility
-------------------------------------

The loader accepts Roman moment archives and reads a neighboring
``.metadata.json`` when present. Retain it to specify fixed-redshift support.
Physical banks use the array conventions above, independently of storage
format or scheduler. Preserve the mass grid and base measure, population
and contrast scores, selection, calibration draws and target likelihoods
when reproducing a calculation.
