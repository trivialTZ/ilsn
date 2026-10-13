Using the package
=================

Run examples from the repository root with ``PYTHONPATH=.``. There is no
package-index installation step. Dependencies are in ``requirements.txt``;
development dependencies and the test command are described in the README.

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

This is the saved illustrative training event: its injected magnification
is about 1.20, while its noisy brightness likelihood peaks near 1.35. The
response includes the fitting scatter; a single event need not peak at its
injection. See ``data/README.rst`` for the response domain and limitations.

Build the standardized observable
---------------------------------

.. code-block:: python

    from ilsn.supernova import read_fitres, fixed_raw_tripp, roman_quality_selection

    table = read_fitres("simulation.FITRES.gz")
    keep = roman_quality_selection(table)
    selected = {key: value[keep] for key, value in table.items()}
    y = fixed_raw_tripp(selected) - response.control_scalar

Use ``standardized_residual`` when supplying a different explicit reference
distance or coefficients, and rebuild the response for that observable.
An existing BBC-corrected ``MURES`` is not interchangeable with this raw Tripp
residual. Response-conditioning true redshifts and observed fitted redshifts
in standardization have different roles. This release does not marginalize
an uncertain redshift in the lens geometry.

Calibrate a new response
------------------------

``ResponseCalibration.from_generated`` consumes arrays for *all generated*
parents at the photon-shift nodes, including failed/deselected outcomes.
The delivered mask defines the selected moments and the generated rows define
the delivery denominator. Matched parent indices retain correlations across
photon shifts in the bootstrap. Supply the same quality/classification
selection and observable you will use for inference. Consult the API for the
row arguments and conditioning metadata. A selected-only table cannot measure
the delivery probability.

Construct source and target banks
---------------------------------

The physical construction is part of the library. For example:

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

This grid has 1,620 states;
``MassGrid.primary()`` uses 4,860 and ``MassGrid.roman()`` uses 72,900. The latter
retains core scale and redistribution freedom. Inspect ``support_invalid``,
``support_unresolved`` and the separate geometry/endpoint masks. For large
jobs, ``state_slice`` permits building independent physical blocks. The
default direct lens solver is intended for readable calculations and controls,
not the historical production solver's throughput.

``observation.paired_los`` compresses paired empirical convergence/shear pools.
``observation.source_sample`` turns a physical source bank into photon shifts,
local LOS weights, a dust template and prescribed stellar microlensing variance.
Passing ``los=None`` explicitly removes external convergence/shear; it does
not fetch a default cosmological environment. ``td_selection`` and
``imaging_log_likelihood`` construct the target weights and compressed inner
observations. Both preserve the mass-state axes.

The public inference works with NumPy arrays, including predictions from a
different lens engine. The axes are:

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
Source and target states share parameter coordinates, not identical physical
galaxies. Foreground radii, source/lens redshifts, stellar sizes and mass
proxies belong in each object's physical predictions.

At each mass state, local component weights sum to at most one. Set zero
weight on invalid or multi-image source components. Selection is applied
inside the likelihood from the calibrated delivery response; do not include
it a second time in those pre-delivery component weights. A vector of local
weights broadcasts over L. Per-source weights require shape ``(N,1,1)``.

Conditional transfer
--------------------

The central inference sequence is:

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

The array names in this snippet are inputs, not bundled global variables.
``examples/quickstart.py`` constructs every one of these ingredients from
physical lenses and a small synthetic dataset.

For correlated common/cell calibration or dust priors, construct
``SharedNuisance`` directly: each row is a joint realization. An offset array
of shape ``(Q,N)`` assigns the correlated residual shift to each source.
The independent Cartesian factory above is only one choice. Set response
index ``-1`` for an explicitly fixed point calibration.

For a large joint population/calibration rule, use
``SourceLikelihood.aligned_log_likelihood(population, eta, nuisance_index)``.
Each eta row is paired with one shared nuisance row, avoiding the full
eta-by-calibration Cartesian product. Pass those eta rows, the returned
conditional likelihoods, and their *joint* prior quadrature weights to
``population_transfer``. For importance samples, those weights include the
joint prior divided by the proposal density. Do not multiply the response
prior again. Retain the nuisance indices alongside the saved node arrays.

Streaming a joint calculation
-----------------------------

``examples/roman_transfer.py`` is a complete saved-array run of the actual
eight-coordinate model. Its central source step is:

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
A plain Gaussian eta rule combined with independently prior-sampled response
draws would instead have equal prior weights. This function does not build or
claim convergence of an adaptive proposal.

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

The Gaussian sum for ``before_nodes`` assumes independent zero-mean Gaussian
source and contrast priors. Every receiving node is common to all target
lenses; all their imaging evidence updates its weight. ``target_population``
may have a different number of states from ``source_population`` but must use
the same frozen score coordinates. Check both receiving node count and seed,
as well as the independently supplied source rule, before quoting precision.
The runnable example also calculates the before-SN distribution from the same
finite source rule with its prior/proposal weights. With an uninformative
source likelihood, that matched comparison is unchanged. Its difference from
the Gaussian-reference baseline diagnoses how poorly the coarse source rule
represents the prior; it is not a bound on integration error. The Gaussian
reference width ratio must not be interpreted as SN-only improvement when
this difference is appreciable.

Existing research calculations
------------------------------

The response loader accepts existing Roman moment archives and automatically
reads a neighboring ``.metadata.json`` when present. Retain that metadata for
fixed-redshift support. Physical banks can be supplied as the arrays above;
NPZ/CSV storage, scheduler settings and paper-specific filenames are not
coupled to the inference classes. The package does not require the historical
research repository, its raw simulation directories or its manuscript files.

The public examples reproduce the scientific operations, not every large
paper run. For paper-scale work, preserve the actual mass grid and its base
measure, all prescribed population and contrast scores, selection, calibration
draws and target likelihoods. Changing them changes the inference.

The bundled reduced calculation was compared with the original research
implementation after independently rebuilding its physical banks and
recomputing source likelihoods. Source probabilities and the full receiving
covariance agree to numerical precision. The expanded 72,900-state base measure
and material admission were also compared, with a smaller forward-model probe.
These are implementation checks; the full 1,144-source production forecast
has not been rerun by this source distribution.
