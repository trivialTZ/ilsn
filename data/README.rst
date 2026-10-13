Bundled simulation inputs
=========================

Roman response
--------------

``roman_response.npz`` contains the TRAIN20 calibration from simulated Roman
Type Ia SNe processed with SNANA/PIPPIN: selected means and variances, delivery
probabilities, 2,000 joint calibration draws and their probabilities, the fixed
control origin, and metadata.

The design uses 600 matched training parents at each true redshift
0.55, 0.6, 0.85, 1.2, 1.7 and 2.475. Each parent has a zero control and outcomes
on a 20-knot photon-shift grid. All attempted parents, including failed
delivery, enter the delivery denominator. The calibration population is true
Ia; standardization and quality selection use observable quantities.

The fixed raw Tripp observable is

.. code-block:: text

   R = mB + 0.1405*x1 - 3.1136*c
       - 0.010*(0.5 - expit((HOST_LOGMASS - 10)/0.01))
       - mu_reference(zHD, zHEL)
   Y = R - C0
   C0 = -19.35848721346268 mag

The reference distance is flat with Om=0.315, H0=70 km/s/Mpc, w0=-0.8 and wa=-1.
It uses observed zHD in the distance integral and observed zHEL in the
luminosity prefactor. ``ilsn.supernova.fixed_raw_tripp`` implements R without
per-batch BBC corrections. C0 is the equal-bin mean of selected zero-control
residuals and stays fixed across all calibration draws.

Response assumptions
~~~~~~~~~~~~~~~~~~~~

* The selected density is Gaussian with the calibrated mean and variance.
  Mean, variance and delivery interpolate linearly in photon shift. Evaluation
  is restricted to the six true-redshift centers, with tolerance 1e-6.
* Out-of-grid photon shifts raise an error by default. The population example
  uses ``unit_photon`` continuation: mean extends with unit photon slope;
  variance and delivery remain at their edge values. Redshift support is fixed.
* Calibration draws use positive exponential parent multipliers normalized
  within generation strata. Each multiplier is shared across the parent's
  control, injections and failed outcomes, keeping mean, variance and delivery
  paired. Variance uses the fixed-point effective-count correction.
* The draws describe finite-training uncertainty under the simulated population
  and selection. Survey zero points, standardization coefficients and simulator
  mismatch require separate constraints or priors.
* TRAIN20 refines the original 12-knot grid using the same training parents,
  after the earlier validation was examined. Independent validation of the
  refined grid and its Gaussian tails remains outstanding.

Run the individual-SN calculation::

    PYTHONPATH=. python examples/roman_magnification.py

The included training event has residual -0.3513132671710686 mag at true
redshift 1.2 and injected photon shift -0.2 mag, corresponding to
``mu = 10**0.08 = 1.2022644``. Its likelihood averages over all response draws
and peaks near 1.3514. The log-likelihood-drop-0.5 interval is approximately
[1.1668, 1.5679], with fractional half-width 0.1484. This width describes that
noisy event under the response model. The calculation fixes attenuation,
extra microlensing and post-delivery offsets to zero and applies no lens-mass
prior. Delivery cancels at fixed magnification.

Population transfer input
-------------------------

``roman_transfer.npz`` contains two simulated SN foregrounds and ten independent
strong-lens targets. It includes two residual realizations, physical lens
predictions, paired line-of-sight samples, mock imaging, mass weights/scores,
and a 64-node joint population/calibration importance rule. The example
recomputes source likelihoods and target covariance from these inputs::

    PYTHONPATH=. python examples/roman_transfer.py
    PYTHONPATH=. python examples/roman_transfer.py --realization 1 --nodes 2048
    PYTHONPATH=. python examples/roman_transfer.py --contrast-scale 2

The mass grid is ``MassGrid.primary(eta_nodes=(-0.2, 0, 0.2))``: 1,620 states
with q=5, zero compensated redistribution and the primary eight-score
definition. Predictions are built with ``build_lens_bank``.

Physical and population assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The source prior is zero-mean Gaussian with covariance
  diag(1,1,1,1,1,1,1,0.25).
* The independent target contrast is zero-mean Gaussian with covariance
  diag(0.35^2,0.35^2,0.35^2,0.35^2,0.35^2,0.35^2,0.35^2,0.5^2), shared across
  all targets.
* Shared dust amplitudes [0,0.5,1] have weights [0.25,0.5,0.25] and template
  0.01*(2*Re/radius)^2 mag. External convergence and shear retain eight paired
  simulated rows and four orientations. Additional post-delivery microlensing
  variance is 0.04*kappa_star mag squared.
* Targets use caustic-area weighting, a 150--350 km/s SIS velocity proxy,
  and compressed simulated theta_E/xi_E observations.

The 64-node rule is coarse. The example reports a Gaussian-prior reference
and a before-SN baseline using the same finite source rule as the posterior.
Use the matched baseline to isolate the SN update. The substantial difference
between the two baselines reflects the coarse prior representation, not SN
information.
Source and receiving integration need separate refinement; ``--nodes``
changes only the receiving rule. Response indices sample the 2,000 joint
calibration draws with the saved importance weights. This two-source input
is intended for exploring the calculation and its assumptions.

Array groups
~~~~~~~~~~~~

``source_*``
    Residuals, design covariates, physical predictions, mass weights/scores,
    population nodes, response indices, joint prior/proposal weights and offsets.
``target_*``
    Target design, predictions and mass weights/scores.
``los_*``
    Paired environment samples and their integration weights.
``observed_*`` and ``sigma_*``
    Fixed mock inner observations and their uncertainties.
``source_prior_covariance``, ``contrast_covariance`` and dust arrays
    Population and nuisance priors.

See ``examples/roman_transfer.py`` for array usage and
`CITATION.rst <../CITATION.rst>`_ for input sources.
