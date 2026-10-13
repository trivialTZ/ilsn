Roman selected-response calibration
===================================

``roman_response.npz`` is a compact copy of the TRAIN20 selected-response
calibration from simulated Roman Type Ia SNe processed with the SNANA/PIPPIN
workflow. It contains point selected means, selected variances, delivery
probabilities, all 2000 joint calibration draws, their prior probabilities,
the fixed control origin, and scientific metadata. It contains no individual
parent records. The original numerical arrays are retained without rounding.

The generation design used 600 matched training parents at each of six true
redshifts: 0.55, 0.6, 0.85, 1.2, 1.7 and 2.475. Each parent has a zero control
and photon-shift outcomes on a full 20-knot grid. All attempted parents,
including failed delivery, enter the delivery denominator. The simulated
calibration population is true Ia; truth-type labels do not enter observable
standardization or light-curve quality selection.

The standardized observable is

.. code-block:: text

   R = mB + 0.1405*x1 - 3.1136*c
       - 0.010*(0.5 - expit((HOST_LOGMASS - 10)/0.01))
       - mu_reference(zHD, zHEL)
   Y = R - C0
   C0 = -19.35848721346268 mag

This is a fixed raw Tripp observable, with no per-batch BBC residual or bias
correction. The reference distance uses a flat model with Om=0.315, H0=70
km/s/Mpc, w0=-0.8 and wa=-1; the distance integral uses observed zHD and the
luminosity prefactor uses observed zHEL. ``ilsn.supernova.fixed_raw_tripp``
implements this observable. C0 is the equal-bin mean of selected zero-control
raw residuals, and every draw remains in that same fixed numerical coordinate.

The selected Gaussian density approximates each conditional residual
distribution using its own selected mean and variance. Absolute mean,
variance and delivery interpolate linearly between photon-shift knots.
Broad lookup edges do not establish continuous redshift support: evaluation
is restricted to the six declared true-redshift centers within 1e-6. Photon
shifts beyond the grid raise an error by default. ``unit_photon`` continuation
is an explicit alternative used in the Roman population calculation: the
edge mean is extended with unit photon slope and its variance/delivery are
held fixed. This tail law is an assumption and does not extend redshift support.

The 2000 response draws use positive exponential matched-parent multipliers,
normalized within the declared generation strata. A parent has one multiplier
shared across its zero control, all injected nodes and failed outcomes. The
same weights determine mean, variance and delivery together. Variance uses
the fixed point effective-count correction. These draws approximate finite
simulated-training sampling uncertainty conditional on the supplied simulator,
selection and finite parent allocation. They do not measure real survey
zero-point, standardization-coefficient or population-model systematics, or
establish unconditional simulator/population portability.

The 20-knot table is a training-grid refinement made after the original
12-knot validation was opened. It uses already generated training parents
under the same observable and selection law. The refinement has no new
independent validation or coverage claim. Matching selected moments also does
not establish Gaussian tail or coverage accuracy.

Run the individual-SN illustration from the public repository root:

.. code-block:: bash

   PYTHONPATH=. python examples/roman_magnification.py

The included residual, -0.3513132671710686 mag at true redshift 1.2, is one
training event injected with photon shift -0.2 mag, or magnification
``10**0.08 = 1.2022644``. Its conditional brightness likelihood averages the
Gaussian selected density over all response draws. The likelihood peak is
approximately 1.3514; the interval with log-likelihood drop at most 0.5 is
approximately [1.1668, 1.5679], with fractional half-width 0.1484. Delivery
cancels at each fixed magnification. Attenuation, extra microlensing and
post-delivery offsets are fixed to zero. No lens-mass prior or magnification
Jacobian is applied. This single training example is neither a coverage test
nor a universal independent SN error or a survey-mean precision forecast.

Reduced Roman population input
==============================

``roman_transfer.npz`` contains two simulated SN foregrounds and ten independent
strong-lens targets from an existing reduced Roman workflow calculation. It
includes both residual realizations, physical source/target banks, paired LOS
quadrature, mock imaging observations, mass weights/scores, and a 64-node joint
source/calibration importance rule. It contains no fitted posterior weights.
The public example recalculates all source evidence before target readout::

    PYTHONPATH=. python examples/roman_transfer.py
    PYTHONPATH=. python examples/roman_transfer.py --realization 1 --nodes 2048
    PYTHONPATH=. python examples/roman_transfer.py --contrast-scale 2

The physical input is ``MassGrid.primary(eta_nodes=(-0.2, 0, 0.2))``: 1,620
states with q=5 and no compensated redistribution, evaluated using the primary
eight-score definition. Source/target fields were rebuilt through the public
``build_lens_bank`` functions. The 64 joint nodes retain the existing importance
correction; they are a coarse integration rule, not evidence of convergence.
An increased receiving node count refines only the receiving integration.
The runner reports both a Gaussian-prior baseline and a baseline obtained
from the same finite source rule before applying SN data. Only the latter's
width ratio isolates the SN update on that finite rule. The no-SN difference
between these baselines is substantial in this 64-node input; it diagnoses
coarse integration and must not be counted as ilSN information. Refining
the receiving rule alone does not repair the source-prior representation.
The response has 2,000 possible joint draws; this importance rule samples them
through its saved response indices and correct joint weights.

The source prior is zero-mean Gaussian with covariance diag(1,1,1,1,1,1,1,0.25).
The independent target contrast is zero-mean Gaussian with covariance
diag(0.35^2,0.35^2,0.35^2,0.35^2,0.35^2,0.35^2,0.35^2,0.5^2).
Shared dust amplitudes [0,0.5,1] have weights [0.25,0.5,0.25]; their template
is 0.01*(2*Re/radius)^2 mag. External convergence and shear retain eight paired
empirical rows and four angles. Additional post-delivery microlensing variance
is 0.04*kappa_star. Targets use caustic-area weighting with the 150--350 km/s
SIS velocity proxy and compressed simulated theta_E/xi_E observations.

Arrays named ``source_*`` contain the source residuals, fixed design covariates,
physical predictions, base measure, scores, eta nodes, response indices, joint
log prior/proposal weights and node-by-source offsets. ``target_*`` contains
the receiving design, its own physical predictions and base measure/scores.
``los_*`` contains the common paired environment rule; ``observed_*`` and
``sigma_*`` are fixed inner observations and errors. ``source_prior_covariance``,
``contrast_covariance`` and the dust arrays make the population priors explicit.
The full array usage is in ``examples/roman_transfer.py``.

This input serves as a compact executable scientific calculation. Two sources
do not establish the full Roman sample's precision. A realized posterior can
widen, and a narrow interval alone says nothing about bias or coverage. The
complete survey, extended radial families, and quadrature convergence need
their own calculations with correspondingly larger inputs.
