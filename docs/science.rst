Scientific model
================

The package separates a measured outer brightness from a model-dependent
inner inference and a conditional population transfer. It does not assume
that a single magnification measurement uniquely determines an internal
mass sheet, a Fermat-potential difference, or H0.

The SN observable and response
------------------------------

For positive flux magnification, the photon magnitude shift is

.. math::

   d = -2.5\log_{10}\mu + A_{\rm dust}.

The data are standardized residuals ``y = R - C0``. A fixed reference distance
law, standardization coefficients and control origin define R and C0. The
response supplies three jointly calibrated functions at fixed true/design
source redshift:

.. math::

   m_\psi(d,z),\qquad v_\psi(d,z),\qquad S_\psi(d,z).

Here S is the probability of delivery, and the Gaussian moment approximation
K for the delivered residual has mean m and variance v. A response realization
psi includes whole curves of all three quantities. Between knots we linearly
interpolate variance, not standard deviation. A fixed-redshift simulation
does not define a continuous-redshift response simply because bin edges exist.

The default response rejects photon shifts outside the measured knots.
The Roman population calculation explicitly chooses ``unit_photon``
continuation: beyond an edge, the mean gains the additional photon shift
with unit slope while variance and delivery stay at the edge value. This is
an imposed tail law, not additional measured support. It never extends the
true-redshift domain. The choice is recorded in the runnable example's output.

The simulated selected variance already contains the SN/fitting/redshift
scatter present in that simulation. Do not add it again as a generic 15%
per-SN magnification error. Additional microlensing variance in ``SourceSample``
is explicitly a post-delivery approximation. Dust changes d before both K
and S. Additional common/cell offsets, when supplied, shift y after delivery.
The finite-parent uncertainty of the calibration does not measure physical
simulator mismatch or every possible photometric calibration error.

At fixed magnification, S cancels from the conditional single-SN density.
``magnification_log_likelihood`` implements this brightness illustration.
Its likelihood width is neither a universally assigned uncertainty nor a
coverage measurement. SNANA/PIPPIN light-curve simulation and fitting are
upstream inputs, not reimplemented here.

Source selection and common errors
----------------------------------

Let k index mass states, l local nuisance components and w denote the mass
population. Component weights a include pre-delivery physical/geometry
admission. For one delivered source,

.. math::

   L_i(\eta,\psi) =
   \frac{\sum_{k,l} w_k(\eta)a_{ikl}
     S_\psi(d_{ikl},z_i)K_\psi(y_i\mid d_{ikl},z_i)}
   {\sum_{k,l} w_k(\eta)a_{ikl}S_\psi(d_{ikl},z_i)}.

The shared prior is integrated after the catalogue product:

.. math::

   L_{\rm SN}(\eta)=\sum_\psi p(\psi)\prod_i L_i(\eta,\psi).

Dust, mean response, variance response and delivery remain paired within the
same node. Counts and source covariates are conditioned upon. There is no
Poisson survey-yield term. If multiple catalogues share calibration, add their
*conditional* log likelihoods before integrating the common prior once; a
product of independently marginalized catalogues would change the model.

Physical radial freedom
-----------------------

The supplied spherical model contains independently normalized Sersic stellar
mass with a radial M/L gradient, a generalized NFW halo, and optional signed
finite cores. A core has

.. math::

   \Sigma(R)=\Sigma_0[1+(R/r_c)^2]^{-3/2},\qquad
   M_{\rm total}=2\pi\Sigma_0 r_c^2.

The reference constructor assigns ``r_c=q*Re_ref`` and scales the amplitude
with q so that the broad signed mass remains
``2*reference_q**2*eta*M_ref``. A signed pair can redistribute mass with zero
total mass. The fixed reference mass and size are declared covariates;
they are not inferred anew from each model's Einstein radius.

Each model solves its own Einstein ring and returns

.. math::

   s_E=1-\kappa(\theta_E),\qquad
   \xi_E=\frac{\theta_E\alpha''(\theta_E)}{1-\alpha'(\theta_E)}.

Under an exact mass-sheet transformation, s_E scales with lambda and xi_E
is invariant. A finite core is only an approximate radial mass-sheet mode.
``MSTLens`` provides the exact mathematical transformation for comparison;
an infinite uniform sheet is not substituted for an admitted finite galaxy.

``RadialTransfer`` combines the outer SN density with supplied mass-state
weights and any inner-data likelihood. Its forecast averages posterior
variance over the selected predictive distribution. For a fair comparison,
both prior and posterior include the same inner observation. One favorable
SN noise draw is not the expected radial information gain.

Physical admissibility, information and numerical precision are distinct.
Signed cores must be added to the positive components before checking total
density and monotonicity. The research bank builder uses
``certify_material_support``: analytic global bounds on each signed
finite-core/halo density ratio and derivative ratio, followed by adaptive
interval bounds on their sum. It certifies the halo-plus-core component as
positive and decreasing for every radius; adding the independently positive
stellar component preserves this. Requiring halo-plus-core positivity is a
declared family restriction: a galaxy rescued only by stars is excluded.
Physical failure and unresolved numerical certification have separate masks.
This is not a positive-distribution-function certificate. The lower-level
``physical_support`` remains a sampled diagnostic for arbitrary profiles;
it is not the admission rule of the Roman bank builder.
Circular single-image selection checks the source against the radial caustic,
not only the sign of the magnification. The caustic search includes a declared
central deflection limit and refines bracketed roots; refine the bracketing
grid and check its inner boundary for new or narrow profile features.

An outer-null radial mode remains unconstrained even with exact outer
magnification. The ``radial_models`` example constructs such models directly.
Imaging, kinematics or justified physical priors can restrict their inner
freedom; they must enter as explicit observations or assumptions.

Conditional population transfer
-------------------------------

The mass-state population is an exponential tilt of a base quadrature:

.. math::

   w_k(\eta)\propto w_{0k}\exp[\eta^T T_k],\qquad
   \eta_{\rm target}=\eta_{\rm source}+\delta.

Scores T describe the allowed population shifts and correlations. Radial
family weights outside these scores remain imposed. One contrast delta is
shared across all target lenses. Its prior is independent of the source
likelihood; source data alone cannot measure an independent target contrast.

``MassGrid.primary()`` supplies 4,860 states in stellar mass multiplier,
halo residual, concentration residual, stellar M/L gradient, halo cusp and
signed core amplitude. Its eight scores are the standardized first six
coordinates, core amplitude squared, and the gradient--core interaction.
Their centers and scales are frozen to that primary measure, so refinement
does not change the meaning of eta or delta. They are exponential-tilt
coordinates, not literal Gaussian scatter or correlation coefficients.
``MassGrid.roman()`` adds q=3, 5, 10 and compensated radial redistribution,
giving 72,900 states. These extra radial freedoms retain imposed conditional
weights: the population does not learn their frequencies. Vary those weights
as a model-prior sensitivity calculation. A small ``MassGrid()`` retains the
same score definition but is only a 72-state numerical tutorial.

``PhysicalRelations`` exposes the assigned stellar--halo and concentration
relations and scatters. Reference masses and radii are supplied simulation
covariates, not observations of stellar mass or dynamics. ``td_selection``
weights by circular caustic area and, by default, a 150--350 km/s SIS velocity
proxy. It approximates a TD-like conditional sample; it is not a calibrated
survey discovery model or a measured velocity dispersion. The compressed
Gaussian theta_E/xi_E likelihood uses fixed observed uncertainties. Real pixel
imaging or dynamics likelihoods may enter through ``TargetSample`` instead.

For target j, selection t and its own imaging/dynamics likelihood I produce

.. math::

   p_j(k\mid\eta,\delta,I_j,\mathrm{selected})\propto
       w_k(\eta+\delta)t_{jk}I_j(k),

with physical admission included in t. Its evidence is

.. math::

   Z_j(\eta,\delta)=
      \frac{\sum_k w_k(\eta+\delta)t_{jk}I_j(k)}
           {\sum_k w_k(\eta+\delta)t_{jk}}.

The product of all Z_j weights the common population nodes. The comparison
without ilSN and the comparison with ilSN both include this product; only
the latter additionally includes the source-survey likelihood.

At fixed common parameters, target mass states are independent. After
marginalization they are correlated:

.. math::

   \mathrm{Cov}(s_i,s_j)=
   \mathrm{Cov}_{\eta,\delta}(m_i,m_j)
      +\delta_{ij}\,E_{\eta,\delta}[v_i].

The finite catalogue-mean variance is the sum of all covariance elements
divided by the square of the target count. It is not the single-SN error
divided by the square root of the number of SNe. A narrower conditional
posterior also does not prove smaller bias or correct interval coverage.
Some noise realizations or population shifts can increase a posterior width.

Scope and verification
----------------------

The package exposes the same conditional likelihood law and physical profile
equations as the research calculation, but uses direct quadrature for clarity.
Large mass grids, thousands of response draws and large source samples can
be computationally expensive. Process eta chunks separately and combine their
likelihood arrays in their original order; apply the population prior once.
``aligned_log_likelihood`` instead evaluates an explicitly paired joint
population/calibration rule, avoiding the full Cartesian product. The caller
must supply its correct joint prior or importance weights and check integration
convergence; the method does not infer a proposal from the data.
``inference.source_posterior`` streams source blocks and sums their conditional
likelihoods before marginalizing shared dust and calibration. This avoids a
survey-wide source-by-state-by-LOS cube; it does not claim production speed.
``sampling.receiving_nodes`` forms eta+delta directly from the fresh source
posterior. ``target_readout`` applies all target evidence and returns the full
covariance. Source and target may use different state counts if their score
definition is fixed. The simpler ``population_transfer`` uses a Cartesian
rule and is useful for small calculations and direct numerical comparisons.
The distributed scheduler and paper-specific acceleration caches are outside
this source distribution.

Check the quantities that matter under finer mass, population, contrast,
calibration and radial quadrature. Tail support, finite calibration, radial
model freedom, selection mismatch and population mismatch require separate
scientific tests. No claim of calibrated coverage, all-systematics accuracy,
or H0 precision follows from the illustrative examples.
