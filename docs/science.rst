Scientific model
================

The assumed radial model connects outer SN magnification to s_E at the
Einstein radius. The conditional population model describes which
mass-distribution parameters source and target lenses share.

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

S is the delivery probability. K approximates the delivered residual as a
Gaussian with mean m and variance v; each realization psi contains jointly
calibrated curves of all three quantities. Variance is interpolated linearly
between knots. Redshift support is limited to the calibrated true/design
redshifts, even when an archive supplies bin edges.

The default response rejects photon shifts outside the measured knots.
The Roman example records its chosen ``unit_photon`` tail law: the mean
continues with unit photon-shift slope while variance and delivery retain
their edge values. This extrapolation assumes the tail response and preserves
the calibrated true-redshift domain.

The selected variance includes the simulated SN/fitting/redshift scatter;
adding a generic 15% per-SN magnification error would count it twice.
``SourceSample`` treats additional microlensing variance as a post-delivery
approximation. Dust changes d before K and S; common/cell offsets shift y
after delivery. Finite-parent calibration uncertainty describes sampling
variation; simulator mismatch and photometric calibration errors require
separate models.

At fixed magnification, S cancels from the conditional single-SN density in
``magnification_log_likelihood``. Its width describes the calibrated brightness
likelihood for that event; interval coverage requires a separate test.
SNANA/PIPPIN supplies the upstream light-curve simulation and fitting.

Source selection and common errors
----------------------------------

Let k index mass states, l local nuisance components and w denote the mass
population. Component weights a include pre-delivery physical/geometry
admission. For one delivered source,

.. math::

   L_i(\eta,\psi) =
   \frac{\sum_{k,l} w_k(\eta)F_{ik}a_{ikl}
     S_\psi(d_{ikl},z_i)K_\psi(y_i\mid d_{ikl},z_i)}
   {\sum_{k,l} w_k(\eta)a_{ikl}S_\psi(d_{ikl},z_i)}.

F is an optional likelihood of independent foreground data, such as a
measured velocity dispersion, for each mass state (one by default). It enters
the numerator only: selection acts on the SN, not on the foreground data.

The shared prior is integrated after the catalogue product:

.. math::

   L_{\rm SN}(\eta)=\sum_\psi p(\psi)\prod_i L_i(\eta,\psi).

Each node keeps dust, mean, variance and delivery paired. The likelihood
conditions on counts and source covariates, omitting a Poisson yield term.
For catalogues sharing calibration, sum their conditional log likelihoods
before integrating the common prior once.

Physical radial freedom
-----------------------

The supplied spherical model contains independently normalized Sersic stellar
mass with a radial M/L gradient, a generalized NFW halo, and optional signed
finite cores. A core has

.. math::

   \Sigma(R)=\Sigma_0[1+(R/r_c)^2]^{-3/2},\qquad
   M_{\rm total}=2\pi\Sigma_0 r_c^2.

The reference constructor assigns ``r_c=q*Re_ref`` and scales the amplitude
with q to keep the broad signed mass at ``2*reference_q**2*eta*M_ref``.
A signed pair can redistribute mass with zero total mass. Reference mass
and size are fixed covariates independent of each model's Einstein radius.

Each model solves its own Einstein ring and returns

.. math::

   s_E=1-\kappa(\theta_E),\qquad
   \xi_E=\frac{\theta_E\alpha''(\theta_E)}{1-\alpha'(\theta_E)}.

Under an exact mass-sheet transformation, s_E scales with lambda and xi_E
is invariant. Finite cores approximate radial mass-sheet freedom within the
galaxy family; ``MSTLens`` supplies an exact mathematical comparison.

``RadialTransfer`` combines the outer SN density, supplied mass-state weights
and inner-data likelihood. Expected information gain averages posterior
variance over the selected predictive distribution, with the same inner
observation in both the before-SN and after-SN calculations.

Admission uses the summed signed and positive components.
``certify_material_support`` bounds each finite-core/halo density and
derivative ratio globally, then bounds their sum with adaptive intervals.
It certifies a positive, decreasing halo-plus-core density at every radius;
adding the positive stellar component preserves this property. This family
restriction excludes models requiring stars to rescue a negative halo-plus-core
density and does not certify a positive distribution function. Separate masks
record physical failure and unresolved certification. Roman banks use this
certificate for admission; ``physical_support`` is a sampled diagnostic for
arbitrary profiles.

Circular single-image selection uses the radial caustic. The search imposes
a declared central deflection limit and refines bracketed roots. Refine its
bracketing grid and check the inner boundary when introducing narrow or new
profile features.

The ``radial_models`` example constructs outer-null modes that remain
unconstrained even with exact outer magnification. Their inner freedom
requires explicit imaging, kinematics or justified physical priors.

Conditional population transfer
-------------------------------

The mass-state population is an exponential tilt of a base quadrature:

.. math::

   w_k(\eta)\propto w_{0k}\exp[\eta^T T_k],\qquad
   \eta_{\rm target}=\eta_{\rm source}+\delta.

Scores T define the allowed population shifts and correlations; other radial
family weights are prescribed. All targets share one contrast delta with a
prior independent of the source likelihood, so source data alone leave that
contrast prior unchanged.

``MassGrid.primary()`` supplies 4,860 states in stellar mass multiplier,
halo residual, concentration residual, stellar M/L gradient, halo cusp and
signed core amplitude. Its eight scores are the standardized first six
coordinates, core amplitude squared, and the gradient--core interaction.
Their centers and scales are frozen to the primary measure, preserving the
meaning of eta and delta under refinement. These are exponential-tilt
coordinates rather than Gaussian scatter or correlation coefficients.
``MassGrid.roman()`` adds q=3, 5, 10 and compensated radial redistribution,
giving 72,900 states. Their conditional weights are prescribed and should be
varied to assess model-prior sensitivity. The 72-state ``MassGrid()`` tutorial
uses the same score definition.

``PhysicalRelations`` specifies the stellar--halo and concentration relations
and scatters; reference masses and radii are simulation covariates.
``td_selection`` approximates a TD-like conditional sample with circular
caustic area and a default 150--350 km/s SIS velocity proxy. Calibrated survey
selection and measured dynamics require their own inputs.

``aperture_dispersion`` predicts the luminosity-weighted line-of-sight
dispersion inside a circular aperture from the spherical Jeans equation with
constant anisotropy beta. The Sersic light (without the M/L gradient) traces
the full potential of stars, halo and signed cores:

.. math::

   \nu\sigma_r^2(r)=r^{-2\beta}\int_r^\infty s^{2\beta}\nu(s)
       \frac{GM(s)}{s^2}\,ds,\qquad
   \sigma_{\rm ap}^2=\frac{\int\nu\sigma_r^2(r-s)
       [1-\beta(r-s)(2r+s)/(3r^2)]\,r\,dr}{\int\nu(r-s)\,r\,dr},

with s=sqrt(max(r^2-R^2, 0)) for aperture radius R. Seeing, slit geometry and
rotation are not modelled. The mock inner
likelihood is a compressed Gaussian in theta_E/xi_E with fixed observed
uncertainties; ``TargetSample`` also accepts pixel-imaging or dynamics
likelihoods.

For target j, selection t and its own imaging/dynamics likelihood I produce

.. math::

   p_j(k\mid\eta,\delta,I_j,\mathrm{selected})\propto
       w_k(\eta+\delta)t_{jk}I_j(k),

with physical admission included in t. Its evidence is

.. math::

   Z_j(\eta,\delta)=
      \frac{\sum_k w_k(\eta+\delta)t_{jk}I_j(k)}
           {\sum_k w_k(\eta+\delta)t_{jk}}.

Both before-ilSN and after-ilSN calculations weight common population nodes
by the product of all Z_j; the latter also includes the source likelihood.

At fixed common parameters, target mass states are independent. After
marginalization they are correlated:

.. math::

   \mathrm{Cov}(s_i,s_j)=
   \mathrm{Cov}_{\eta,\delta}(m_i,m_j)
      +\delta_{ij}\,E_{\eta,\delta}[v_i].

The finite catalogue-mean variance uses the sum of all covariance elements
divided by the squared target count. Posterior width depends on the noise
realization and population shift; bias and interval coverage require separate
evaluation.

Numerical integration
---------------------

Direct quadrature supports small calculations and numerical comparisons.
For larger rules, ``aligned_log_likelihood`` pairs population and calibration
nodes using supplied joint prior or importance weights.
``inference.source_posterior`` streams source blocks, summing their conditional
log likelihoods before marginalizing shared dust and calibration.
``sampling.receiving_nodes`` forms eta+delta from the source posterior;
``target_readout`` applies all target evidence and returns the full covariance.
Source and target state counts may differ when their score definition is fixed.
The Cartesian ``population_transfer`` provides a small-rule comparison.
See :doc:`usage` for weighting and matched-baseline conventions.

Check the reported quantities under finer mass, population, contrast,
calibration and radial quadrature. Assess response tails, finite calibration,
radial model freedom, selection mismatch and population mismatch separately:
numerical convergence establishes precision within the chosen model.
