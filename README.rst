ilSN
====

Absolute supernova magnification and conditional mass priors for strong lenses.

``ilSN`` follows three steps: standardized SN Ia brightness measures
magnification at the SN position; physical mass models transfer that
information to the Einstein radius; a conditional population model shares
information with independent strong lenses that have no supernova of their own.

The code supports simulation studies. Its forecasts depend on the supplied
SN response, radial mass family, selection functions and source--target
population relation. An uncertainty in magnification is not automatically
an uncertainty in the mass sheet or in H0.

Getting started
---------------

Use Python 3.10 or newer. Run directly from this source directory; installing
the package itself is unnecessary::

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    PYTHONPATH=. python examples/quickstart.py
    PYTHONPATH=. python examples/roman_transfer.py

The example simulates SN residuals, evaluates radial transfer with mock
imaging, and compares independent target lenses before and after sharing
the source population. It writes a three-panel figure and numerical results
to ``outputs/quickstart/``. The small simulation uses prescribed priors and
is not the paper's full Roman forecast.

``roman_transfer.py`` instead reads a bundled reduced input from the actual
Roman simulation calculation. It recomputes the selected-source likelihood,
shared-calibration population posterior and independent target covariance.
The input contains two sources, ten targets, 1,620 mass states and the actual
eight population coordinates. No saved posterior is used. This makes the
research calculation inspectable without downloading the large production
banks. It is a numerical workflow example, not a Roman precision estimate.

A physical lens can also be constructed directly:

.. code-block:: python

    from ilsn.lens_model import GalaxyProfile, CompositeProfile

    galaxy = GalaxyProfile(
        z_l=0.3, re_arcsec=0.8, stellar_mass_msun=2e11,
        halo_mass_msun=1e13, concentration=7.0,
    )
    profile = CompositeProfile.from_reference(
        galaxy, reference_mass_msun=2e11, reference_re_kpc=galaxy.re_kpc,
        eta=0.03, q=5.0, compensation=0.0,
    )
    lens = profile.lens(z_s=1.2)
    theta_e = lens.theta_E()          # arcsec, solved for this model
    s_e = lens.s_E()                 # 1 - kappa(theta_e)
    mu = lens.mu(3.0 * theta_e)      # signed macro magnification

For an ilSN, check single-image geometry separately before interpreting
``abs(mu)`` as its flux magnification. The complete example does this.

Examples
--------

* ``examples/quickstart.py``: the complete residual → radial → population
  calculation, with the same target imaging on both sides of the comparison.
* ``examples/roman_magnification.py``: a conditional brightness likelihood
  from the bundled simulated Roman calibration, including its joint draws.
* ``examples/roman_transfer.py``: fresh population inference from saved Roman
  simulation inputs, with paired environments, common calibration, target
  imaging and an explicit source--target contrast prior.
* ``examples/radial_models.py``: different physical radial profiles that share
  an outer magnification and retain different inner structure.

Run each using ``PYTHONPATH=. python examples/<name>.py``. The figures use
serif text and a common blue/orange palette. ``--output-dir`` chooses where
results are written. No download or cluster access is needed.

Scientific interface
--------------------

``supernova``
    Explicit-reference standardization, magnification shifts, and SNANA FITRES
    input. The Roman raw-Tripp observable and its selection are included.
``response``
    Selected mean, variance and delivery probability, their shared calibration
    draws, paired-parent calibration from simulations, and NPZ input/output.
``lens_model`` and ``geometry``
    Sersic stars with a mass-to-light gradient, gNFW halos, signed finite cores,
    compensated redistribution, circular lens geometry, and MST identities.
``mass_grid`` and ``lens_bank``
    Physical covariates, stellar/halo relations, the actual weighted Roman
    mass grids and frozen eight-score mapping, all-radius material admission,
    and source/target predictions. Source and target grids can be refined
    separately without redefining population parameters.
``observation``
    Paired external convergence/shear, photon magnification and attenuation,
    prescribed microlensing, TD-like selection and compressed inner imaging.
``likelihood``
    Source selection, local nuisance components, common response/dust/offsets,
    and fixed-size selected-sample simulation.
``inference`` and ``sampling``
    A joint source/calibration rule evaluated in source blocks, followed by
    paired receiving nodes. Shared errors are integrated after all blocks.
``radial``
    Inner mass-state posterior and predictive-average information gain,
    with optional inner imaging or dynamics likelihoods.
``population``
    Shared mass-population coordinates, an explicit common target contrast,
    target selection and imaging evidence, and the full target covariance.

Arrays are ordinary NumPy arrays. Physical banks from other lens engines can
be supplied to ``SourceSample`` and ``TargetSample`` without inheriting a
package-specific lens class. See ``docs/usage.rst`` for shapes and a complete
population-transfer call, and ``docs/science.rst`` for equations and assumptions.

Testing and contributions
-------------------------

::

    python -m pip install -r requirements_dev.txt
    python -m pytest

Tests cover physical identities, selected-density normalization, common
calibration across source blocks, radial null directions, imaging evidence,
material admission, population coordinates and target covariance.
Please keep new code close to the underlying equations and describe units,
array axes and physical assumptions in its docstrings. See ``CONTRIBUTING.rst``.

Credit
------

The project is maintained by Xianzhe “TZ” Tang. It follows the focused
scientific interfaces of `PSFr <https://github.com/sibirrer/psfr>`_ and
`SLSim <https://github.com/LSST-strong-lensing/slsim>`_. Neither is a runtime
dependency of this source distribution. Citation guidance and the distinction
between software and scientific input citations are in ``CITATION.rst``.

License: MIT; see ``LICENSE``.
