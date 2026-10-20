ilSN
====

Absolute supernova magnification and conditional mass priors for strong lenses.

``ilSN`` models how intermediate-lensed Type Ia supernovae inform strong-lens
mass distributions in three steps:

1. Standardized SN brightness measures magnification at the SN position.
2. Physical mass models and inner observations transfer that information to
   the Einstein radius.
3. A conditional population model informs independent strong lenses through
   shared population parameters and an explicit source--target difference.

The examples use simulations. Transfer gains depend on the SN response,
allowed mass profiles, selection and population priors; see the
`scientific model <docs/science.rst>`_ for the equations and assumptions.

Getting started
---------------

Use Python 3.10 or newer and run from this source directory::

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    PYTHONPATH=. python examples/quickstart.py

The quickstart simulates SN residuals, combines them with mock inner imaging,
and calculates the covariance of independent target lenses. It saves a
three-panel figure and numerical results to ``outputs/quickstart/``.

For step-by-step calculations, open the
`notebook guide <notebooks/README.md>`_:

* `Roman magnification <notebooks/01_roman_magnification.ipynb>`_: light-curve
  brightening, the selected response, and a noisy SN's brightness likelihood.
* `Radial transfer <notebooks/02_radial_transfer.ipynb>`_: mass-model freedom,
  inner imaging, and information at the Einstein radius.
* `Population transfer <notebooks/03_population_transfer.ipynb>`_: shared
  calibration, independent targets, population differences and full covariance.

::

    python -m pip install -r requirements_notebooks.txt
    python -m jupyterlab notebooks

Command-line examples
---------------------

Run each with ``PYTHONPATH=. python examples/<name>.py``. All inputs are bundled;
``--output-dir`` selects the output directory.

* ``quickstart.py``: the complete residual → radial → population calculation.
* ``roman_magnification.py``: a brightness likelihood using the simulated
  Roman response and its joint calibration draws.
* ``roman_transfer.py``: source inference and target covariance from a reduced
  Roman input with two SN foregrounds, ten targets and 1,620 mass states.
  Its coarse integration rule illustrates the workflow; the notebook compares
  matched baselines to separate the SN update from integration differences.
* ``radial_models.py``: profiles with the same outer magnification and different
  inner structure.
* ``radial_information.py``: how one SN's information on s_E changes with its
  angle in Einstein radii, for three radial families, with and without inner
  imaging.
* ``information_flow.py``: a linearized budget of which population directions
  set the TD-lens s_E hypermean, and how much SN brightness, a velocity
  dispersion of the same foregrounds (individual or stacked), or both remove.

See `input data <data/README.rst>`_ for the simulation setup and numerical rule.

Scientific interface
--------------------

``supernova`` and ``response``
    SNANA FITRES input, standardized residuals, selected response moments,
    delivery probabilities and shared calibration draws.
``lens_model``, ``geometry``, ``mass_grid`` and ``lens_bank``
    Stars, dark halos, finite cores and mass redistribution; physical mass
    grids, single-image geometry and source/target predictions.
``dynamics``
    Spherical Jeans aperture velocity dispersions of the same mass models.
``observation`` and ``likelihood``
    Paired line-of-sight environments, dust, microlensing, source selection,
    shared errors and mock inner imaging.
``radial``
    Inner mass-state inference and predictive-average information gain.
``population``, ``inference`` and ``sampling``
    Shared mass-population parameters, source--target differences, source
    inference and the full covariance of receiving lenses.

Arrays are NumPy arrays. Predictions from other lens engines can be supplied
to ``SourceSample`` and ``TargetSample``. The
`usage guide <docs/usage.rst>`_ gives array conventions and worked API calls;
the `API reference <docs/api.rst>`_ documents individual functions.

Testing and contributions
-------------------------

::

    python -m pip install -r requirements_dev.txt
    python -m pytest

GitHub Actions runs the scientific tests, documentation build, examples and
all three notebooks. See `CONTRIBUTING.rst <CONTRIBUTING.rst>`_ for code and
test conventions.

Credit
------

The bundled inputs use:

* `SNANA <https://github.com/RickKessler/SNANA>`_ and
  `PIPPIN <https://github.com/dessn/Pippin>`_ for Roman SN simulation and response
  calibration.
* `SLSim <https://github.com/LSST-strong-lensing/slsim>`_ for foreground galaxy
  simulations.
* `Tang et al. (2025) <https://arxiv.org/abs/2506.04201>`_ for paired line-of-sight
  convergence and shear samples, available in the
  `LSST strong-lensing data repository
  <https://github.com/LSST-strong-lensing/data_public/tree/main/Line_of_sight_kg_distributions>`_.

See `CITATION.rst <CITATION.rst>`_ for attribution.

License
-------

MIT; see `LICENSE <LICENSE>`_.
