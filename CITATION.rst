Citation and data credits
=========================

This package accompanies the ilSN simulation study by Xianzhe “TZ” Tang and
collaborators. The manuscript citation will be added when publicly available.

The bundled simulation inputs use the following contributions:

* **Roman SN response:** ``data/roman_response.npz`` contains the project's
  TRAIN20 calibration, generated with
  `SNANA <https://github.com/RickKessler/SNANA>`_ and
  `PIPPIN <https://github.com/dessn/Pippin>`_. The selected means, variances,
  delivery probabilities and joint calibration draws were computed for this
  study from matched simulated SN Ia light curves.
* **Foreground galaxies:** the mass, size and redshift design in
  ``data/roman_transfer.npz`` descends from a population generated with
  `SLSim <https://github.com/LSST-strong-lensing/slsim>`_. The physical lens
  banks, SN residual realizations and mock inner-imaging observations were
  generated for this study.
* **Line-of-sight environments:** the paired external convergence and shear
  samples come from `Tang et al. (2025), Analyzing Line-of-sight selection
  biases in galaxy-scale strong lensing with external convergence and shear
  <https://arxiv.org/abs/2506.04201>`_. The source distributions are available
  in the `LSST strong-lensing data repository
  <https://github.com/LSST-strong-lensing/data_public/tree/main/Line_of_sight_kg_distributions>`_.
  The reduced input retains convergence and shear from the same sampled rows.

These are simulated inputs. Their construction and assumptions are described
in ``data/README.rst``. Please cite the relevant sources when using them.
