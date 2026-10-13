Citation and data credits
=========================

This package accompanies the ilSN simulation study by Xianzhe “TZ” Tang and
collaborators. The manuscript citation will be added when publicly available.

The bundled inputs use the following contributions:

* **Roman SN response:** ``data/roman_response.npz`` contains the project's
  TRAIN20 calibration from matched SN Ia simulations processed with
  `SNANA <https://github.com/RickKessler/SNANA>`_ and
  `PIPPIN <https://github.com/dessn/Pippin>`_.
* **Foreground galaxies:** the mass, size and redshift design in
  ``data/roman_transfer.npz`` derives from a population generated with
  `SLSim <https://github.com/LSST-strong-lensing/slsim>`_. Lens-model predictions,
  SN residual realizations and mock inner-imaging observations were generated
  for this study.
* **Line-of-sight environments:** paired external convergence and shear
  samples come from `Tang et al. (2025), Analyzing Line-of-sight selection
  biases in galaxy-scale strong lensing with external convergence and shear
  <https://arxiv.org/abs/2506.04201>`_. The source distributions are in the
  `LSST strong-lensing data repository
  <https://github.com/LSST-strong-lensing/data_public/tree/main/Line_of_sight_kg_distributions>`_.

Please cite the relevant sources when using these inputs. Their construction
and assumptions are described in `data/README.rst <data/README.rst>`_.
