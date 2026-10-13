Contributing
============

Organize additions around scientific quantities. Use functions for numerical
operations and classes for reused observations or models. Document equations,
units and array axes, expose settings as explicit parameters, and format
Python with Black.

Tests should answer a scientific or numerical question: does the mass integral
agree with the density, does a probability normalize, does a shared zero point
produce correlated errors, or does a limiting case have the correct result?
Use numerical tolerances appropriate to the calculation and test scientific
behavior rather than file layout or source text.

When changing a likelihood, update its equation in ``docs/science.rst`` and
add an analytic or independently enumerated comparison. For radial families,
state the physical admission assumptions and identify prescribed and
empirically calibrated priors.

Run ``python -m pytest`` and the affected example before contributing. For
large calculations, compare the actual reported quantities under a finer
quadrature rather than relying on a generic numerical quality flag.
