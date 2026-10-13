Contributing
============

Organize additions around scientific quantities. Use functions for numerical
operations and classes when observations or a model are reused. Keep the
equation, units and array axes visible; prefer an explicit parameter over
an implicit global setting. Format Python with Black.

Tests should answer a scientific or numerical question: does the mass integral
agree with the density, does a probability normalize, does a shared zero point
produce correlated errors, or does a limiting case have the correct result?
Use numerical tolerances appropriate to the calculation and test scientific
behavior rather than file layout or source text.

For changes to a likelihood, update its equation in ``docs/science.rst`` and
add a small analytic or independently enumerated comparison. Changes to a
radial family should state its physical admission assumptions and preserve
the distinction between a prescribed prior and an empirically calibrated one.

Run ``python -m pytest`` and the affected example before contributing. For
large calculations, compare the actual reported quantities under a finer
quadrature rather than relying on a generic numerical quality flag.
