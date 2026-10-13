# ilSN tutorials

Run the scientific pipeline one step at a time, inspect its predictions, and
change its physical assumptions. These notebooks call the same public modules
as the command-line examples. Figures and short numerical outputs are saved
so the notebooks can also be read directly on GitHub.

| Notebook | Scientific question | Inputs |
| --- | --- | --- |
| [01 — Roman magnification](01_roman_magnification.ipynb) | What absolute magnification information does one SN provide? | Bundled simulated Roman response and one noisy training event |
| [02 — Radial transfer](02_radial_transfer.ipynb) | How much of that information reaches the Einstein radius? | Physical mass states, mock imaging and a prescribed brightness response |
| [03 — Population transfer](03_population_transfer.ipynb) | How does a source population inform independent strong lenses? | Reduced Roman simulation input, shared calibration and a common target-population contrast |

Start with basic Python/NumPy and familiarity with SN brightness and lens
magnification. Each notebook states its units and assumptions near the relevant
calculation. Follow them in order for the scientific story, or run any one in
a fresh kernel.

## Run locally

From the repository root, using your Python environment:

```bash
python -m pip install -r requirements_notebooks.txt
python -m jupyterlab notebooks
```

Select the Python kernel from that environment. No package-index installation
of `ilsn`, external catalogue download or cluster connection is needed. The
setup cells work when the kernel starts in either the repository root or its
`notebooks` directory.

To execute all notebooks without opening a browser:

```bash
python -m nbconvert --to notebook --execute notebooks/*.ipynb \
  --output-dir=outputs/notebooks --ExecutePreprocessor.timeout=180
```

GitHub Actions runs this same command alongside the scientific tests, examples
and documentation build. Executed copies go to the ignored `outputs` directory.

## Interpret the results

The notebooks illustrate simulation-based inference. The single-SN likelihood
is not a universal 15% error assignment. Radial transfer depends on the mass
family and its prior. Population transfer retains a shared source–target
difference and each target's own observations.

The population input has only two SN foregrounds and a coarse 64-node source
rule. The notebook compares identical finite rules before and after the SN
update and shows their difference from a Gaussian-prior reference. That
numerical discrepancy is not ilSN information. These tutorials do not establish
the full Roman survey's precision, interval coverage or an H0 constraint.

