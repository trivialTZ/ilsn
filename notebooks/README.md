# ilSN tutorials

Follow the inference from SN brightness to radial lens structure and a shared
mass population. The notebooks use the public modules, with saved figures and
numerical outputs for reading on GitHub.

| Notebook | Scientific question | Inputs |
| --- | --- | --- |
| [01 — Roman magnification](01_roman_magnification.ipynb) | What magnification information does one SN provide? | Simulated Roman response and one noisy training event |
| [02 — Radial transfer](02_radial_transfer.ipynb) | How does that information reach the Einstein radius? | Physical mass states, mock imaging and a prescribed brightness response |
| [03 — Population transfer](03_population_transfer.ipynb) | How does the source population inform independent strong lenses? | Reduced Roman simulation, shared calibration and a common population contrast |

Start with basic Python/NumPy and familiarity with SN brightness and lens
magnification. Read in order or run any notebook in a fresh kernel. Each explains
its assumptions and numerical comparisons beside the calculation.

## Run locally

From the repository root:

```bash
python -m pip install -r requirements_notebooks.txt
python -m jupyterlab notebooks
```

Choose the Python kernel from that environment. Setup cells import the local
source from the repository root or `notebooks/`. All input data are bundled.

To execute every notebook:

```bash
python -m nbconvert --to notebook --execute notebooks/*.ipynb \
  --output-dir=outputs/notebooks --ExecutePreprocessor.timeout=180
```

GitHub Actions runs this command alongside tests, examples and the documentation
build. Executed copies go to the ignored `outputs` directory.
