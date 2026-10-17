# DimenSR

**Dimension-constrained Neural-guided Symbolic Regression** — a research implementation combining a unit-aware GRU expression generator, genetic programming, and NSGA-II model selection.

[中文使用说明](docs/README.zh-CN.md) · [Algorithm](docs/ALGORITHM.md) · [Source mapping](docs/SOURCES.md) · [Experiment protocol](docs/EXPERIMENTS.md)

DimenSR discovers analytic relationships from numeric data and supplied physical units. It implements the proposed RNN–GP workflow in a small, independent PyTorch package, drawing on PhySO and DSO and the accompanying literature. It is an experimental algorithm, not an official release of either upstream project. Comparative superiority and research novelty have not been established.

## Install

Python 3.10+ is required. A CPU is sufficient. Clone the repository and install:

```bash
git clone https://github.com/Ashore521/dimensr.git
cd dimensr
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

For ordinary use, omit `[dev]`. No PyPI release is currently advertised. Dependencies install into your virtual environment; PhySO and legacy TensorFlow-based DSO are not required.

## Quick start

```bash
dimensr demo --case force --config examples/quick_config.json
```

```python
import numpy as np
from dimensr import DimenSRRegressor, SearchConfig

rng = np.random.default_rng(42)
X = rng.uniform(0.2, 2.0, (160, 2))
y = X[:, 0] * X[:, 1]

model = DimenSRRegressor(
    SearchConfig(random_state=42),
    feature_names=["mass", "acceleration"],
    feature_units=[[0, 1, 0, 0, 0, 0, 0], [1, 0, -2, 0, 0, 0, 0]],
    target_units=[1, 1, -2, 0, 0, 0, 0],
).fit(X, y)
print(model.expression_)
print(model.predict([[3.0, 2.0]]))
model.save("force_model.json")
restored = DimenSRRegressor.load("force_model.json")
```

`X` uses `(samples, features)` orientation. Unit exponents follow **L, M, T, I, Theta, N, J**: length, mass, time, current, temperature, amount, luminous intensity. Use consistent numeric units (for example all lengths in metres); metadata does not convert centimetres to metres. Unspecified units are dimensionless. Supply `constant_units` when a formula needs dimensional coefficients; each occurrence is fitted independently.

The estimator provides `fit`, `predict`, `score`, scikit-learn parameter cloning, `expression_`, `pareto_front_`, `history_`, and `diagnostics_`. `expression_` is a SymPy display expression; predictions use the original tree and its strict numerical domains. It does not include confidence intervals. JSON files store inference models, not resumable optimizer states.

## Fit your data

```bash
dimensr fit --data measurements.csv --target force --metadata examples/force_metadata.json --output outputs/model.json
dimensr predict --model outputs/model.json --data new_measurements.csv --output outputs/predictions.csv
```

CSV files require unique headers and finite numeric data. The feature order comes from `feature_names` in the metadata or from CSV column order excluding the target. The prediction command matches columns by name. Pass `--validation validation.csv` for a predefined validation set. Python callers can use `fit(X, y, X_val=X_val, y_val=y_val)`.

By default, `fit` makes a reproducible 80/20 split. Constants, search rewards and complexity estimates use only the training portion. Validation selects the final member of the training Pareto front. Hold back a separate test set yourself. Time series and grouped observations require an appropriate explicit validation split. No final refit on validation data is performed.

## What is implemented

- Exact rational unit checking, completion-aware generation masks and same-unit subtree replacement.
- GRU token probabilities trained by a risk-seeking policy gradient; separate GP elite imitation.
- Typed mutation/crossover and bounded nonlinear least-squares fitting of constants.
- Top 30% candidate screening, with a requested minimum of 50 unique valid trees, limited by what search actually discovers.
- NSGA-II over training error and mixed complexity; optional size-density adjustment of crowding distance.
- A reproducible Monte Carlo estimate over an explicit finite local perturbation family. This is a heuristic complexity term, **not a certified generalization bound**.
- Validation-based final model selection, portable JSON models, CLI, ablations, tests and CI.

Generation uses a finite unit catalog built from supplied units and a documented one-step closure. Some dimensionally legal expressions are outside this catalog. Supply `extra_units` or change search limits when needed; see [algorithm details](docs/ALGORITHM.md). Unsupported numerical domains are rejected rather than rewritten using protected operators.

## Reproduce and extend

```bash
python -m pytest -q
dimensr benchmark --config examples/quick_config.json --cases force kinetic nguyen1 trigonometric --seeds 0 1 2
dimensr benchmark --config examples/quick_config.json --cases kinetic --seeds 0 1 2 --variants full no_rnn no_gp no_units no_sd no_rad --output outputs/ablations.json
python -m build
```

Benchmarks save per-run configurations, raw model trees, errors, evaluation counts, constant-fit work and wall time. IID, extrapolation and exact-zero/boundary probe sets are not used during search. Numerical recovery checks all three and is not a proof of symbolic equivalence. These small cases exercise functionality; larger, matched-budget comparisons against upstream methods are still needed.

See [validation results](docs/VALIDATION.md) for checks actually run for this release. Training progress and benchmark checkpoints are written alongside CLI outputs. An interrupted benchmark checkpoint is inspectable; automatic training resume is not implemented.

### Real experimental data

The repository also includes a grouped pilot on UCI Airfoil Self-Noise, Concrete
Compressive Strength and Yacht Hydrodynamics. See the [Chinese results report](docs/REAL_DATA_RESULTS.zh-CN.md)
and [fixed protocol](docs/REAL_DATA_PROTOCOL.md). To reproduce from the repository root:

```bash
python -m pip install -e ".[experiments]"
python -m dimensr.real_data
```

Original downloads stay in `outputs/real_data/cache`. Per-run metrics, split indices,
training-only scales, dimensional audits and portable models are in `results/real_data`.
The three grouped splits are a feasibility pilot; they do not establish superiority
to established symbolic regression systems. Dataset licenses and attribution are
documented separately from the MIT code license.

## Attribution and license

MIT license for this implementation. [PhySO](https://github.com/WassimTenachi/PhySO) is MIT licensed and [DSO](https://github.com/dso-org/deep-symbolic-optimization) is BSD-3-Clause licensed. Both are acknowledged in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with inspected revisions and source mappings in [docs/SOURCES.md](docs/SOURCES.md). No upstream source files or paper PDFs are bundled. Use [references.bib](references.bib) to credit the underlying methods in research.
