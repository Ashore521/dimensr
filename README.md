# DimenSR

**Dimension-constrained Neural-guided Symbolic Regression** — a research implementation combining a unit-aware GRU expression generator, genetic programming, and NSGA-II model selection.

[中文使用说明](docs/README.zh-CN.md) · [Algorithm](docs/ALGORITHM.md) · [Source mapping](docs/SOURCES.md) · [Experiment protocol](docs/EXPERIMENTS.md) · [Acceptance results](docs/ACCEPTANCE_RESULTS.zh-CN.md)

[Version 0.3 boundary/search results](docs/BOUNDARY_SEARCH_RESULTS.zh-CN.md): 54 completed
fits; improved strict numerical recovery on the small study, with remaining
composition-search failures and an explicit extrapolation-domain counterexample.

Version 0.4 adds bounded interval-domain screening and complementary-feature
search. Use `model.audit_domain(input_bounds=...)` to inspect the selected raw
tree over an intended rectangular operating range; an unresolved result is
reported explicitly. See the [study protocol](docs/INTERVAL_SEARCH_PROTOCOL.zh-CN.md)
and [results](docs/INTERVAL_SEARCH_RESULTS.zh-CN.md): 18/18 strict numerical recoveries
for 0.4 versus 12/18 for the contemporaneous 0.3 baseline on six reused noiseless tasks.

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

Version 0.3 checks every intermediate operation and probes training-range endpoints,
zero when in range, and bounded corner combinations. Pass `fit(..., input_bounds=...,
X_domain=...)` to declare additional valid input ranges or unlabelled points.
For coupled/categorical domains, disable `boundary_probes` and set
`domain_probe_count=0`, then provide permitted `X_domain` points only.
Finite probes are **not a continuous-domain proof**. See the [domain API](docs/README.zh-CN.md#声明合法输入域).

With `interval_checks=True` (0.4 default), candidate acceptance also requires
interval screening to return `enclosed` on the declared box, or on the training
box when automatic boundaries are enabled. `invalid` and `unresolved` are rejected.
This numerical check may conservatively reject valid expressions and is not a
formally verified mathematical certificate. `predict` still checks supplied rows;
it does not automatically approve a new operating range.

## Fit your data

```bash
dimensr fit --data measurements.csv --target force --metadata examples/force_metadata.json --output outputs/model.json
dimensr predict --model outputs/model.json --data new_measurements.csv --output outputs/predictions.csv
```

CSV files require unique headers and finite numeric data. The feature order comes from `feature_names` in the metadata or from CSV column order excluding the target. The prediction command matches columns by name. Pass `--validation validation.csv` for a predefined validation set. Python callers can use `fit(X, y, X_val=X_val, y_val=y_val)`.

By default, `fit` makes a reproducible 80/20 split. Constants, search rewards and complexity estimates use only the training portion. Validation selects the final member of the training Pareto front. Hold back a separate test set yourself. Time series and grouped observations require an appropriate explicit validation split. No final refit on validation data is performed.

## What is implemented

- Exact rational unit checking, completion-aware generation masks and same-unit subtree replacement.
- GRU token and legal unit-split probabilities trained by a risk-seeking policy gradient; separate, diverse GP elite imitation.
- Typed mutation/crossover with inherited constants, additive seeds, persistent elites and bounded duplicate retries.
- Bounded, unit-checked composite-feature and sparse common-denominator initialization using training targets only (`structured_seeds=True`).
- Domain-aware constant starts, exact affine coefficient subproblems and bounded elite polishing.
- Accuracy/size/Pareto-aware candidate screening, retaining a requested 30% or at least 50 unique valid trees when available.
- NSGA-II over training error and mixed complexity; optional size-density adjustment of crowding distance.
- A reproducible Rademacher estimate for an explicit local Jacobian output ball, normalized for mixed-complexity selection. This is **not a certified generalization bound for symbolic regression**.
- Reserved NSGA-II evaluation budget; optional output bounds and training-only numerical-domain probes.
- Strict intermediate-operation checks, automatic boundary probes, and explicit unlabelled input-domain declarations.
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

The separate [acceptance pilot](docs/ACCEPTANCE_RESULTS.zh-CN.md) compares frozen
DimenSR 0.2.0 with an unmodified, pinned PhySO revision on eight equations, three
seeds and two noise levels, plus five internal ablations (126 runs). Its
[source-backed protocol](docs/ACCEPTANCE_PROTOCOL.zh-CN.md) distinguishes final
selection from candidate discovery, numerical recovery from symbolic equivalence,
and unit validity from domain validity. This is a small adapted-budget comparison;
compute and free-constant semantics differ, and legacy DSO has not been run.
Artifacts and rerun instructions are in [results/acceptance_v1](results/acceptance_v1).

### Real experimental data

The repository also includes a grouped pilot on UCI Airfoil Self-Noise, Concrete
Compressive Strength and Yacht Hydrodynamics. See the [Chinese results report](docs/REAL_DATA_RESULTS.zh-CN.md)
and [fixed protocol](docs/REAL_DATA_PROTOCOL.md). To reproduce from the repository root:

```bash
python -m pip install -e ".[experiments]"
python -m dimensr.real_data --output outputs/new_real_data
python examples/compare_optimization.py run --output outputs/my_optimization --workers 4
python examples/compare_optimization.py verify --output outputs/my_optimization
```

Original downloads stay in `outputs/real_data/cache`. Per-run metrics, split indices,
training-only scales, dimensional audits and portable models are in `results/real_data`.
The original pilot is frozen at commit `5492122`; rerunning it with the current
algorithm requires a new output directory and does not reproduce the old search.
The 0.2 comparison uses the same splits and seeds, with all three capacity variants
declared before new test scoring. See [optimization results](docs/OPTIMIZATION_RESULTS.zh-CN.md).
These reused test sets support exploratory comparison. The three grouped splits do not establish superiority
to established symbolic regression systems. Dataset licenses and attribution are
documented separately from the MIT code license.

## Attribution and license

MIT license for this implementation. [PhySO](https://github.com/WassimTenachi/PhySO) is MIT licensed and [DSO](https://github.com/dso-org/deep-symbolic-optimization) is BSD-3-Clause licensed. Both are acknowledged in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with inspected revisions and source mappings in [docs/SOURCES.md](docs/SOURCES.md). No upstream source files or paper PDFs are bundled. Use [references.bib](references.bib) to credit the underlying methods in research.
