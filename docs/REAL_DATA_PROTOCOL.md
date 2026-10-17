# Real-data pilot protocol

Defined on 2026-10-17 before inspecting this pilot's test scores. This is a local,
versioned protocol, not an externally preregistered study. The executable protocol
and its freeze time/hash are saved in `results/real_data/protocol.json`.

This is the historical **0.1** protocol at commit `5492122`. Current algorithm
changes require a new output directory. The follow-up in
`results/optimization_v2/protocol.json` freezes the new configurations and preserves
these original splits/results; it is an exploratory reuse of inspected test sets.

## Data and attribution

Download the original archives from the UCI Machine Learning Repository:

| Dataset | Experimental provenance | Rows / inputs | Source |
| --- | --- | --- | --- |
| Airfoil Self-Noise | NASA aerodynamic/acoustic wind-tunnel measurements | 1503 / 5 | [Brooks, Pope & Marcolini, 1989](https://archive.ics.uci.edu/dataset/291/airfoil+self-noise), DOI 10.24432/C5VW2C |
| Concrete Compressive Strength | Laboratory strength measurements for concrete mixes at different ages | 1030 / 8 | [I-Cheng Yeh, 1998](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), DOI 10.24432/C5PK67 |
| Yacht Hydrodynamics | Delft ship hydromechanics experimental data, 22 hull geometries | 308 / 6 | [Gerritsma, Onnink & Versluis, 1981](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics), DOI 10.24432/C5XG7R |

UCI lists all three under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The concrete archive also requests retention of credit to Prof. I-Cheng Yeh and
*Modeling of strength of high performance concrete using artificial neural networks*,
Cement and Concrete Research 28(12), 1797–1808 (1998).
Raw data are downloaded into an ignored local cache, not bundled under this
repository's MIT license. The archive/member hashes, source URLs, data quality
checks and numerical transformations are recorded. Published predictions are derived
from these attributed datasets; dataset rights remain under their source licenses.

## Splits and preprocessing

- Seeds: 0, 1, 2. Each controls both a grouped split and a search initialization;
  variation therefore combines split and optimizer variability.
- GroupShuffleSplit holds out 20% of groups for test, then 25% of the remaining
  groups for validation (approximately 60/20/20). Seeds are `seed` and `seed+1000`.
  Row fractions differ because groups have different sizes and counts are rounded.
- Airfoil: group by angle, chord, free-stream velocity and boundary thickness;
  the frequency sweep stays in one partition. These are inferred operating-condition
  groups, not recovered laboratory batch IDs.
- Concrete: group by all seven ingredient quantities, across ages. Exact duplicates
  are retained, but cannot cross partitions. Equal recipes identify the same group;
  unrecorded source/lab dependence and near-duplicate recipes may remain.
- Yacht: group by all five geometry columns, across Froude numbers. This tests new
  hull geometries; there are only 22 independent groups in the whole dataset.
- All rows must be finite and match the official shape. No imputation, target
  filtering, duplicate removal, feature selection or target-driven grouping.
- Convert angle degrees to radians, concrete age days to seconds, and MPa to Pa.
  Other fields use their published units. Then use train RMS feature scaling and
  train standard-deviation target scaling without centering. These stored scaling
  factors are **dimensionless numerical multipliers**; the transformed physical
  variables retain their dimension vectors. They are not quantities with units used
  to declare every variable dimensionless. All conversions are reversible.
- RMSE/MAE are reported in the source target scale (dB, MPa, or the published
  yacht coefficient scale). R² uses the held-out target mean. NRMSE uses train std.

## Models and fixed budget

Full DimenSR uses `examples/real_data_config.json`: population 64, 10 neural/GP
epochs, 3 GP generations per epoch, 6 final NSGA-II generations, at most 21 nodes
and depth 7, 4000 unique evaluation ceiling, 2 coefficient starts of up to 40
optimizer evaluations. Operators are `add, sub, mul, div, square, sqrt, log, exp`.
Other parameters retain SearchConfig defaults. This is a deliberately bounded pilot;
the loop schedule need not consume the whole evaluation ceiling.

Constant types are the deduplicated set `{dimensionless, target units,
target units - each feature's units}`. This permits dimensional intercepts and slopes.
Coefficients are still train-fitted; their type list is fixed from metadata before
search. Allowing dimensional coefficients broadens the physically admissible family
and does not prove causal or mechanistic correctness.

Airfoil dB is treated as a dimensionless logarithmic response, not pressure in Pa.
Dimension rules alone do not encode all semantics of decibel arithmetic. Yacht
features and response are already dimensionless, so it cannot establish a gain from
unit constraints. Its redundant `no_units` variant is omitted.

The `no_units` variant uses the same search settings and constant token count, with
all unit vectors zero. Actual evaluated trees and time may differ. This is not an
equal-compute ablation, nor a stand-in for PhySO or DSO.

Baselines: train-mean DummyRegressor; StandardScaler + Ridge with alpha chosen from
0.01, 0.1, 1, 10, 100 by validation MSE; RandomForestRegressor with 200 trees,
min_samples_leaf chosen from 1, 3, 5 by validation MSE, one CPU worker. They share
the exact partitions and initial input transformations. They are prediction
comparators, not dimension-constrained symbolic methods. No model refits on validation.

## Checks, reporting and interpretation

Test labels are never sent to fitting, coefficient optimization, scaling or model
selection. Score the selected model once; preserve all failures, low scores and
invalid predictions. A test-domain failure gets no score rather than dropping rows.
The pilot configuration is not revised based on test results.

An experiment-only Evaluator wrapper reconstructs each new evaluated tree, including
neural seeds and GP offspring, to validate every operator and terminal dimension,
including candidates later rejected numerically. Repeated cached trees were checked
on their first call. This audit does not alter search or silently repair trees.
Saved model predictions are checked after JSON reload. Dedicated tests also exercise
generation, mutation and crossover using each dataset's metadata.

Record native-scale train/validation/test RMSE, MAE and R²; test predictions; model
trees and coefficient dimensions; node counts; actual evaluation counts; constant
fit work; timings; hashes; split row indices and scales; package versions; warnings.
Report all three seeds with descriptive mean and sample standard deviation. These
overlapping splits are not independent datasets and do not support significance
claims. Runtime includes auditing and serialization, and is descriptive on this machine.

Feasibility means that the full workflow can produce finite, useful held-out
predictions and interpretable formulas under these conditions. Compare to the mean
baseline and to stronger predictors; poor results still inform feasibility. Neither
good R² nor zero unit violations establishes physical law recovery, novelty, global
domain safety, extrapolation performance, or superiority to established SR systems.

## Reproduction

```bash
python -m pip install -e ".[experiments]"
python -m dimensr.real_data --prepare-only
python -m dimensr.real_data
```

The same command resumes completed per-method files only if source/protocol/split
fingerprints match. An interrupted single fit restarts; it has no optimizer checkpoint.
Use a fresh output directory for changed settings. Raw cache lives under
`outputs/real_data/cache`; results and portable models live under `results/real_data`.
Do not change the source or protocol in that output directory after freezing it.
