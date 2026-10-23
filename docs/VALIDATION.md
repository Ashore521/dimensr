# Validation record

Version **0.4.0**, 2026-10-23: **89 tests passed**. The new
[interval/complementary-search study](INTERVAL_SEARCH_RESULTS.zh-CN.md) completed
54/54 fits. Strict recovery was 12/18 for contemporaneous frozen 0.3, 16/18 for
search-only changes, and 18/18 for full 0.4 on six reused noiseless tasks. Independent
verification covered 54 models and 716 saved trees; all 201 full-0.4 saved tree
records passed declared-box interval screening. One preselected replay matched all
non-timing state and predictions exactly. The prior known pole was unresolved with
32 boxes and produced an actual witness after 80 boxes with allowance 128.
Numerical interval screening is conservative and is not a formal certificate.

Version **0.3.0**, 2026-10-23: **79 tests passed**; wheel/sdist built and an isolated
wheel installation passed the declared-domain CLI and save/load smoke check.
The [boundary and search study](BOUNDARY_SEARCH_RESULTS.zh-CN.md) completed 54/54
fits and independently replayed 941 saved trees. Strict recovery was 12/18 for
0.3 versus 2/18 for frozen 0.2. Registered finite-probe final domain failures were
0/18 versus 5/18; a separate extrapolation pole counterexample remains documented.
One fixed-seed repeat reproduced all non-timing model state and predictions.
Historical results below retain their original versions and scopes.

Historical development validation date: **2026-10-17**. These are actual runs,
not expected scores. A subsequent **2026-10-18** [acceptance report](ACCEPTANCE_RESULTS.zh-CN.md)
adds a limited, adapted-budget comparison with pinned upstream PhySO; DSO and
publication-scale comparative validation remain incomplete. The initial release
passed all four [GitHub CI jobs](https://github.com/Ashore521/dimensr/actions/runs/39024193283);
later checks are available on the repository Actions page.

The acceptance tooling suite passed **67 tests in 10.54 seconds** on 2026-10-18
(48 existing tests plus 19 data-isolation, raw-domain, scoring and verifier checks).
All 126 planned fits completed. One preselected independent repeat reproduced all
non-timing model/run state and predictions exactly. See the machine-readable
`results/acceptance_v1/software_checks.json` and `reproducibility.json` for scope
and hashes. These engineering checks do not imply that the algorithm passed every
scientific acceptance threshold; those decisions are in the separate report.

## Environment and commands

Windows 11, Python 3.12.10, NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1+cpu. CLI runs use one PyTorch CPU thread. Some development commands ran concurrently, so wall times are diagnostics rather than controlled runtime comparisons. Full per-run configurations and environment strings are stored in the result JSON files.

```bash
python -m pytest -q
dimensr demo --case kinetic --config examples/quick_config.json --output outputs/kinetic_model.json
dimensr benchmark --config examples/quick_config.json --seeds 0 1 2 --output results/quick_benchmark.json
dimensr benchmark --seeds 0 1 2 --output results/default_benchmark.json
dimensr benchmark --config examples/quick_config.json --cases kinetic --seeds 0 1 2 --variants full no_rnn no_gp no_units no_sd no_rad --output results/ablation_benchmark.json
python -m build --no-isolation
```

## Software checks

The discovery/ablation tables below are historical **0.1** results and retain their original artifacts. Version 0.2 adds coefficient-domain repair, affine fitting, effective typed variation, stage budgeting, learned unit splits and tangent-family complexity checks. Its separate real-data comparison is documented in [optimization results](OPTIMIZATION_RESULTS.zh-CN.md); historical scores must not be attributed to the new algorithm.

The 0.2 development checks comprise the existing suite plus 11 new tests (48 total): feasible square-root offsets, multiscale affine coefficients, commutative coefficient identity/domain preservation, explicit physical rejection, non-starved NSGA-II evaluation, eight-variable affine recovery, inherited coefficients, redundant-parameter Rademacher invariance, fast first-front equivalence, learned unit-split replay, and scheduling-amendment fingerprint protection. Before the new study, all 42 old real-data metric records and 15 saved models were independently reverified unchanged.

**Final 0.2 suite: 48 passed in 5.75 seconds.** The 0.2.0 wheel was built and installed without dependencies into a separate directory; importing that installed copy and predicting force for mass 3 and acceleration 2 returned 6.0. The new study completed all 27 fits. Reloading all models reproduced training/validation/test metrics and test predictions; 732 saved Pareto models passed dimension audits. Source/config fingerprints, coefficient finiteness, node/depth/phase/evaluation limits, and training/validation physical bounds passed for all 27 selected models. NSGA-II evaluated 364–768 new structures per run. Baseline files remain unchanged.

The preset 41-node configuration achieved mean test R² 0.673369 (airfoil), 0.813731 (concrete), and 0.989896 (yacht), versus 0.423921, 0.495706, and 0.990008 originally. The improvements on the first two tasks accompany substantially more computation; the yacht task did not improve at that capacity. These are exploratory, reused-test comparisons, not equal-compute or single-component ablations. No dimensional violations were found in 138,139 within-run-unique tree evaluations, of which 93,539 had nontrivial physical units. Finite probes do not guarantee global positivity: concrete seed 0 still predicts -0.550327 MPa with 31 nodes and -0.230290 MPa with 41 nodes at original row 688. Both values remain in test scoring; `predict` rejects them under configured output bounds.

**Historical 0.1 suite: 37 tests passed**, including eight added real-data loader/split/scaling/metadata checks. Original coverage includes invalid dimension rejection; rational/half-power units; 60 generated trees and their typed mutation/crossover variants; completion and depth/node bounds; impossible target units; dimensional constants; strict numerical domains; policy parameter updates and replay masks; unchanged global RNG during initialization; nondominated sorting and duplicate objectives; size-density effects; local Rademacher sensitivity; train-only constant fitting; unchanged search when validation labels change; seed reproducibility; budget limits; ablations; invalid inputs; failed-refit state; model JSON round trips including fractional units; tampered metadata rejection; CLI CSV training/prediction; and zero-point probes that detect singularities.

Editable installation succeeded. Source distribution and wheel construction succeeded. The release wheel is additionally installed into a separate target directory and checked with the physical demo during packaging. GitHub separately checks Linux/Windows with Python 3.10/3.12; hosted status is distinct from the local run.

The kinetic CLI demo returned `0.5*mass*velocity**2`, with validation RMSE approximately `1.44e-16`. Its model can be loaded for prediction without training again.

## Discovery benchmarks

Recovery requires normalized RMSE `<1e-6` on all three independent evaluation groups: random IID, disjoint OOD, and fixed exact-zero/boundary probes. This is stricter than fitting error but remains numerical evidence, not symbolic equivalence over a continuous domain.

| Task | Quick budget recovery | Default budget recovery | Default median IID NRMSE |
| --- | ---: | ---: | ---: |
| Force `m*a` | 3/3 | 3/3 | 0 |
| Kinetic energy `0.5*m*v²` | 3/3 | 3/3 | 0 |
| Nguyen-1 `x³+x²+x` | 0/3 | 2/3 | 1.30e-16 |
| Trigonometric `sin(x)+x²` | 1/3 | 3/3 | 1.23e-16 |
| Total | **7/12** | **11/12** | — |

Default configuration is the unmodified `SearchConfig`; quick configuration is `examples/quick_config.json`. Search settings were defined before these measurements. The failing default Nguyen-1 seed (1) has excellent error on random points but contains division by `x` in the original tree and fails at `x=0`. It is counted as a recovery failure. The software deliberately preserves strict tree evaluation domains rather than silently filling the hole after symbolic cancellation.

All twelve default fitting runs completed. Thus “11/12 recovery” and “12/12 successful fitting executions” describe different outcomes. The quick failures are retained in the result files; smaller budgets can produce approximate formulas with poor extrapolation.

## Kinetic-energy module removals

| Variant, quick configuration | Recovery including domain probes |
| --- | ---: |
| Full | 3/3 |
| Without RNN | 3/3 |
| Without GP | 1/3 |
| Without units | 1/3 |
| Without size diversity | 3/3 |
| Without local Rademacher penalty | 3/3 |

The two no-GP failures are isolated zero-domain failures despite small random-point errors. The two no-units failures have appreciable approximation error. This small, simple task does **not** establish independent gains from RNN, SD or the Rademacher penalty; removing RNN is also faster here. Actual evaluations differ by variant despite the common ceiling. These are 18 exploratory ablation runs, not a statistically powered equal-budget comparison.

## Evidence files and limits

- [Quick results](../results/quick_benchmark.json): 12 complete runs.
- [Default results](../results/default_benchmark.json): 12 complete runs.
- [Ablation results](../results/ablation_benchmark.json): 18 complete runs.
- [Source hashes](../results/source_manifest.json): source/config/test fingerprints for this local release.

The result files include raw model trees and coefficients for auditing numerical domains. These files contain **42 synthetic benchmark runs**, noiseless, with 160 rows per generated dataset before the internal validation split. A further **42 public real-data runs** are documented in the [real-data report](REAL_DATA_RESULTS.zh-CN.md), including independent grouped tests and prediction baselines. All 15 real-data symbolic models and all 42 held-out metric records were independently reloaded/recomputed successfully. One full-model concrete test prediction is negative despite being dimensionally valid; this limitation is retained in the report. No large benchmark suite, noise study or direct PhySO/DSO comparison has been completed. Follow [EXPERIMENTS.md](EXPERIMENTS.md) before making stronger accuracy, robustness, novelty or generalization claims.
