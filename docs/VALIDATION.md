# Validation record

Validation date: **2026-10-17**. These are actual local runs, not expected scores. No upstream comparative benchmark or hosted CI run is claimed.

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

**29 tests passed.** Coverage includes invalid dimension rejection; rational/half-power units; 60 generated trees and their typed mutation/crossover variants; completion and depth/node bounds; impossible target units; dimensional constants; strict numerical domains; policy parameter updates and replay masks; unchanged global RNG during initialization; nondominated sorting and duplicate objectives; size-density effects; local Rademacher sensitivity; train-only constant fitting; unchanged search when validation labels change; seed reproducibility; budget limits; ablations; invalid inputs; failed-refit state; model JSON round trips including fractional units; tampered metadata rejection; CLI CSV training/prediction; and zero-point probes that detect singularities.

Editable installation succeeded. Source distribution and wheel construction succeeded. The release wheel is additionally installed into a separate target directory and checked with the physical demo during packaging. GitHub's configured Linux/Windows CI matrix has not been run locally and must be checked after publication.

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

The result files include raw model trees and coefficients for auditing numerical domains. In total, **42 benchmark runs** are recorded. Data are synthetic and noiseless, with 160 rows per generated dataset before the internal validation split. There is no application dataset in the supplied project, no large benchmark suite, no noise study and no direct PhySO/DSO comparison. Follow [EXPERIMENTS.md](EXPERIMENTS.md) before making stronger accuracy, robustness, novelty or generalization claims.
