# Reproducible experiment protocol

## Material Passport

- Deliverable: implementation validation and a proposed comparative protocol.
- Inputs: supplied algorithm proposal, five papers, public PhySO/DSO revisions listed in SOURCES.md.
- Evidence status: implemented mechanisms are testable; only measurements saved in this repository's result files count as completed experiments.
- Scope: computational symbolic regression, synthetic data; no human participants or external model API calls.

## Included checks

Four generated tasks cover product units (`force`), a fitted coefficient and square (`kinetic`), a polynomial (`nguyen1`), and a transcendental-plus-polynomial expression (`trigonometric`). Input draws and fitting noise are deterministic per seed. Separate IID test and OOD test sets are generated without exposing them to `fit`. OOD ranges are intentionally disjoint from the training range.

`numerical_recovery` means NRMSE below `1e-6` on the IID set, OOD set, and fixed exact-zero/boundary probes. Random samples alone almost surely miss isolated singularities such as division by zero. Probes are evaluated only after fitting. This remains a numerical diagnostic, not symbolic equivalence or recovery on an entire domain. RMSE, NRMSE, R², raw tree/size, fitted expression, validation RMSE, unique evaluation count, actual residual calls and reported optimizer evaluations, seed and wall time are saved per run. Invalid domains and failed searches are retained as failures.

```bash
dimensr benchmark --config examples/quick_config.json --seeds 0 1 2 --output outputs/quick.json
dimensr benchmark --config examples/quick_config.json --cases kinetic --seeds 0 1 2 --variants full no_rnn no_gp no_units no_sd no_rad --output outputs/ablation.json
```

These commands are a development-scale run. Ablation flags retain the same maximum unique-evaluation ceiling, but actual consumed evaluations and wall times can differ; current output is not an equal-compute causal comparison. `no_gp` removes both GP loops, while preserving final Pareto/validation selection. `no_units` makes all unit metadata dimensionless. The neural initialization and data seeds are shared when applicable.

## Publication-scale evaluation still required

1. Freeze a versioned dataset suite: Nguyen, Feynman/physical equations and the actual application data. Define sample size, noise distribution, domain, unit convention and permitted operators before running.
2. Separate training/validation/test by physical sample, experiment batch or time as appropriate. Use nested validation to select complexity weights and unit catalogs. Do not choose settings using test recovery.
3. Compare full DimenSR to DSR, neural-seeded GP/DSO, PhySO, ordinary GP and a strong established SR baseline. Run upstream repositories in their supported isolated environments, using pinned commits and unchanged source unless an adaptation is documented.
4. Run all five module removals plus an alternative simple complexity penalty. Add an original Raymond-style heuristic comparison if evaluating the Rademacher change as the main contribution.
5. Use enough independent seeds (initial target: 20 or more per task, adjusted after pilot variability) and two budget views: equal unique evaluations and equal elapsed CPU time. Record constant-optimization costs and hardware; neither unique trees alone nor a shared upper ceiling ensures equal compute.
6. Report task-level and seed-level distributions: recovery rates with binomial intervals, paired NRMSE differences, complexity and runtime. State paired units, dependence, multiple-comparison handling and failure denominators. Avoid treating different seeds from one task as independent tasks.
7. Inspect recovered equations, domains and dimensional validity. Compare algebraic equivalence with explicit assumptions; use out-of-domain numerical checks only as supporting evidence.

Upstream baselines have not been run by the development test suite. Do not relabel the internal `no_rnn` or `no_units` variants as PhySO or DSO. Their architectures and search schedules differ.
