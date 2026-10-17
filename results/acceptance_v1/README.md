# Acceptance pilot v1

Protocol was frozen before the first benchmark fit. See
[`ACCEPTANCE_PROTOCOL.zh-CN.md`](../../docs/ACCEPTANCE_PROTOCOL.zh-CN.md)
for sources, adaptations, thresholds, and limits.

Artifacts:

- `protocol.json`: planned jobs, configurations, source/data hashes.
- `provenance.json`: pinned upstream archives, environment and DSO compatibility probe.
- `data/*_fit.npz`: training and validation arrays; `*_holdout.npz`: separate test/OOD/probe arrays.
- `runs/`: every planned job, including failures and timeouts.
- `models/`: DimenSR portable JSON models, native selection and full search history.
- `summary.json`: post-fit numeric and symbolic audit of all exported candidates.
- `verification.json`: independent metric, dimensional and selection verification.
- `acceptance.json`: machine-readable decisions against the frozen pilot gates.
- `reproducibility.json`: separate rerun check, if completed.
- `scoring_amendment.json`: bounded symbolic-scoring adapter, its hash and frozen run hashes.
- `symbolic_diagnostics.json`: completed versus indeterminate symbolic comparisons.
- `software_checks.json`: local test count, command and test-file hashes.
- `repeat/`: independently refitted model and run record for the preselected repeat.

Generated logs and per-run scoring checkpoints are not versioned. Published `runs/`
are never overwritten by the dispatcher; to repeat an experiment use a new output
directory. `prepare` rejects an existing protocol. Source or data changes require a
new study directory. The library version was not retuned using this test suite.

## Reproduction

Install this repository and its test dependencies. Obtain the unmodified upstream
PhySO checkout at `bfbfa88d5def09d3d0cd47f3e2252b5a25836721`, then install its
documented dependencies in an isolated environment. Make that checkout importable
through `PYTHONPATH`. The original local run used Python 3.12 with extra PhySO-only
dependencies in a separate directory; exact versions are recorded in provenance.

Set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, `MPLBACKEND=Agg`,
and a writable `MPLCONFIGDIR`. From the repository root:

```bash
python -m pytest -q
python examples/acceptance_benchmark.py prepare --out outputs/acceptance_repeat
python examples/acceptance_benchmark.py run --out outputs/acceptance_repeat --workers 4
python examples/score_acceptance.py --out outputs/acceptance_repeat
```

The scorer refuses to proceed until every planned fit has a terminal result. It
never re-selects models by test performance. The separate scoring adapter delegates
numeric scoring to the frozen runner and isolates each symbolic comparison in a
subprocess with a 10-second limit (30 seconds for worker startup). Timeouts/errors
are recorded as indeterminate, excluded from symbolic successes, and retained on
resume. This reporting-only amendment was added before holdout scoring; training,
selection, data and numeric thresholds were unchanged. Per-comparison checkpoints
are in ignored `scored/symbolic_cache.json`; the final summary retains all verdicts.
To re-verify the committed study and render its report:

```bash
python examples/report_acceptance.py
python examples/reproduce_acceptance.py
```

No DSO run is included: the pinned repository requires TensorFlow 1.14 and a legacy
compatible Python environment unavailable on this host. Do not label an internal
DimenSR ablation as DSO. This remains an explicit gap for full comparative acceptance.
