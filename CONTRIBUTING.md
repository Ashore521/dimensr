# Contributing

Install `.[dev]`, run `python -m pytest -q`, and include focused tests for behavior changes. Keep source under `src/dimensr` and update the algorithm specification when a search objective or unit rule changes.

For a bug, include the package/environment versions, seed, complete SearchConfig, minimal synthetic data, unit metadata and traceback. Avoid uploading private experimental data. For a performance claim, attach per-seed results, data splits, actual evaluation counts, constant-fit work and runtime. All reported baselines must have been run; mark unavailable measurements explicitly.

New operators must specify their mathematical domain, unit rule, NumPy evaluation, SymPy export and tests. No silent protected operators or validation-dependent search fitness should be introduced.

