# Frozen boundary/search comparison

54 fits: 6 tasks × 3 seeds × 3 variants (frozen 0.2, boundary-only, full 0.3).
All fits completed. See the [Chinese report](../../docs/BOUNDARY_SEARCH_RESULTS.zh-CN.md)
and [pre-fit protocol](../../docs/BOUNDARY_SEARCH_PROTOCOL.zh-CN.md).

- `protocol.json`: pre-fit source/data/config hashes and environment.
- `data/`: separate fitting and holdout arrays. Fitting includes unlabelled domain points.
- `models/`, `runs/`: raw trees, candidates, work counts, provenance and outcomes.
- `scores.json`: strict intermediate-node scoring across IID/OOD/boundary/stress inputs.
- `verification.json`: independent serialized-tree replay and declared-probe audit.
- `reproducibility.json`, `replay/`: one exact fixed-seed replay, excluding timing only.
- `software_checks.json`, `checks/`: local tests, packaging and installation evidence.
- `posthoc_*.json`: disclosed post-hoc diagnostics, not changes to registered scoring.

Use the repository version whose normalized source hashes match `protocol.json`.
`python examples/verify_boundary_study.py` rechecks the saved evidence.
For a new run, use a separate `--out` directory and run prepare, run, score, verify.
Finite domain probes do not establish continuous-domain safety; an explicit
extrapolation pole counterexample is included. This is not an equal-compute or
blind-task comparison.
