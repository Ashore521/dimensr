# Frozen interval/complementary-search study

Six reused formula tasks, three fresh seeds (71, 82, 93), and three variants:
frozen 0.3, new search only, and full 0.4. All variants receive identical declared
boxes spanning the interpolation and extrapolation regions. See the
[pre-fit protocol](../../docs/INTERVAL_SEARCH_PROTOCOL.zh-CN.md).

All 54 fits completed successfully. Strict recovery was 12/18, 16/18 and 18/18,
respectively. See the [Chinese result report](../../docs/INTERVAL_SEARCH_RESULTS.zh-CN.md).

- `protocol.json`: source, data, configuration and environment frozen before fits.
- `data/`: separate fitting and holdout arrays; fitting includes unlabelled domain inputs.
- `runs/`, `models/`: all run outcomes, raw trees and saved Pareto candidates.
- `scores.json`: registered numerical recovery and finite-probe metrics.
- `verification.json`: independent raw-tree replay, units/budgets/probe checks, and interval audits.
- `reproducibility.json`, `replay/`: the preselected exact single-run replay.
- `prior_counterexample_audit.json`: interval screening of the earlier known pole.
- `software_checks.json`, `checks/`: tests, packaging and isolated-install evidence.

Run `python examples/verify_interval_study.py` at the matching source version to
verify results. Use a new output directory for new experiments. This study is not
blind-task discovery or equal-compute benchmarking. `enclosed` is a numerical
interval-screening result, not a formal mathematical certificate. Coupled-domain
points outside the declared rectangle receive finite-point checks only.
