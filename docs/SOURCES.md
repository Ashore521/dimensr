# Source-to-implementation map

Inspection date: 2026-10-17. The source set was supplied by the project owner. Local paper text and the design document's equation images were inspected; this is not an exhaustive novelty search.

## Public repositories

| Source | Inspected revision | License | Inspected mechanisms |
| --- | --- | --- | --- |
| [PhySO](https://github.com/WassimTenachi/PhySO) | `bfbfa88d5def09d3d0cd47f3e2252b5a25836721` | MIT | README, `physo/physym/dimensional_analysis.py`, `physo/learn/loss.py` |
| [DSO](https://github.com/dso-org/deep-symbolic-optimization) | `8348d5b08d1eef6170fdbfd222a492ded990ea12` | BSD-3-Clause | README, `dso/dso/gp/gp_controller.py`, `dso/dso/policy_optimizer/pg_policy_optimizer.py`, `dso/setup.py` |

DSO's inspected installation specifies `tensorflow==1.14`, `numpy<=1.19`, and `numba==0.53.1`. DimenSR therefore implements the referenced concepts independently using current PyTorch instead of combining incompatible dependency stacks. No claim is made that its behavior or benchmark scores reproduce the upstream implementations exactly. The inspected raw files were retrieved from default-branch URLs; the revision endpoints were checked during the same inspection session.

For the subsequent acceptance pilot, full archives at these exact commits were
retrieved and relevant source/config/data files verified byte-for-byte against
the archives. Archive hashes, package versions and the DSO compatibility probe
are recorded in `results/acceptance_v1/provenance.json`; benchmark definitions and
paper-to-protocol differences are documented in [the acceptance protocol](ACCEPTANCE_PROTOCOL.zh-CN.md).

## Literature and proposal

| Evidence | Adopted idea | Implementation and differences |
| --- | --- | --- |
| User's “符号回归方法设计方案” | RNN–GP, dimensional constraints, top-30% candidate pool with a minimum 50, mixed complexity, NSGA-II/SD, weighted final choice | Complete workflow in `regressor.py`; insufficient candidates are reported; final weighting uses normalized validation error |
| Petersen et al., ICLR 2021, *Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients* | Tail-reward policy gradients, sequential expression generation | `policy.py`; GRU with explicit units/budget context and grammar masks |
| Mundhenk et al., NeurIPS 2021, *Symbolic Regression via Neural-Guided Genetic Programming Population Seeding* | Neural population seeding and GP feedback | `regressor.py`, `genetics.py`; separate on-policy gradient and off-policy imitation, bounded typed subtree operations |
| Tenachi, Ibata & Diakogiannis, 2023, *Deep Symbolic Regression for Physics Guided by Units Constraints: Toward the Automated Discovery of Physical Laws*, ApJ 959:99 | Physical units restrict expression generation | `units.py`, `expression.py`, `grammar.py`; exact rational exponents and finite-catalog completion costs, not PhySO's full live unit-inference machinery |
| Raymond et al., 2019, *Genetic Programming with Rademacher Complexity for Symbolic Regression* | Functional complexity alongside fit error | `complexity.py`; v0.2 uses an explicit local Jacobian output-ball family (v0.1 used finite coefficient perturbations), distinct from the paper's output-binarization heuristic |
| Zhang et al., CEC 2025, *A Multi-Objective Genetic Programming with Size Diversity for Symbolic Regression Problem* | Neighborhood size-density penalty | `pareto.py`; density adjusts crowding inside a rank, while complexity remains the second objective; the paper replaces that objective with SD |

The proposal's formula image writes `C_mix = alpha1*N_op + alpha2*N_var + alpha3*R_rad`; its prose defines the first two counts as symbol count and function-kind count. DimenSR follows those prose definitions and names them `node_count` and `distinct_operator_count` to avoid ambiguity.

## Verified source links

- DSR paper link recorded by upstream: https://openreview.net/forum?id=m5Qsh0kBQG
- Neural-guided GP paper link recorded by upstream: https://proceedings.neurips.cc/paper/2021/hash/d073bb8d0c47f317dd39de9c9f004e9d-Abstract.html
- PhySO paper DOI, present in the supplied PDF: https://doi.org/10.3847/1538-4357/ad014c
- Size-diversity paper DOI, present in the supplied PDF: https://doi.org/10.1109/CEC65147.2025.11042993
- Raymond et al.: title, author list and year checked against the supplied PDF; no DOI is asserted here.

The standalone combination and the local sensitivity term are proposed design choices. The supplied sources establish provenance, not novelty priority. A broader related-work search and direct comparative experiments are necessary before making a publication-level novelty claim.
