# Algorithm specification

## Scope

DimenSR implements the supplied proposal as a two-stage search. The implementation is independent of PhySO and DSO code. It retains their central ideas while making the search grammar, feedback objective and validation boundary explicit. The working hypothesis is that physically valid exploration plus diversity-aware selection and a local sensitivity penalty can improve the error/complexity tradeoff. This remains a hypothesis until adequately powered comparisons are run.

## 1. Typed expressions and bounded grammar

Each node has a seven-component vector of rational SI exponents, ordered `(L,M,T,I,Theta,N,J)`. Addition/subtraction require equal input units. Multiplication adds exponents; division subtracts them. Square doubles exponents, square root halves them. `sin`, `cos`, `exp`, `log` require dimensionless inputs and return dimensionless outputs. A free constant has an explicitly declared unit vector; the default is dimensionless. All descendants and root dimensions are checked on construction.

Let `B` be terminal units, target units, zero and user-supplied `extra_units`. The finite catalog is:

`U = B ∪ {−2u, −u, 2u, u/2 : u∈B} ∪ {u+v, u−v : u,v∈B}`.

The closure is applied once, not recursively. The implementation rejects catalogs above 256 types. An expression may be physically valid yet outside `U`; this is a deliberate search-space limitation, not a proof that no physical model exists. `extra_units` explicitly expands the catalog. Operators only admit productions whose child and result units belong to `U`.

For every unit type and remaining depth, dynamic programming stores the minimum node count of a completing tree. Generation masks any token with no production fitting the remaining node and depth budgets. After selecting a token, a second policy head scores its legal child-unit splits using the GRU state, selected token, child dimensions and minimum completion costs. Both decisions are replayed in the policy likelihood. Without a neural policy both are sampled uniformly. Child budgets reserve the minimum completing size of later siblings. This guarantees termination and valid dimensions within the finite grammar. Allocation remains sequential; a fully learned subtree-size allocator is not implemented.

GP mutation replaces a selected subtree with a generated tree of the same units and compatible budget. Crossover draws an eligible, structurally different donor subtree of equal units without exceeding node/depth limits. Unchanged and donated coefficients are inherited as fitting starts. An additive mutation can append an eligible typed seed term. Seen structures trigger a bounded retry (`variation_attempts=4`). These operations preserve units by construction. `use_units=False` is an explicit ablation that marks all units dimensionless and deduplicates constant types.

## 2. Neural seeding and GP feedback

The GRU receives the previous token, parent token, required units, normalized remaining node budget and depth budget. It predicts a masked distribution over tokens in prefix order. It is initialized from a seed and learns online from the supplied regression task; it is not pretrained on an external expression corpus. The first batch therefore uses an untrained stochastic neural policy.

Training reward is `R(f)=1/(1+RMSE_train(f)/s_y)`, where `s_y` is the training target standard deviation, with an RMS/one fallback for constant targets. Constants minimize training residuals only. Trees structurally affine in all constants use a column-scaled linear least-squares solve (bounded if necessary); others use bounded nonlinear least squares with Jacobian scaling and deterministic restarts. Direct constant offsets in log/sqrt/div receive domain-aware starts, with bounded multiscale fallback if initial starts are invalid. This does not solve arbitrary domain feasibility. An exact affine unconstrained optimum that violates physical constraints may be rejected; a general constrained affine optimizer is not implemented. No scaling is added outside the tree.

`linear_seeds=True` introduces a small library of legal weighted inputs, unary/ratio terms and balanced additive structures. Seeds obey the same grammar, units, operators, depth and size bounds as random trees. Their coefficients use training data only. Inverse-unit normalizing constants must be explicitly supplied; the optimization-study runner declares these so expressions such as `log(c_time_inverse * age)` are available. Dimensional metadata on unknown coefficients is not evidence of a physical mechanism.

Version 0.3 additionally enables `structured_seeds`. A bounded library (at most 512 retained features, at most 9 nodes each) includes unary compositions, small products/ratios/sums, and offset/scale log or square-root terms. Training correlations shortlist 48 features; a width-three sparse beam fits at most three terms, optionally over one of at most 64 eligible common denominators. Coefficient units must match the requested numerator/output units. Initial coefficients and rankings use training targets only. Final trees are replayed through the grammar and the ordinary coefficient/domain evaluator. At most `min(2*population_size, floor(stage_one_limit/4))` selected proposals enter that evaluator. Feature previews and linear subproblems are extra work, recorded separately; the unique-tree ceiling is not a total compute budget. This finite library does not cover arbitrary compositions or parameter sharing.

For policy-sampled expressions, `b` is the empirical `(1−epsilon)` reward quantile. The loss contains:

`L_RSPG = − mean[ I(R>=b) (R−b) log p_theta(tree) ] / epsilon`.

Entropy regularization uses the sum of conditional token and unit-split entropies. Masks, contexts, legal production features and chosen production indices are replayed for gradients.

Each new neural batch joins a retained fraction of the previous population. Stage-one tournament selection prefers lower training RMSE (tree size breaks ties); survival also reserves representatives across tree sizes. Diverse elites from the training archive contribute a **separate length-normalized imitation loss**. They are not treated as on-policy samples in the policy-gradient term. The total objective is `L_RSPG − beta*entropy + lambda*L_imitation`. Gradients are clipped. Neural duplicates are still included in the on-policy batch; retry filtering is confined to GP proposals.

## 3. Candidate screening and mixed complexity

All evaluated trees are cached by immutable structure, including terminal units and constant-token types, after ordering commutative add/multiply children and their coefficients. No cancellation or domain-changing simplification occurs. Constant values do not change the structural cache key. Invalid fits are cached, but explicit evaluator refinement can retry a cached tree. `max_evaluations` caps new structure evaluations, not raw proposals or coefficient work. When NSGA-II is enabled, `selection_budget_fraction=0.25` reserves that fraction of the unique-evaluation ceiling for stage two. Fixed generation limits can still leave budget unused. `polish_candidates=8` adds bounded training-only coefficient refinement before final selection; it does not consume new-structure slots.

Retain `min(A, max(min_candidates, ceil(candidate_fraction*A)))`, where `A` is the number of unique valid trees. Fill the pool with an accuracy/size-diverse subset, training Pareto candidates and remaining accuracy-ranked candidates, in that order. Defaults are 30% and 50. If fewer than 50 are found, diagnostics and a warning expose this. Not every non-dominated model is guaranteed a slot when the requested pool is small.

The mixed objective is:

`C(f) = alpha1 * node_count + alpha2 * distinct_operator_count + alpha3 * R_scaled`.

This follows the proposal's textual definitions: the second count denotes function/operator kinds, not the number of input variables, despite the original formula's `N_var` notation. Complexity counts the unsimplified tree so it reflects the actual search program.

### Local Rademacher term

Version 0.2 uses the Jacobian of training predictions with respect to the fitted constants, in normalized output units. Finite differences use central or feasible one-sided steps. Nonzero columns are normalized and an SVD yields an orthonormal column-space basis `Q`; singular values below `1e-6` of the largest are discarded.

Define the explicit local **linearized output family** `G_f = {Q a : ||a||_2 <= rho*sqrt(n)}`. For fixed training-row Rademacher sign draws `sigma_k`, its empirical estimate has the closed-form supremum:

`R_local = rho/sqrt(n) * mean_k ||Q.T sigma_k||_2`.

All candidates share the same sign draws, using training rows only. Redundant or rescaled coefficient coordinates give the same subspace at regular points, within finite-difference/rank tolerances. This avoids the old finite-perturbation penalty multiplying just because one coefficient is written as a product of two. No constants or a zero Jacobian gives zero. An unrecoverable numerical Jacobian receives the conservative maximum `rho`.

For selection, `R_scaled = sqrt(n)/rho * R_local`; if `rho=0`, both terms are zero. This makes the default weight of 2 numerically meaningful alongside node counts. Positive `rho` changes the raw family's radius but cancels in the scaled penalty; tune `complexity_weights[2]` for penalty strength. Both raw `rademacher` and `rademacher_complexity` are saved. `perturbations` is retained only to load 0.1 configurations; it has no effect on the new estimator.

This term measures local independent prediction directions, not absolute coefficient sensitivity. It is not invariant at singular parameterizations and can miss higher-order changes where the Jacobian vanishes. It does not measure full structural capacity or certify generalization for the searched nonlinear models. It also differs from Raymond et al.'s output-binarization heuristic. The 0.1 results retain their original finite-perturbation definition; raw mixed complexities across versions are not directly comparable.

## 4. NSGA-II and final selection

Stage two starts from screened candidates and evolves against `(training_NRMSE, C)`. Since the training scale is fixed, NRMSE and RMSE have the same dominance order. Nondominated sorting and normalized objective crowding follow standard NSGA-II.

For tree size `s`, `SD(s,l) = count(|size_i−s|<=l)/(2l+1)` over the current combined population. Within a front the crowding score becomes `crowding / (1 + SD/mean(SD))`; infinite boundary scores stay infinite. Ties prefer smaller SD and then stable order. Rank always takes precedence. This is a deliberate extension of the proposal's `(rank, crowding, SD)` tie-breaking: it makes SD affect interior truncation even without exact crowding ties. It is **not** an exact reproduction of Zhang et al., whose SD replaces the second objective. `use_size_diversity=False` restores ordinary crowding ordering.

The final training Pareto front is computed over screened seeds and evaluated stage-two offspring after bounded coefficient polishing. Validation is evaluated only after the search ends. Select a finite-domain, configured-output-bound-compliant front member minimizing

`validation_RMSE/s_y + selection_complexity_weight * C/max_nodes`.

The fixed normalization avoids unit-dependent raw RMSE/complexity addition and shifting front-wise min–max scales. If all front members fail on validation domains, fitting fails explicitly. No coefficient refit uses validation. A held-out test set is not part of `fit`.

## 5. Physical bounds and diagnostics

`output_bounds=(lower, upper)` optionally declares output limits in the supplied target's numeric units; either end can be `None`. Candidate acceptance checks training inputs, `domain_probe_count` deterministic convex combinations of training input rows, and (since 0.3) bounding-box endpoints, midpoints and zero where in range. Coordinate checks are always retained; Cartesian combinations are capped at 256 deterministic points. Optional `fit(input_bounds=...)` declares a finite closed box containing the training inputs; `fit(X_domain=...)` adds permitted unlabelled points. Initial domain repair also uses these unlabelled points, while coefficient residuals use training labels only. Final selection checks validation compliance. `predict` raises for nonfinite or out-of-bound predictions; experiment scoring evaluates the original tree directly so violations remain visible and are never clipped or omitted.

Every intermediate node is required to be finite; an outer finite value cannot hide an invalid inner division or logarithm. No algebraic cancellation changes this rule. Finite probes do not prove safety over a continuous domain or outside the checked inputs. Boxes and convex combinations may be unsuitable for categorical/manifold-constrained inputs; disable `boundary_probes`, set `domain_probe_count=0`, omit `input_bounds`, and supply permitted `X_domain` points for those tasks. Explicit bounds always activate their box checks. No automatic monotonicity constraints are inferred. Domain diagnostics retain input bounds, probe counts and a point-set hash. `domain_rejections` counts rejected coefficient trials, not distinct trees.

Progress records phase-specific new evaluations, neural proposal/valid/unique counts, best current expression, sizes and optimization metrics. Final diagnostics include GP proposal/novel counts by operator, cache hits, linear solves, feasible-start repairs, optimizer termination statuses, coefficient residual evaluations, Rademacher prediction evaluations, and refinement counts. JSON remains an inference artifact, not a resumable generation checkpoint.

## Implementation boundaries

- CPU reference implementation; no vectorized GPU generation, multiprocessing or optimizer checkpoint resume.
- Finite operator list and unit catalog; no arbitrary symbolic powers, shared coefficients or learned unit conversion.
- Full NSGA-II sorting of bounded populations uses quadratic pairwise comparisons. Archive first-front extraction uses an O(N log N) two-objective scan; the structural cache still grows with the evaluation budget.
- Fixed-seed replay is tested within one environment; floating point/library changes can alter search paths.
- Selection hyperparameters must be tuned inside training/validation protocols. A validation score is not an unbiased final test score.
- Dimensional consistency is necessary for physical plausibility but does not imply causal or physical correctness.
