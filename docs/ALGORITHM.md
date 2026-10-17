# Algorithm specification

## Scope

DimenSR implements the supplied proposal as a two-stage search. The implementation is independent of PhySO and DSO code. It retains their central ideas while making the search grammar, feedback objective and validation boundary explicit. The working hypothesis is that physically valid exploration plus diversity-aware selection and a local sensitivity penalty can improve the error/complexity tradeoff. This remains a hypothesis until adequately powered comparisons are run.

## 1. Typed expressions and bounded grammar

Each node has a seven-component vector of rational SI exponents, ordered `(L,M,T,I,Theta,N,J)`. Addition/subtraction require equal input units. Multiplication adds exponents; division subtracts them. Square doubles exponents, square root halves them. `sin`, `cos`, `exp`, `log` require dimensionless inputs and return dimensionless outputs. A free constant has an explicitly declared unit vector; the default is dimensionless. All descendants and root dimensions are checked on construction.

Let `B` be terminal units, target units, zero and user-supplied `extra_units`. The finite catalog is:

`U = B ∪ {−2u, −u, 2u, u/2 : u∈B} ∪ {u+v, u−v : u,v∈B}`.

The closure is applied once, not recursively. The implementation rejects catalogs above 256 types. An expression may be physically valid yet outside `U`; this is a deliberate search-space limitation, not a proof that no physical model exists. `extra_units` explicitly expands the catalog. Operators only admit productions whose child and result units belong to `U`.

For every unit type and remaining depth, dynamic programming stores the minimum node count of a completing tree. Generation masks any token with no production fitting the remaining node and depth budgets. For multiply/divide there may be several feasible child-unit splits: a token is sampled from the GRU, then its feasible split is chosen uniformly. Child budgets reserve the minimum completing size of later siblings. This guarantees termination and valid dimensions within the finite grammar.

GP mutation replaces a selected subtree with a generated tree of the same units and compatible budget. Crossover draws an eligible donor subtree of equal units without exceeding node/depth limits. These operations preserve units by construction. `use_units=False` is an explicit ablation that marks all units dimensionless.

## 2. Neural seeding and GP feedback

The GRU receives the previous token, parent token, required units, normalized remaining node budget and depth budget. It predicts a masked distribution over tokens in prefix order. It is initialized from a seed and learns online from the supplied regression task; it is not pretrained on an external expression corpus. The first batch therefore uses an untrained stochastic neural policy.

Training reward is `R(f)=1/(1+RMSE_train(f)/s_y)`, where `s_y` is the training target standard deviation, with an RMS/one fallback for constant targets. Constants minimize training residuals only, using SciPy bounded least squares and deterministic structure-specific restarts. No unconstrained affine scaling is added outside the tree. Ordinary mathematical operators are used; undefined or overflowing candidates are invalid.

For policy-sampled expressions, `b` is the empirical `(1−epsilon)` reward quantile. The loss contains:

`L_RSPG = − mean[ I(R>=b) (R−b) log p_theta(tree) ] / epsilon`.

Entropy regularization uses the sum of conditional token entropies. The uniform unit-split probabilities are independent of the network and omitted from the derivative, not learned implicitly. Masks and contexts are replayed for gradients.

Each batch seeds a GP population. Stage-one tournament selection prefers lower training RMSE (tree size breaks ties); parents and offspring are retained elitistically. GP elites contribute a **separate length-normalized imitation loss**. They are not treated as on-policy samples in the policy-gradient term. The total objective is `L_RSPG − beta*entropy + lambda*L_imitation`. Gradients are clipped. A new neural batch starts the next epoch, closing the feedback loop. `use_rnn=False` uses uniform feasible token proposals.

## 3. Candidate screening and mixed complexity

All evaluated trees are cached by immutable structure, including terminal units and constant-token types. Constant values do not change the structural cache key. Invalid structures are also cached. `max_evaluations` caps new structure evaluations; it does not equal the number of raw proposals or least-squares function calls.

Rank valid stage-one trees by training RMSE and retain `min(A, max(min_candidates, ceil(candidate_fraction*A)))`, where `A` is the number of unique valid trees. Defaults are 30% and 50. If fewer than 50 are found, diagnostics and a warning expose this. There is no guarantee of 50 candidates for small or tightly constrained spaces.

The mixed objective is:

`C(f) = alpha1 * node_count + alpha2 * distinct_operator_count + alpha3 * R_local`.

This follows the proposal's textual definitions: the second count denotes function/operator kinds, not the number of input variables, despite the original formula's `N_var` notation. Complexity counts the unsimplified tree so it reflects the actual search program.

### Local Rademacher term

For fitted constants `c`, fixed random sign directions `d_j` and radius `rho`, let

`c_j = c + rho * max(1, abs(c)) * d_j`,

`g_j(x) = clip((f(x;c_j) − f(x;c))/s_y, −1, 1)`.

The **finite symmetric local class** is `G_f = {0, +g_1, −g_1, ..., +g_J, −g_J}`. With fixed training-row Rademacher sign draws `sigma_k`, compute

`R_local = mean_k max(0, max_j abs(mean_i sigma_ki * g_j(x_i)))`.

Zero and both signs are represented explicitly by zero and absolute correlations. All candidates share the same Monte Carlo draws; no validation rows are involved. If there are no free constants, this local perturbation class is `{0}` and the term is zero. An invalid perturbation receives penalty 1. The estimate lies in `[0,1]`.

This differs from Raymond et al.'s midpoint-binarized output heuristic. It defines the finite function family over which a supremum is approximated, but it still is a data-dependent **local coefficient sensitivity heuristic**. It does not measure full structural capacity, provide a global supremum, or certify a generalization bound. Its value depends on numerical units and coefficient scaling; fix the unit convention across experiments. A scientifically useful novelty claim requires testing this choice against simpler penalties and against the original heuristic.

## 4. NSGA-II and final selection

Stage two starts from screened candidates and evolves against `(training_NRMSE, C)`. Since the training scale is fixed, NRMSE and RMSE have the same dominance order. Nondominated sorting and normalized objective crowding follow standard NSGA-II.

For tree size `s`, `SD(s,l) = count(|size_i−s|<=l)/(2l+1)` over the current combined population. Within a front the crowding score becomes `crowding / (1 + SD/mean(SD))`; infinite boundary scores stay infinite. Ties prefer smaller SD and then stable order. Rank always takes precedence. This is a deliberate extension of the proposal's `(rank, crowding, SD)` tie-breaking: it makes SD affect interior truncation even without exact crowding ties. It is **not** an exact reproduction of Zhang et al., whose SD replaces the second objective. `use_size_diversity=False` restores ordinary crowding ordering.

The final training Pareto front is computed over the screened seeds and evaluated stage-two offspring. Validation is evaluated only after the search ends. Select a finite-domain front member minimizing

`validation_RMSE/s_y + selection_complexity_weight * C/max_nodes`.

The fixed normalization avoids unit-dependent raw RMSE/complexity addition and shifting front-wise min–max scales. If all front members fail on validation domains, fitting fails explicitly. No coefficient refit uses validation. A held-out test set is not part of `fit`.

## Implementation boundaries

- CPU reference implementation; no vectorized GPU generation, multiprocessing or optimizer checkpoint resume.
- Finite operator list and unit catalog; no arbitrary symbolic powers, shared coefficients or learned unit conversion.
- Pareto sorting uses quadratic pairwise comparisons. Large archives require memory/time planning.
- Fixed-seed replay is tested within one environment; floating point/library changes can alter search paths.
- Selection hyperparameters must be tuned inside training/validation protocols. A validation score is not an unbiased final test score.
- Dimensional consistency is necessary for physical plausibility but does not imply causal or physical correctness.
