from dataclasses import replace
import numpy as np

from dimensr import SearchConfig, DimenSRRegressor
from dimensr.evaluation import Evaluator
from dimensr.expression import Node
from dimensr.fitting import affine_constants
from dimensr.genetics import fitted_offspring
from dimensr.grammar import Grammar
from dimensr.policy import Policy
from dimensr.pareto import fronts, first_front
from dimensr.complexity import local_rademacher
from dimensr.units import ZERO, unit


def test_feasible_sqrt_offset_is_not_lost_at_invalid_initialization():
    X = np.linspace(10, 20, 30)[:, None]
    tree = Node("sqrt", (Node("sub", (Node("c0"), Node("x0"))),))
    evaluator = Evaluator(X, np.sqrt(30 - X[:, 0]), SearchConfig())
    fit = evaluator(tree)
    assert fit is not None
    assert fit.train_rmse < 1e-7
    np.testing.assert_allclose(fit.constants, [30], atol=1e-5)
    assert evaluator.domain_repairs > 0
    # Numerical domains are still strict at prediction time.
    assert np.isnan(fit.tree.evaluate(np.array([[31.0]]), fit.constants)).all()


def test_affine_solve_handles_different_scales_and_independent_constants():
    X = np.column_stack((np.linspace(0, 1000, 40), np.linspace(0, 1, 40) ** 2))
    terms = [Node("mul", (Node("c0"), Node(f"x{i}"))) for i in range(2)]
    tree = Node("add", (Node("add", tuple(terms)), Node("c0")))
    evaluator = Evaluator(X, 2.5 * X[:, 0] - 7 * X[:, 1] + 20, SearchConfig())
    fit = evaluator(tree)
    assert fit.train_rmse < 1e-9
    assert evaluator.linear_fits == 1
    assert evaluator.constant_nfev == 0
    assert not affine_constants(Node("mul", (Node("c0"), Node("c0"))))


def test_commutative_cache_preserves_coefficient_occurrences_and_domain_holes():
    a = Node("add", (Node("x0"), Node("c0")))
    b = Node("mul", (Node("c0"), Node("x0")))
    tree = Node("add", (a, b))
    X = np.linspace(1, 3, 10)[:, None]
    canonical, constants = tree.canonical([5, 2])
    np.testing.assert_array_equal(tree.evaluate(X, [5, 2]), canonical.evaluate(X, constants))
    evaluator = Evaluator(X, 3 * X[:, 0] + 5, SearchConfig())
    first = evaluator(tree)
    second = evaluator(Node("add", (b, a)))
    assert first is second
    assert evaluator.calls == 1 and evaluator.cache_hits == 1
    hole = Node("div", (Node("x0"), Node("x0")))
    root, c = hole.canonical()
    assert np.isnan(root.evaluate(np.array([[0.0]]), c)).all()


def test_physical_bounds_reject_finite_but_negative_predictions():
    X = np.linspace(-1, 1, 15)[:, None]
    evaluator = Evaluator(X, X[:, 0], SearchConfig(output_bounds=(0, None)))
    assert evaluator(Node("x0")) is None
    assert evaluator.failures["numeric_or_physical_rejection"] == 1


def test_phase_two_has_real_offspring_when_phase_one_exhausts_its_share():
    X = np.linspace(0.2, 2, 40)[:, None]
    cfg = SearchConfig(population_size=16, epochs=20, gp_generations=2, selection_generations=6,
                       max_nodes=7, max_depth=4, max_evaluations=25, min_candidates=1,
                       use_rnn=False, constant_restarts=1, constant_max_nfev=10,
                       operators=("add", "mul", "square"), random_state=7)
    model = DimenSRRegressor(cfg).fit(X, X[:, 0] ** 2 + X[:, 0])
    assert sum(r["new_evaluations"] for r in model.history_ if r["phase"] == "nsga2") > 0
    assert model.diagnostics_["evaluations"] <= 25
    assert max(r["evaluations"] for r in model.history_ if r["phase"] == "neural_gp") <= 18


def test_multivariable_affine_seed_reaches_known_solution_with_enough_capacity():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 8))
    y = X @ np.arange(1, 9) + 3
    cfg = SearchConfig(population_size=32, epochs=1, gp_generations=0, selection_generations=0,
                       max_nodes=41, max_depth=8, use_rnn=False, operators=("add", "mul"),
                       max_evaluations=1000, min_candidates=1, polish_candidates=0)
    model = DimenSRRegressor(cfg).fit(X[:80], y[:80], X_val=X[80:], y_val=y[80:])
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)
    assert model.best_.tree.size == 33


def test_fitted_offspring_inherits_parent_and_added_term_coefficients():
    X = np.linspace(0.2, 2, 40)[:, None]
    e = Evaluator(X, 3 * X[:, 0] + 5, SearchConfig())
    parent = e(Node("mul", (Node("c0"), Node("x0"))))
    term = e(Node("c0"))
    grammar = Grammar([ZERO], ZERO, [ZERO], ("add", "mul"), 9, 5)
    child, coefficients, operator = fitted_offspring(parent, term, grammar, np.random.default_rng(8), term=term)
    np.testing.assert_allclose(child.evaluate(X, coefficients),
                               parent.tree.evaluate(X, parent.constants) + term.tree.evaluate(X, term.constants))
    assert operator == "add_term"


def test_tangent_rademacher_is_invariant_to_redundant_regular_parameterization():
    X = np.linspace(-1, 1, 60)[:, None]
    signs = np.random.default_rng(2).choice([-1, 1], (128, len(X)))
    x, c = Node("x0"), Node("c0")
    a = Node("mul", (c, x))
    b = Node("mul", (Node("mul", (c, c)), x))
    ra = local_rademacher(a, [2], X, 1, signs)
    rb = local_rademacher(b, [1, 2], X, 1, signs)
    np.testing.assert_allclose(ra, rb, rtol=1e-6)


def test_fast_front_agrees_with_full_sort_including_objective_ties():
    rng = np.random.default_rng(21)
    for _ in range(10):
        values = rng.integers(0, 12, (80, 2))
        assert set(first_front(values)) == set(fronts(values)[0])


def test_production_policy_replays_and_learns_unit_splits():
    import torch
    length = unit([1, 0, 0, 0, 0, 0, 0])
    inverse = unit([-1, 0, 0, 0, 0, 0, 0])
    grammar = Grammar([length, inverse, ZERO], ZERO, [ZERO], ("mul", "div", "add"), 11, 5)
    policy = Policy(len(grammar.tokens), seed=5)
    traces = []
    rng = np.random.default_rng(5)
    for _ in range(24):
        tree, trace = grammar.generate(rng, policy)
        replay = grammar.trace(tree)
        assert trace.production_indices == replay.production_indices
        for a, b in zip(trace.production_features, replay.production_features):
            np.testing.assert_array_equal(a, b)
        traces.append(trace)
    before = policy.production_query.weight.detach().clone()
    policy.update(traces, np.linspace(0, 1, len(traces)), traces[-4:])
    assert not torch.equal(before, policy.production_query.weight.detach())
