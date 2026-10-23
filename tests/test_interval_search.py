import numpy as np
import pytest
from dimensr import DimenSRRegressor, SearchConfig
from dimensr.evaluation import Evaluator
from dimensr.expression import Node
from dimensr.grammar import Grammar
from dimensr.intervals import audit_box
from dimensr.structure import operator, structured_seeds
from dimensr.units import ZERO


def test_off_grid_pole_is_not_accepted_as_enclosed():
    x, c = Node("x0"), Node("c0")
    tree = operator("div", c, operator("sub", x, c))
    constants = [1., .37123456789]
    assert audit_box(tree, constants, [[0, 1]])["status"] != "enclosed"
    assert audit_box(tree, constants, [[.5, 1]])["status"] == "enclosed"
    X = np.array([.5, .7, .9, 1.1, 1.5, 2.])[:, None]
    fixed = operator("div", x, operator("log", x))
    y = fixed.evaluate(X, [])
    assert Evaluator(X, y, SearchConfig(interval_checks=False))(fixed) is not None
    assert Evaluator(X, y, SearchConfig())(fixed) is None


def test_subdivision_resolves_dependency_without_rewriting_tree():
    x, c = Node("x0"), Node("c0")
    denominator = operator("add", operator("sub", operator("square", x), x), c)
    tree = operator("div", c, denominator)
    assert audit_box(tree, [1., 1.], [[0, 1]], max_boxes=1)["status"] == "unresolved"
    assert audit_box(tree, [1., 1.], [[0, 1]], max_boxes=32)["status"] == "enclosed"


def test_narrow_invalid_root_and_true_zero_root_are_distinguished():
    x, c = Node("x0"), Node("c0")
    tree = operator("sqrt", operator("mul", operator("sub", x, c), operator("sub", x, c)))
    assert audit_box(tree, [.21, .29], [[0, 1]])["status"] != "enclosed"
    legal = operator("sqrt", operator("square", x))
    assert audit_box(legal, [], [[-1, 1]])["status"] == "enclosed"
    assert audit_box(operator("log", x), [], [[0, 1]])["status"] == "invalid"


def test_box_budget_and_bounds_are_validated():
    with pytest.raises(ValueError, match="max_boxes"):
        audit_box(Node("x0"), [], [[0, 1]], max_boxes=0)
    with pytest.raises(ValueError, match="bounds"):
        audit_box(Node("x0"), [], [[1, 0]])


def test_explicit_only_domain_does_not_impose_an_interval_box():
    X = np.array([[0., 1.], [1., 0.], [.2, .8], [.8, .2], [.5, .5]])
    cfg = SearchConfig(boundary_probes=False, domain_probe_count=0)
    evaluator = Evaluator(X, np.ones(5), cfg, X_domain=X)
    assert evaluator.interval_bounds is None


def test_enclosed_generated_trees_are_finite_on_independent_dense_points():
    rng = np.random.default_rng(910)
    ops = ("add", "sub", "mul", "div", "square", "sqrt", "sin", "cos", "exp", "log", "neg")
    grammar = Grammar([ZERO], ZERO, [ZERO], ops, 13, 6)
    X = np.linspace(-1, 1, 1001)[:, None]
    enclosed = 0
    for _ in range(120):
        tree, _ = grammar.generate(rng)
        constants = rng.uniform(-2, 2, tree.n_constants)
        if audit_box(tree, constants, [[-1, 1]], max_boxes=16)["status"] == "enclosed":
            enclosed += 1
            assert np.isfinite(tree.evaluate(X, constants)).all()
    assert enclosed > 10


def test_output_bounds_and_intermediate_overflow_are_checked():
    x = Node("x0")
    assert audit_box(x, [], [[-1, 1]], output_bounds=(0, None))["status"] == "invalid"
    tree = operator("div", x, operator("exp", x))
    assert audit_box(tree, [], [[800, 900]])["status"] == "invalid"


@pytest.mark.parametrize("case", ["log_sum", "exp_product"])
def test_complementary_compositions_survive_library_and_sparse_screening(case):
    rng = np.random.default_rng(810)
    X = rng.uniform(.05, 1.7, (100, 2 if case == "exp_product" else 1))
    y = (2*np.log1p(X[:, 0]) + .7*np.log1p(X[:, 0]**2) if case == "log_sum" else
         1.4*np.exp(X[:, 0])*np.cos(X[:, 1]) + .3*X[:, 0]*X[:, 1])
    ops = ("add", "sub", "mul", "div", "square", "sqrt", "sin", "cos", "exp", "log")
    grammar = Grammar([ZERO]*X.shape[1], ZERO, [ZERO], ops, 21, 7)
    proposals, work = structured_seeds(grammar, X, y, 96)
    error = min(np.max(np.abs(t.evaluate(X, c)-y)) for t, c in proposals)
    assert error < 1e-8
    assert work["retained_features"] <= 512 and work["numeric_duplicates"] > 0


def test_saved_model_retains_interval_audit(tmp_path):
    X = np.linspace(.1, 2, 25)[:, None]
    cfg = SearchConfig(population_size=8, epochs=1, gp_generations=0, selection_generations=0,
                       min_candidates=1, max_evaluations=40, operators=("add", "mul"),
                       use_rnn=False, polish_candidates=0)
    model = DimenSRRegressor(cfg).fit(X, 2*X[:, 0], input_bounds=[[0, 2]])
    model.save(tmp_path/'model.json')
    restored = DimenSRRegressor.load(tmp_path/'model.json')
    assert restored.audit_domain()["status"] == "enclosed"
    assert restored.audit_domain([[0, 10]])["status"] == "enclosed"
