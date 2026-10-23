import numpy as np
import pytest

from dimensr import SearchConfig, DimenSRRegressor
from dimensr.domains import domain_probes
from dimensr.evaluation import Evaluator
from dimensr.expression import Node
from dimensr.grammar import Grammar
from dimensr.structure import operator, structured_seeds
from dimensr.units import ZERO, unit


def test_zero_hole_rejected_even_when_training_does_not_contain_zero():
    X = np.array([[-2.], [-1.], [.5], [1.], [2.]])
    tree = operator("div", operator("square", Node("x0")), Node("x0"))
    assert Evaluator(X, X[:, 0], SearchConfig())(tree) is None
    assert Evaluator(X, X[:, 0], SearchConfig(boundary_probes=False))(tree) is not None


def test_explicit_unlabelled_zero_probe_rejects_hidden_hole():
    X = np.linspace(1, 2, 20)[:, None]
    tree = operator("div", operator("square", Node("x0")), Node("x0"))
    evaluator = Evaluator(X, X[:, 0], SearchConfig(), X_domain=[[0.]])
    assert evaluator(tree) is None and evaluator.domain_rejections > 0
    valid = evaluator(Node("x0"))
    assert valid is not None


def test_intermediate_division_by_zero_cannot_be_hidden_by_finite_outer_result():
    tree = operator("div", Node("c0"), operator("div", Node("c0"), Node("x0")))
    prediction = tree.evaluate(np.array([[0.], [2.]]), [1., 1.])
    assert np.isnan(prediction[0]) and prediction[1] == 2.
    X = np.linspace(1, 2, 12)[:, None]
    assert Evaluator(X, X[:, 0], SearchConfig(), X_domain=[[0.]])(tree) is None


def test_declared_box_corners_and_bounds_validation():
    X = np.array([[1., 2.], [2., 3.]])
    probes, audit = domain_probes(X, SearchConfig(), input_bounds=[[0, 4], [0, 5]])
    assert [0., 0.] in probes.tolist() and [4., 5.] in probes.tolist()
    assert audit["source"] == "declared_box"
    with pytest.raises(ValueError, match="contain all training"):
        domain_probes(X, SearchConfig(), input_bounds=[[0, 1], [0, 2]])
    with pytest.raises(ValueError, match="X_domain"):
        domain_probes(X, SearchConfig(), X_domain=[[np.nan, 0]])


def test_log_and_sqrt_use_their_actual_boundary_domains():
    X = np.linspace(.2, 2, 20)[:, None]
    x = Node("x0")
    assert Evaluator(X, np.log(X[:, 0]), SearchConfig(), X_domain=[[0.]])(operator("log", x)) is None
    assert Evaluator(X, np.sqrt(X[:, 0]), SearchConfig(), X_domain=[[0.]])(operator("sqrt", x)) is not None
    assert Evaluator(X, np.sqrt(X[:, 0]), SearchConfig(), X_domain=[[-.1]])(operator("sqrt", x)) is None


def test_coupled_domain_can_use_only_explicit_permitted_inputs():
    X = np.array([[0., 1.], [1., 0.]])
    cfg = SearchConfig(boundary_probes=False, domain_probe_count=0)
    probes, audit = domain_probes(X, cfg, X_domain=[[0., 2.], [2., 0.]])
    np.testing.assert_array_equal(probes, [[0., 2.], [2., 0.]])
    assert audit["source"] == "explicit_and_interpolation"


def test_composed_trigonometric_seed_uses_only_training_targets():
    X = np.linspace(-.9, .9, 70)[:, None]
    y = 2 * np.sin(X[:, 0] ** 2) * np.cos(X[:, 0]) + .4
    grammar = Grammar([ZERO], ZERO, [ZERO], ("add", "mul", "square", "sin", "cos"), 17, 7)
    proposals, work = structured_seeds(grammar, X, y, 12)
    assert min(np.max(np.abs(t.evaluate(X, c)-y)) for t, c in proposals) < 1e-9
    assert work["feature_evaluations"] <= 512 and work["linear_subproblems"] > 0


def test_common_denominator_seed_preserves_units_and_recovers_weighted_average():
    rng = np.random.default_rng(78)
    X = rng.uniform(.4, 3, (90, 4))
    mass, length = unit([0, 1, 0, 0, 0, 0, 0]), unit([1, 0, 0, 0, 0, 0, 0])
    y = (2*X[:, 0]*X[:, 2] + .7*X[:, 1]*X[:, 3])/(X[:, 0]+X[:, 1])
    grammar = Grammar([mass, mass, length, length], length, [ZERO], ("add", "mul", "div"), 21, 7)
    proposals, _ = structured_seeds(grammar, X, y, 12)
    assert min(np.max(np.abs(t.evaluate(X, c)-y)) for t, c in proposals) < 1e-9
    assert all(t.units == length and t.size <= 21 for t, _ in proposals)
    for tree, _ in proposals:
        grammar.trace(tree)


def test_fitted_model_retains_domain_audit_after_save(tmp_path):
    X = np.linspace(.2, 2, 24)[:, None]
    cfg = SearchConfig(population_size=8, epochs=1, gp_generations=0, selection_generations=0,
                       max_evaluations=40, min_candidates=1, operators=("add", "mul"),
                       use_rnn=False, polish_candidates=0)
    model = DimenSRRegressor(cfg).fit(X, 3*X[:, 0], X_domain=[[0.]])
    path = tmp_path / "model.json"
    model.save(path)
    restored = DimenSRRegressor.load(path)
    np.testing.assert_array_equal(model.predict([[0.], [2.]]), restored.predict([[0.], [2.]]))
    assert restored.diagnostics_["domain_audit"]["explicit_rows"] == 1
