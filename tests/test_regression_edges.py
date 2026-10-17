from dataclasses import replace
from fractions import Fraction
import numpy as np
from dimensr import DimenSRRegressor, SearchConfig
from dimensr.evaluation import Evaluator
from dimensr.expression import Node
from dimensr.benchmarks import dataset


def test_constant_target_and_explicit_constant_units():
    cfg = SearchConfig(population_size=8, epochs=1, gp_generations=0, selection_generations=0,
                       operators=("add",), max_nodes=3, max_depth=2, min_candidates=1,
                       constant_restarts=1, use_rnn=False)
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    model = DimenSRRegressor(cfg, feature_units=[[1, 0, 0, 0, 0, 0, 0]],
                             target_units=[0, 0, 1, 0, 0, 0, 0],
                             constant_units=[[0, 0, 1, 0, 0, 0, 0]]).fit(X, np.full(30, 4.2))
    np.testing.assert_allclose(model.predict(X), 4.2, atol=1e-8)


def test_fractional_unit_model_roundtrip(tmp_path):
    cfg = SearchConfig(population_size=4, epochs=1, gp_generations=0, selection_generations=0,
                       operators=("sqrt",), max_nodes=2, max_depth=2, min_candidates=1)
    X = np.linspace(0.1, 2, 20).reshape(-1, 1)
    model = DimenSRRegressor(cfg, feature_units=[[1, 0, 0, 0, 0, 0, 0]],
                             target_units=[Fraction(1, 2), 0, 0, 0, 0, 0, 0],
                             constant_units=[]).fit(X, np.sqrt(X[:, 0]))
    model.save(tmp_path / "sqrt.json")
    restored = DimenSRRegressor.load(tmp_path / "sqrt.json")
    np.testing.assert_array_equal(model.predict(X), restored.predict(X))


def test_domain_invalid_tree_and_fitting_cost_are_recorded():
    cfg = replace(SearchConfig(), constant_restarts=1)
    X = np.array([[-2.0], [-1.0]])
    evaluator = Evaluator(X, np.ones(2), cfg)
    assert evaluator(Node("log", (Node("x0"),))) is None
    fitted = evaluator(Node("mul", (Node("c0"), Node("x0"))))
    assert fitted is not None
    assert evaluator.constant_residual_calls > 0
    assert evaluator.linear_fits == 1


def test_benchmark_probes_expose_removable_singularity():
    data = dataset("nguyen1", n=20)
    x = Node("x0")
    # This equals x on nonzero samples, but its actual program is undefined at zero.
    tree = Node("div", (Node("square", (x,)), x))
    assert np.isfinite(tree.evaluate(data["X_test"], [])).all()
    assert not np.isfinite(tree.evaluate(data["X_probe"], [])).all()
