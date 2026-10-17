from dataclasses import replace
import json
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from dimensr import SearchConfig, DimenSRRegressor
from dimensr.expression import Node
from dimensr.evaluation import Evaluator
from dimensr.benchmarks import dataset


def config(**kwargs):
    return replace(SearchConfig(population_size=8, epochs=2, gp_generations=1,
                                selection_generations=1, max_nodes=5, max_depth=3,
                                operators=("mul",), min_candidates=1,
                                constant_restarts=1, constant_max_nfev=20), **kwargs)


def model_for_force(cfg=None):
    d = dataset("force", n=40)
    m = DimenSRRegressor(cfg or config(), feature_units=d["feature_units"],
                         target_units=d["target_units"], feature_names=d["feature_names"])
    return m, d


def test_physical_discovery_serialization_and_sklearn(tmp_path):
    model, d = model_for_force()
    clone(model)
    with pytest.raises(NotFittedError):
        model.predict(d["X"])
    model.fit(d["X"], d["y"])
    assert model.score(d["X_test"], d["y_test"]) > 1 - 1e-10
    assert model.best_.tree.units == tuple(d["target_units"])
    assert any(row["phase"] == "nsga2" for row in model.history_)
    model.save(tmp_path / "model.json")
    loaded = DimenSRRegressor.load(tmp_path / "model.json")
    np.testing.assert_array_equal(model.predict(d["X_test"]), loaded.predict(d["X_test"]))
    with pytest.raises(ValueError, match="Feature count"):
        loaded.predict(np.ones((5, 3)))


def test_constant_optimization():
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    tree = Node("add", (Node("mul", (Node("c0"), Node("x0"))), Node("c0")))
    fit = Evaluator(X, 2.5 * X[:, 0] - 0.7, config())(tree)
    assert fit.train_rmse < 1e-8
    _, expected = tree.canonical([2.5, -0.7])
    np.testing.assert_allclose(fit.constants, expected, atol=1e-8)


def test_validation_does_not_change_search_or_fitted_constants():
    a, d = model_for_force()
    b, _ = model_for_force()
    a.fit(d["X"], d["y"], X_val=d["X_test"], y_val=d["y_test"])
    b.fit(d["X"], d["y"], X_val=d["X_test"], y_val=d["y_test"] + 100)
    for left, right in zip(a.history_, b.history_):
        assert {k: v for k, v in left.items() if k != "elapsed_seconds"} == {
            k: v for k, v in right.items() if k != "elapsed_seconds"}
    assert [{k: v for k, v in r.items() if k not in ("validation_rmse", "selection_score")}
            for r in a.pareto_front_] == [
                {k: v for k, v in r.items() if k not in ("validation_rmse", "selection_score")}
                for r in b.pareto_front_]


def test_seed_reproducibility():
    a, d = model_for_force()
    b, _ = model_for_force()
    a.fit(d["X"], d["y"])
    b.fit(d["X"], d["y"])
    assert a.pareto_front_ == b.pareto_front_
    np.testing.assert_array_equal(a.training_indices_, b.training_indices_)
    assert set(a.training_indices_).isdisjoint(a.validation_indices_)


def test_ablations_and_evaluation_budget():
    model, d = model_for_force(config(use_rnn=False, use_gp=False, use_rademacher=False,
                                      use_size_diversity=False, max_evaluations=4))
    model.fit(d["X"], d["y"])
    assert model.policy_ is None
    assert all(r["phase"] != "nsga2" for r in model.history_)
    assert model.best_.rademacher == 0
    assert model.diagnostics_["evaluations"] <= 4


@pytest.mark.parametrize("patch", [{"epochs": 0}, {"risk_fraction": 1}, {"max_nodes": 0},
                                    {"complexity_weights": [1, -1, 0]}, {"operators": ["unknown"]}])
def test_invalid_config(patch):
    with pytest.raises(ValueError):
        replace(SearchConfig(), **patch).validate()


def test_invalid_inputs_and_failed_refit():
    model, d = model_for_force()
    with pytest.raises(ValueError, match="both"):
        model.fit(d["X"], d["y"], X_val=d["X_test"])
    model.fit(d["X"], d["y"])
    with pytest.raises(ValueError):
        model.fit(np.array([[np.nan, 1]] * 10), np.ones(10))
    with pytest.raises(NotFittedError):
        model.predict(d["X"])


def test_tampered_model_rejected(tmp_path):
    model, d = model_for_force()
    model.fit(d["X"], d["y"])
    path = tmp_path / "model.json"
    model.save(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["target_units"] = [0] * 7
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        DimenSRRegressor.load(path)
