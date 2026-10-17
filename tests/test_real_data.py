import io
import zipfile
import numpy as np
import pytest

from dimensr.expression import Node
from dimensr.grammar import Grammar
from dimensr.genetics import crossover, mutate
from dimensr.real_data import (DATASETS, audit_tree, constant_dimensions, grouped_split,
                               load_dataset, metrics, scaling)
from dimensr.units import ZERO, unit


def test_group_split_keeps_repeated_conditions_together():
    groups = np.repeat(np.arange(25), 4)
    split = grouped_split(groups, 2)
    assert sorted(np.concatenate(list(split.values())).tolist()) == list(range(100))
    group_sets = [set(groups[v]) for v in split.values()]
    assert [len(g) for g in group_sets] == [15, 5, 5]
    assert all(not a & b for i, a in enumerate(group_sets) for b in group_sets[:i])
    for key, value in grouped_split(groups, 2).items():
        np.testing.assert_array_equal(value, split[key])


def test_scaling_uses_only_training_rows_and_round_trips_native_units():
    spec = DATASETS["concrete"]
    X = np.arange(1, 49, dtype=float).reshape(6, 8)
    y = np.arange(1, 7, dtype=float)
    train = np.array([0, 1, 2])
    Xm, ym, scales = scaling(X, y, train, spec)
    changed = X.copy()
    changed[3:] *= 10000
    changed_y = y.copy()
    changed_y[3:] += 100000
    _, _, other = scaling(changed, changed_y, train, spec)
    assert other == scales
    np.testing.assert_allclose(Xm / scales["x_native_to_model"], X)
    np.testing.assert_allclose(ym * scales["y_model_to_native"], y)
    assert not scales["centering"]


@pytest.mark.parametrize("name", ["airfoil", "concrete", "yacht"])
def test_real_metadata_generation_and_genetic_variation_preserve_dimensions(name):
    spec = DATASETS[name]
    features = tuple(unit(u) for u in spec["feature_units"])
    target = unit(spec["target_units"])
    constants = constant_dimensions(spec)
    grammar = Grammar(features, target, constants, ("add", "sub", "mul", "div", "square", "log"), 15, 6)
    rng = np.random.default_rng(23)
    trees = [grammar.generate(rng)[0] for _ in range(15)]
    for tree in trees:
        for candidate in (tree, mutate(tree, grammar, rng), crossover(tree, trees[0], grammar, rng)):
            audit_tree(candidate, features, target, constants)
            assert candidate.size <= 15


def test_unit_audit_rejects_inconsistent_leaf_metadata():
    with pytest.raises(ValueError, match="Leaf dimensions"):
        audit_tree(Node("x0"), [unit([1, 0, 0, 0, 0, 0, 0])], ZERO, [ZERO])


def test_nonfinite_predictions_are_not_scored_on_a_successful_subset():
    result = metrics(np.array([1, 2, 3]), np.array([1, np.inf, 3]), 1.0)
    assert result == {"status": "invalid_predictions", "invalid_count": 1, "n": 3}
    assert "r2" not in result


def test_loader_checks_official_shape_without_network(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("yacht_hydrodynamics.data", "1 2 3 4 5 6 7\n")
    (tmp_path / "yacht.zip").write_bytes(stream.getvalue())
    with pytest.raises(ValueError, match="Unexpected shape"):
        load_dataset("yacht", tmp_path)
