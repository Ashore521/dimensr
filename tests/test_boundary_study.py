"""Study scoring must keep raw-tree holes and invalid physical inputs visible."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import boundary_search_study as study
from dimensr.expression import Node
from dimensr.structure import operator


def test_scoring_does_not_cancel_raw_tree_holes():
    x = Node("x0")
    tree = operator("div", operator("square", x), x)
    score = study.metrics(tree, [], np.array([[0.], [1.]]), np.array([0., 1.]))
    assert score == dict(invalid=1, nrmse=None, r2=None)


def test_declared_mass_contract_excludes_only_forbidden_zero_total():
    cases = study.read(study.CASES)
    points = study.domain_points(cases["Feynman-I.18.4"])
    assert np.all(points[:, 0] + points[:, 1] > 0)
    assert np.any(points[:, 0] == 0) and np.any(points[:, 1] == 0)


def test_fitting_data_has_no_holdout_targets_or_boundary_labels():
    cases = study.read(study.CASES)
    fit, holdout = study.generate(cases["Feynman-I.10.7"], 3, 41)
    assert set(fit) == {"X_train", "y_train", "X_val", "y_val", "X_domain"}
    assert np.any(fit["X_domain"][:, 1] == 0)
    assert all(np.isfinite(v).all() for v in holdout.values())
    assert not np.array_equal(fit["X_train"], holdout["X_test"][:len(fit["X_train"])])
