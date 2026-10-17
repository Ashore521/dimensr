"""Guard the acceptance harness against leakage and false domain recovery."""
import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1] / "examples/acceptance_benchmark.py"
spec = importlib.util.spec_from_file_location("acceptance", path)
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def test_paired_data_and_holdout_noise_isolation():
    case = acceptance.read(acceptance.CASES_FILE)["Nguyen-1"]
    clean = acceptance.generate(case, 0, 11, 0.)
    noisy = acceptance.generate(case, 0, 11, .01)
    for split in ("train", "val", "test", "ood", "probe"):
        np.testing.assert_array_equal(clean[f"X_{split}"], noisy[f"X_{split}"])
    for split in ("test", "ood", "probe"):
        np.testing.assert_array_equal(clean[f"y_{split}"], noisy[f"y_{split}"])
    assert not np.array_equal(clean["y_train"], noisy["y_train"])
    assert not np.array_equal(clean["X_train"][:128], clean["X_val"])


def test_raw_domain_hole_survives_symbolic_simplification():
    from dimensr.expression import Node
    x = Node("x0")
    tree = acceptance.attach_node("div", [acceptance.attach_node("square", [x]), x])
    assert str(tree.sympy([], ["x"])) == "x"
    data = {"X_probe": np.array([[0.], [1.]]), "y_probe": np.array([0., 1.])}
    result = acceptance.metric(tree, [], data, "probe")
    assert result == {"invalid": 1, "nrmse": None, "r2": None}


def test_all_ground_truth_probes_finite_and_ood_disjoint():
    for i, case in enumerate(acceptance.read(acceptance.CASES_FILE).values()):
        data = acceptance.generate(case, i, 11, 0.)
        assert all(np.isfinite(v).all() for v in data.values())
        assert len(data["X_probe"]) > 0
        assert (data["X_ood"] >= np.array(case["bounds"])[:, 1]).all()


def test_symbolic_constant_offset_is_not_exact():
    from dimensr.expression import Node
    x = Node("x0")
    tree = acceptance.attach_node("add", [x, Node("c0")])
    c = {"tree": tree.to_dict(), "constants": [3.]}
    verdict = acceptance.symbolic(c, {"names": ["x"], "truth": "x"}, 8)
    assert not verdict["exact"]
    assert verdict["constant_difference_or_ratio"]
