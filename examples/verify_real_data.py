"""Recompute published pilot metrics from cached original data and saved models.

Run from the repository root after `python -m dimensr.real_data`.
No symbolic model is retrained. Baseline test metrics use saved predictions.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from dimensr import DimenSRRegressor
from dimensr.real_data import DATASETS, audit_tree, grouped_split, load_dataset, metrics, scaling


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(output, cache):
    protocol = read(output / "protocol.json")
    saved_data = read(output / "datasets.json")
    tested, symbolic, failed = 0, 0, 0
    for name in protocol["datasets"]:
        X, y, groups, info = load_dataset(name, cache)
        assert info["member_sha256"] == saved_data[name]["member_sha256"]
        for seed in protocol["seeds"]:
            saved = read(output / "splits" / f"{name}_{seed}.json")
            split = grouped_split(groups, seed)
            assert {k: v.tolist() for k, v in split.items()} == saved["indices"]
            Xm, _, scales = scaling(X, y, split["train"], DATASETS[name])
            assert scales == saved["preprocessing"]
            methods = ["full"] + ([] if name == "yacht" else ["no_units"]) + ["mean", "ridge", "random_forest"]
            for method in methods:
                row = read(output / "runs" / f"{name}_{seed}_{method}.json")
                assert row["protocol_fingerprint"] == protocol["fingerprint"]
                if row["status"] != "ok":
                    failed += 1
                    continue
                prediction = np.array([np.nan if v is None else v for v in row["test_predictions"]])
                recomputed = metrics(y[split["test"]], prediction, float(np.std(y[split["train"]])))
                compare_metrics(recomputed, row["scores"]["test"])
                if method in ("full", "no_units"):
                    model = DimenSRRegressor.load(output / row["model_file"])
                    audit_tree(model.best_.tree, model.feature_units_, model.target_units_, model.constant_units_)
                    assert row["unit_audit"]["violations"] == 0
                    assert row["unit_audit"]["unique_evaluated_trees"] == row["diagnostics"]["evaluations"]
                    for part, indices in split.items():
                        pred = model.best_.tree.evaluate(Xm[indices], model.best_.constants) * scales["y_model_to_native"]
                        recomputed = metrics(y[indices], pred, float(np.std(y[split["train"]])))
                        compare_metrics(recomputed, row["scores"][part])
                        if part == "test":
                            np.testing.assert_allclose(pred, prediction, rtol=1e-10, atol=1e-10, equal_nan=True)
                    symbolic += 1
                tested += 1
    print(f"Verified {tested} successful runs, including {symbolic} reloaded symbolic models; {failed} recorded fit failures.")


def compare_metrics(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if isinstance(value, str):
            assert value == expected[key]
        else:
            np.testing.assert_allclose(value, expected[key], rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/real_data"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/real_data/cache"))
    args = parser.parse_args()
    verify(args.output, args.cache)
