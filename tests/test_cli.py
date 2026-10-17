import csv
import json
import numpy as np
from dimensr.cli import main


def test_csv_training_and_prediction(tmp_path):
    data = tmp_path / "samples.csv"
    with data.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mass", "acceleration", "force"])
        for i in range(1, 30):
            writer.writerow([i / 10, (i % 7 + 1) / 5, i / 10 * (i % 7 + 1) / 5])
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"feature_units": [[0, 1, 0, 0, 0, 0, 0], [1, 0, -2, 0, 0, 0, 0]],
                                "target_units": [1, 1, -2, 0, 0, 0, 0]}))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"population_size": 4, "epochs": 1, "gp_generations": 0,
                               "selection_generations": 0, "operators": ["mul"],
                               "max_nodes": 3, "max_depth": 2, "min_candidates": 1}))
    model = tmp_path / "model.json"
    assert main(["fit", "--data", str(data), "--target", "force", "--metadata", str(meta),
                 "--config", str(cfg), "--output", str(model)]) == 0
    output = tmp_path / "pred.csv"
    assert main(["predict", "--model", str(model), "--data", str(data), "--output", str(output)]) == 0
    rows = np.loadtxt(output, skiprows=1)
    assert rows.shape == (29,)
    assert (tmp_path / "model.json.progress.json").exists()


def test_cli_rejects_missing_target(tmp_path):
    data = tmp_path / "samples.csv"
    data.write_text("x,y\n1,2\n")
    assert main(["fit", "--data", str(data), "--target", "missing"]) == 2
