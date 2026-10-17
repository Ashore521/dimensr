"""Reproducible grouped pilot on three UCI experimental datasets.

Run ``python -m dimensr.real_data --help``. Downloads stay in an ignored cache.
This experiment runner does not change the estimator's search algorithm.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import urllib.request
from unittest.mock import patch
import warnings
import zipfile

import numpy as np
import scipy
import sklearn
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SearchConfig
from .evaluation import Evaluator
from .expression import Node
from .regressor import DimenSRRegressor
from .units import ZERO, sub, unit, encode


L = [1, 0, 0, 0, 0, 0, 0]
T = [0, 0, 1, 0, 0, 0, 0]
DENSITY = [-3, 1, 0, 0, 0, 0, 0]
STRESS = [-1, 1, -2, 0, 0, 0, 0]
DATASETS = {
    "airfoil": {
        "title": "Airfoil Self-Noise", "uci_id": 291,
        "page": "https://archive.ics.uci.edu/dataset/291/airfoil+self-noise",
        "url": "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip",
        "member": "airfoil_self_noise.dat", "rows": 1503,
        "citation": "Brooks, Pope & Marcolini (1989), doi:10.24432/C5VW2C",
        "names": ["frequency", "angle", "chord", "velocity", "thickness"],
        "native_units": ["Hz", "degree", "m", "m/s", "m"], "y_unit": "dB",
        "feature_units": [[0, 0, -1, 0, 0, 0, 0], list(ZERO), L, [1, 0, -1, 0, 0, 0, 0], L],
        "target_units": list(ZERO), "si_factors": [1, np.pi / 180, 1, 1, 1], "y_si_factor": 1,
        "group_columns": [1, 2, 3, 4],
        "group_meaning": "same angle, chord, velocity and displacement thickness; frequencies kept together",
        "unit_note": "Angle becomes radians. dB is a dimensionless logarithmic response, not pressure in Pa.",
    },
    "concrete": {
        "title": "Concrete Compressive Strength", "uci_id": 165,
        "page": "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
        "url": "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
        "member": "Concrete_Data.xls", "rows": 1030,
        "citation": "Yeh (1998), doi:10.24432/C5PK67; Cement and Concrete Research 28(12), 1797-1808",
        "names": ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse", "fine", "age"],
        "native_units": ["kg/m^3"] * 7 + ["day"], "y_unit": "MPa",
        "feature_units": [DENSITY] * 7 + [T], "target_units": STRESS,
        "si_factors": [1] * 7 + [86400], "y_si_factor": 1e6,
        "group_columns": list(range(7)),
        "group_meaning": "same seven ingredient quantities; all ages and repeated tests kept together",
        "unit_note": "Convert age to seconds and strength to Pa before scaling. Unknown coefficients have declared units.",
    },
    "yacht": {
        "title": "Yacht Hydrodynamics", "uci_id": 243,
        "page": "https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics",
        "url": "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip",
        "member": "yacht_hydrodynamics.data", "rows": 308,
        "citation": "Gerritsma, Onnink & Versluis (1981), doi:10.24432/C5XG7R",
        "names": ["buoyancy", "prismatic", "length_displacement", "beam_draught", "length_beam", "froude"],
        "native_units": ["dimensionless"] * 6, "y_unit": "source resistance coefficient scale",
        "feature_units": [list(ZERO)] * 6, "target_units": list(ZERO),
        "si_factors": [1] * 6, "y_si_factor": 1,
        "group_columns": list(range(5)),
        "group_meaning": "same hull geometry; all Froude numbers kept together",
        "unit_note": "All supplied variables are dimensionless; this task cannot establish a benefit from unit constraints.",
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_dataset(name, cache):
    """Read original UCI bytes without dropping, imputing or averaging rows."""
    spec = DATASETS[name]
    path = Path(cache) / f"{name}.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            data = response.read()
        # Check the required member before persisting a network response.
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            archive.getinfo(spec["member"])
        path.write_bytes(data)
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        content = archive.read(spec["member"])
    if name == "concrete":
        import xlrd  # optional experiment dependency, not required for the estimator
        sheet = xlrd.open_workbook(file_contents=content).sheet_by_index(0)
        values = np.array([sheet.row_values(i) for i in range(1, sheet.nrows)], dtype=float)
    else:
        values = np.loadtxt(io.BytesIO(content))
    expected = (spec["rows"], len(spec["names"]) + 1)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Unexpected shape or nonfinite values in {name}: {values.shape}; expected {expected}")
    X, y = values[:, :-1], values[:, -1]
    _, groups = np.unique(X[:, spec["group_columns"]], axis=0, return_inverse=True)
    info = dict(spec, license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
                archive_sha256=digest(raw), member_sha256=digest(content),
                n_groups=len(np.unique(groups)), n_features=X.shape[1], missing_cells=0,
                exact_duplicate_rows=len(values) - len(np.unique(values, axis=0)),
                duplicate_feature_rows=len(X) - len(np.unique(X, axis=0)),
                feature_min=X.min(axis=0).tolist(), feature_max=X.max(axis=0).tolist(),
                feature_zero_counts=(X == 0).sum(axis=0).tolist(), target_range=[float(y.min()), float(y.max())],
                duplicate_policy="Retain original rows; identical features cannot cross group splits.")
    # ZERO uses Fraction internally; published metadata uses integer exponents.
    info["feature_units"] = [[int(v) for v in u] for u in spec["feature_units"]]
    info["target_units"] = [int(v) for v in spec["target_units"]]
    return X, y, groups, info


def grouped_split(groups, seed):
    """Approximately 60/20/20 by group count, never by target values."""
    groups = np.asarray(groups)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    remaining, test = next(outer.split(np.zeros(len(groups)), groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1000)
    tr, va = next(inner.split(np.zeros(len(remaining)), groups=groups[remaining]))
    parts = [np.sort(remaining[tr]), np.sort(remaining[va]), np.sort(test)]
    for i in range(3):
        for j in range(i):
            if set(groups[parts[i]]) & set(groups[parts[j]]):
                raise ValueError("Group leakage")
    return dict(zip(("train", "validation", "test"), parts))


def scaling(X, y, train, spec):
    """Convert to SI then multiply by training-only, dimensionless numeric scales.

    u_j = (SI_factor_j / x_scale_j) * x_native_j and v = (y_SI_factor / y_scale) * y_native.
    x_scale and y_scale are numerical multipliers, NOT dimensioned reference quantities:
    u and v retain their physical dimension vectors. There is no mean subtraction.
    """
    Xsi = X * np.asarray(spec["si_factors"])
    ysi = y * spec["y_si_factor"]
    xs = np.sqrt(np.mean(Xsi[train] ** 2, axis=0))
    xs[xs == 0] = 1
    ys = float(np.std(ysi[train]))
    if ys == 0:
        ys = max(float(np.sqrt(np.mean(ysi[train] ** 2))), 1.0)
    return Xsi / xs, ysi / ys, {"x_scale": xs.tolist(), "y_scale": ys,
                                "x_native_to_model": (np.asarray(spec["si_factors"]) / xs).tolist(),
                                "y_model_to_native": ys / spec["y_si_factor"],
                                "centering": False, "scale_fit_rows": train.tolist()}


def constant_dimensions(spec, include_normalizers=False):
    """Declare intercept/slopes and, optionally, inverse-unit reference scales."""
    target = unit(spec["target_units"])
    dims = [ZERO, target] + [sub(target, unit(u)) for u in spec["feature_units"]]
    if include_normalizers:
        dims += [sub(ZERO, unit(u)) for u in spec["feature_units"]]
    return tuple(dict.fromkeys(dims))


def audit_tree(tree, feature_units, target_units, constant_units):
    """Reconstruct every node to check operator rules, then check leaf declarations."""
    Node.from_dict(tree.to_dict())
    if tree.units != target_units:
        raise ValueError("Root dimensions do not match target")
    for _, node in tree.walk():
        if not node.children:
            declared = feature_units if node.token.startswith("x") else constant_units
            index = int(node.token[1:])
            if index >= len(declared) or node.units != declared[index]:
                raise ValueError("Leaf dimensions do not match declared feature/constant")


def metrics(y, prediction, train_std):
    prediction = np.asarray(prediction, dtype=float)
    invalid = int((~np.isfinite(prediction)).sum())
    if invalid:
        return {"status": "invalid_predictions", "invalid_count": invalid, "n": len(y)}
    rmse = float(np.sqrt(mean_squared_error(y, prediction)))
    return {"status": "ok", "n": len(y), "invalid_count": 0,
            "rmse": rmse, "mae": float(mean_absolute_error(y, prediction)),
            "r2": float(r2_score(y, prediction)), "nrmse_train_std": rmse / train_std}


def fit_symbolic(X, y, split, spec, config, use_units, path, *, include_normalizers=False):
    cfg = replace(config, use_units=use_units)
    physical_features = tuple(unit(u) for u in spec["feature_units"])
    physical_constants = constant_dimensions(spec, include_normalizers)
    features = physical_features if use_units else (ZERO,) * len(physical_features)
    target = unit(spec["target_units"]) if use_units else ZERO
    constants = physical_constants if use_units else ((ZERO,) if physical_constants else ())
    audited = {"unique_evaluated_trees": 0, "checked_calls": 0, "violations": 0,
               "physical_units_enabled": use_units,
               "nontrivial_physical_units": any(u != ZERO for u in physical_features)}

    class AuditedEvaluator(Evaluator):
        def __call__(self, tree, initial_constants=None, *, refine=False):
            tree, warm = tree.canonical(initial_constants)
            audited["checked_calls"] += 1
            if tree not in self.cache and self.calls < self.config.max_evaluations:
                try:
                    audit_tree(tree, features, target, constants)
                except ValueError:
                    audited["violations"] += 1
                    raise
                audited["unique_evaluated_trees"] += 1
            return super().__call__(tree, warm if initial_constants is not None else None, refine=refine)

    model = DimenSRRegressor(cfg, physical_features, unit(spec["target_units"]),
                            physical_constants, spec["names"])
    progress_path = path.with_suffix(".progress.json")

    def progress(entry):
        write_json(progress_path, entry)
        print(f"PROGRESS {path.stem}: {entry['phase']} {entry['step']} "
              f"new={entry['new_evaluations']} total={entry['evaluations']} "
              f"train_rmse={entry['best_train_rmse']:.5g}", flush=True)

    with patch("dimensr.regressor.Evaluator", AuditedEvaluator), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        model.fit(X[split["train"]], y[split["train"]],
                  X_val=X[split["validation"]], y_val=y[split["validation"]], callback=progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    # Re-evaluate after portable serialization; test labels are not accessed here.
    restored = DimenSRRegressor.load(path)
    np.testing.assert_allclose(restored.predict(X[split["validation"]]), model.predict(X[split["validation"]]))
    return model, {"config": asdict(cfg), "diagnostics": model.diagnostics_, "unit_audit": audited,
                   "constant_units": [encode(u) for u in constants], "expression": str(model.expression_),
                   "nodes": model.best_.tree.size, "model_file": f"models/{path.name}",
                   "warnings": sorted({str(w.message) for w in caught})}


def fit_baseline(name, X, y, split, seed):
    if name == "mean":
        candidates = [(None, DummyRegressor(strategy="mean"))]
    elif name == "ridge":
        candidates = [(a, make_pipeline(StandardScaler(), Ridge(alpha=a))) for a in (0.01, 0.1, 1, 10, 100)]
    else:
        candidates = [(leaf, RandomForestRegressor(n_estimators=200, min_samples_leaf=leaf,
                                                   random_state=seed, n_jobs=1)) for leaf in (1, 3, 5)]
    scored = []
    for parameter, model in candidates:
        model.fit(X[split["train"]], y[split["train"]])
        error = float(mean_squared_error(y[split["validation"]], model.predict(X[split["validation"]])))
        scored.append((error, parameter, model))
    best = min(scored, key=lambda row: row[0])
    return best[2], {"selected_parameter": best[1], "validation_mse_grid": [[r[1], r[0]] for r in scored]}


def code_hash():
    source = Path(__file__).parent
    return digest(b"".join(p.name.encode() + p.read_bytes().replace(b"\r\n", b"\n") for p in sorted(source.glob("*.py"))))


def run(args):
    torch.set_num_threads(1)
    cfg = SearchConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    cfg.validate()
    output = Path(args.output)
    protocol = {"schema_version": 1, "config": asdict(cfg), "seeds": args.seeds, "datasets": args.datasets,
                "split": "GroupShuffleSplit: test .2; validation .25 of remaining; random states seed and seed+1000",
                "ridge_alphas": [0.01, 0.1, 1, 10, 100], "rf_leaves": [1, 3, 5], "rf_trees": 200,
                "source_sha256": code_hash(), "preprocessing": "SI conversion, train RMS X / std y, no centering",
                "test_policy": "Test metrics after validation selection only; no test-driven tuning or refitting",
                "scope": "Small grouped real-data pilot, not a powered superiority study or equal-compute comparison"}
    fingerprint = digest(json.dumps(protocol, sort_keys=True).encode())
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError("Protocol/source changed. Use a new output directory to preserve earlier results.")
    if not protocol_path.exists():
        write_json(protocol_path, dict(protocol, fingerprint=fingerprint, frozen_at=datetime.now(timezone.utc).isoformat()))
    info_all, rows = {}, []
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "processor": platform.processor(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "sklearn": sklearn.__version__, "torch": torch.__version__, "torch_threads": 1}
    for name in args.datasets:
        X, y, groups, info = load_dataset(name, args.cache)
        info_all[name] = info
        print(f"DATA {name}: {len(y)} rows, {info['n_groups']} groups, {info['exact_duplicate_rows']} duplicate rows", flush=True)
        if args.prepare_only:
            continue
        spec = DATASETS[name]
        for seed in args.seeds:
            split = grouped_split(groups, seed)
            Xmodel, ymodel, scales = scaling(X, y, split["train"], spec)
            split_info = {"indices": {k: v.tolist() for k, v in split.items()},
                          "group_counts": {k: len(np.unique(groups[v])) for k, v in split.items()},
                          "row_counts": {k: len(v) for k, v in split.items()}, "preprocessing": scales,
                          "data_sha256": info["member_sha256"]}
            split_hash = digest(json.dumps(split_info, sort_keys=True).encode())
            write_json(output / "splits" / f"{name}_{seed}.json", split_info)
            methods = ["full"] + ([] if name == "yacht" else ["no_units"]) + ["mean", "ridge", "random_forest"]
            for method in methods:
                path = output / "runs" / f"{name}_{seed}_{method}.json"
                if path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row["protocol_fingerprint"] != fingerprint or row["split_sha256"] != split_hash:
                        raise ValueError("Existing result has different protocol or data")
                    rows.append(row)
                    continue
                print(f"START {name} seed={seed} method={method}", flush=True)
                start = time.monotonic()
                row = {"dataset": name, "seed": seed, "method": method, "protocol_fingerprint": fingerprint,
                       "split_sha256": split_hash, "environment": environment, "target_metric_unit": spec["y_unit"]}
                try:
                    if method in ("full", "no_units"):
                        model, detail = fit_symbolic(Xmodel, ymodel, split, spec, replace(cfg, random_state=seed),
                                                    method == "full", output / "models" / f"{name}_{seed}_{method}.json")
                    else:
                        model, detail = fit_baseline(method, Xmodel, ymodel, split, seed)
                    row.update(detail)
                    row["fit_seconds"] = time.monotonic() - start
                    scores = {}
                    for part, indices in split.items():
                        if method in ("full", "no_units"):
                            pred = model.best_.tree.evaluate(Xmodel[indices], model.best_.constants)
                        else:
                            pred = model.predict(Xmodel[indices])
                        scores[part] = metrics(y[indices], pred * scales["y_model_to_native"], float(np.std(y[split["train"]])))
                    row.update(status="ok", scores=scores)
                    # Predictions retain original row indices and native target scale; no raw features are redistributed.
                    row["test_predictions"] = [float(v) if np.isfinite(v) else None for v in pred * scales["y_model_to_native"]]
                except (ValueError, RuntimeError, AssertionError, FloatingPointError) as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                row["wall_seconds"] = time.monotonic() - start
                write_json(path, row)
                rows.append(row)
                print(f"DONE {name} seed={seed} method={method}: {row.get('scores', {}).get('test', row.get('error'))}; {row['wall_seconds']:.1f}s", flush=True)
    write_json(output / "datasets.json", info_all)
    if not args.prepare_only:
        summaries = []
        for name in args.datasets:
            for method in dict.fromkeys(r["method"] for r in rows if r["dataset"] == name):
                subset = [r for r in rows if r["dataset"] == name and r["method"] == method]
                valid = [r for r in subset if r["status"] == "ok" and r["scores"]["test"]["status"] == "ok"]
                entry = {"dataset": name, "method": method, "runs": len(subset), "valid_test_runs": len(valid)}
                if valid:
                    for metric in ("r2", "rmse", "mae", "nrmse_train_std"):
                        vals = [r["scores"]["test"][metric] for r in valid]
                        entry[metric] = {"mean": float(np.mean(vals)), "sample_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                                         "min": min(vals), "max": max(vals)}
                    entry["mean_fit_seconds"] = float(np.mean([r["fit_seconds"] for r in valid]))
                summaries.append(entry)
        write_json(output / "summary.json", {"protocol_fingerprint": fingerprint, "summary": summaries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--config", default="examples/real_data_config.json")
    parser.add_argument("--cache", default="outputs/real_data/cache")
    parser.add_argument("--output", default="results/real_data")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
