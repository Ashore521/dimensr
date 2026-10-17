import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
from . import DimenSRRegressor, SearchConfig
from .benchmarks import CASES, VARIANTS, dataset, run_benchmarks


def read_csv(path, target=None, features=None):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        if not names or len(set(names)) != len(names):
            raise ValueError("CSV needs a header with unique column names.")
        if target is not None and target not in names:
            raise ValueError(f"Target column {target!r} is missing.")
        selected = features if features is not None else [n for n in names if n != target]
        if not selected or any(f not in names for f in selected) or target in selected:
            raise ValueError("Feature columns are missing, empty, or include the target.")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows.")
    X = np.asarray([[float(row[f]) for f in selected] for row in rows], dtype=float)
    y = np.asarray([float(row[target]) for row in rows]) if target is not None else None
    return X, y, selected


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a complete prior checkpoint until the replacement is ready.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dimension-constrained neural-guided symbolic regression")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads for this CLI process")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Discover a synthetic physical relationship")
    demo.add_argument("--case", choices=CASES, default="force")
    demo.add_argument("--output", default="outputs/demo.json")
    fit = sub.add_parser("fit", help="Fit a CSV with a named target column")
    fit.add_argument("--data", required=True)
    fit.add_argument("--target", required=True)
    fit.add_argument("--metadata", help="JSON: feature_names, feature_units, target_units, constant_units, extra_units")
    fit.add_argument("--validation", help="Optional independent validation CSV with the same columns")
    fit.add_argument("--output", default="outputs/model.json")
    pred = sub.add_parser("predict", help="Predict using a saved JSON model")
    pred.add_argument("--model", required=True)
    pred.add_argument("--data", required=True)
    pred.add_argument("--output", default="outputs/predictions.csv")
    bench = sub.add_parser("benchmark", help="Run synthetic benchmarks and module ablations")
    bench.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    bench.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    bench.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["full"])
    bench.add_argument("--noise", type=float, default=0.0)
    bench.add_argument("--output", default="outputs/benchmark.json")
    for p in (demo, fit, bench):
        p.add_argument("--config", help="SearchConfig fields as JSON")
        p.add_argument("--seed", type=int, help="Override random_state")
    args = parser.parse_args(argv)
    try:
        if args.threads < 1:
            raise ValueError("threads must be positive.")
        import torch
        torch.set_num_threads(args.threads)
        cfg = SearchConfig()
        if getattr(args, "config", None):
            cfg = SearchConfig(**json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
        if getattr(args, "seed", None) is not None:
            cfg = replace(cfg, random_state=args.seed)
        cfg.validate()
        if args.command == "predict":
            model = DimenSRRegressor.load(args.model)
            X, _, _ = read_csv(args.data, features=model.feature_names_)
            values = model.predict(X)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(args.output, values, delimiter=",", header="prediction", comments="")
            print(f"Saved {len(values)} predictions to {args.output}")
            return 0
        if args.command == "benchmark":
            if args.noise < 0 or not np.isfinite(args.noise):
                raise ValueError("noise must be finite and nonnegative.")

            def checkpoint(rows):
                _write(str(args.output) + ".checkpoint.json", {"results": rows})
                row = rows[-1]
                print(f"{row['case']} seed={row['seed']} {row['variant']}: "
                      f"{row['status']}, test NRMSE={row.get('test_nrmse', 'unavailable')}", flush=True)
            result = run_benchmarks(cfg, args.cases, args.seeds, args.variants, args.noise, checkpoint)
            _write(args.output, result)
            return 0 if all(r["status"] == "ok" for r in result["results"]) else 1
        if args.command == "demo":
            data = dataset(args.case, cfg.random_state)
            X, y = data["X"], data["y"]
            metadata = {k: data[k] for k in ("feature_names", "feature_units", "target_units")}
            validation = {}
        else:
            metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8-sig")) if args.metadata else {}
            X, y, names = read_csv(args.data, args.target, metadata.get("feature_names"))
            metadata["feature_names"] = names
            validation = {}
            if args.validation:
                xv, yv, _ = read_csv(args.validation, args.target, names)
                validation = {"X_val": xv, "y_val": yv}
        model = DimenSRRegressor(cfg, **metadata)
        history = []

        def progress(row):
            history.append(row)
            _write(str(args.output) + ".progress.json", history)
            print(f"{row['phase']} {row['step']}: evaluations={row['evaluations']} "
                  f"train RMSE={row['best_train_rmse']:.6g}", flush=True)
        model.fit(X, y, callback=progress, **validation)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output)
        print(f"Expression: {model.expression_}\nValidation RMSE: {model.validation_rmse_:.6g}\nSaved: {args.output}")
        return 0
    except (ValueError, RuntimeError, OSError, TypeError) as exc:
        print(f"dimensr: {exc}", file=sys.stderr)
        return 2

