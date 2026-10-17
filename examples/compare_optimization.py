"""Frozen, resumable comparison against the original grouped real-data pilot.

python examples/compare_optimization.py run
python examples/compare_optimization.py verify
Each fit is checkpointed. Test labels are scored only after every fit completes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import sklearn
import torch

from dimensr import DimenSRRegressor, SearchConfig
from dimensr.expression import Node
from dimensr.real_data import (DATASETS, audit_tree, code_hash, digest, fit_symbolic,
                               grouped_split, load_dataset, metrics, scaling, write_json)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_hash(value):
    return digest(json.dumps(value, sort_keys=True).encode())


def contexts(args):
    for name in DATASETS:
        X, y, groups, info = load_dataset(name, args.cache)
        assert info["member_sha256"] == read(args.baseline / "datasets.json")[name]["member_sha256"]
        for seed in (0, 1, 2):
            saved = read(args.baseline / "splits" / f"{name}_{seed}.json")
            split = grouped_split(groups, seed)
            assert saved["indices"] == {k: v.tolist() for k, v in split.items()}
            Xm, ym, scales = scaling(X, y, split["train"], DATASETS[name])
            assert scales == saved["preprocessing"]
            yield name, seed, Xm, ym, y, split, scales, info, saved


def protocol(args):
    spec = read(args.config)
    configs = {name: asdict(SearchConfig(**(spec["common"] | patch)))
               for name, patch in spec["variants"].items()}
    for config in configs.values():
        SearchConfig(**config).validate()
    baseline_hashes = {str(p.relative_to(args.baseline)): json_hash(read(p))
                       for p in sorted(args.baseline.rglob("*.json"))}
    value = {"schema_version": 1, "variants": configs, "study_spec": spec,
             "seeds": [0, 1, 2], "datasets": list(DATASETS), "source_sha256": code_hash(),
             "runner_sha256": digest(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
             "baseline_hashes": baseline_hashes,
             "baseline_commit": "5492122",
             "test_policy": "Exploratory reuse of previously inspected test sets; all configurations frozen before reruns. "
                            "Score test labels only after all fits. Report all variants, no test-selected configuration.",
             "budget_policy": "21-node variant keeps original population/epochs/generations/node/evaluation ceilings; "
                              "new fit work and retries differ. 31/41 share expanded settings. Not equal wall-time budgets.",
             "physical_policy": "Known nonnegative strength/resistance; train-only convex-combination probes. "
                                "No test feature ranges or labels used to construct constraints. Never clip test predictions."}
    fingerprint = json_hash(value)
    path = args.output / "protocol.json"
    if path.exists():
        previous = read(path)
        if previous["fingerprint"] != fingerprint:
            old = {k: v for k, v in previous.items() if k not in ("fingerprint", "frozen_at", "runner_sha256")}
            new = {k: v for k, v in value.items() if k != "runner_sha256"}
            if not args.resume_dispatch or json_hash(old) != json_hash(new):
                raise ValueError("Source, baseline or protocol changed; use a new output directory.")
            # An explicit dispatch-only amendment preserves the original protocol
            # and every completed fit. Algorithm/config/data hashes must match.
            write_json(args.output / "dispatch_amendment.json", {
                "protocol_fingerprint": previous["fingerprint"],
                "original_runner_sha256": previous["runner_sha256"],
                "current_runner_sha256": value["runner_sha256"], "workers": args.workers,
                "reason": "Parallel scheduling only; identical algorithm, configurations, splits and test barrier.",
                "recorded_at": datetime.now(timezone.utc).isoformat()})
    else:
        write_json(path, value | {"fingerprint": fingerprint, "frozen_at": datetime.now(timezone.utc).isoformat()})
    return read(path)


def physical(prediction, indices, bounds):
    finite = np.isfinite(prediction)
    bad = ~finite
    if bounds is not None:
        lower, upper = bounds
        if lower is not None:
            bad |= prediction < lower
        if upper is not None:
            bad |= prediction > upper
    return {"nonfinite_count": int((~finite).sum()), "violations": int(bad.sum()),
            "violating_original_rows": indices[bad].tolist(),
            "min": float(prediction[finite].min()) if finite.any() else None,
            "max": float(prediction[finite].max()) if finite.any() else None}


def fit_job(args, p, environment, context, variant):
    torch.set_num_threads(1)
    name, seed, Xm, ym, y, split, scales, info, saved = context
    spec = p["study_spec"]
    config = p["variants"][variant]
    key = f"{name}_{seed}_{variant}"
    path = args.output / "runs" / f"{key}.json"
    cfg = replace(SearchConfig(**config), random_state=seed, output_bounds=spec["physical_bounds"][name])
    row = {"dataset": name, "seed": seed, "method": variant, "environment": environment,
           "execution_workers": args.workers,
           "protocol_fingerprint": p["fingerprint"], "split_sha256": json_hash(saved),
           "target_metric_unit": DATASETS[name]["y_unit"]}
    log = args.output / "logs" / f"{key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
        print(f"START {key}", flush=True)
        started = time.monotonic()
        try:
            model, detail = fit_symbolic(Xm, ym, split, DATASETS[name], cfg, True,
                                        args.output / "models" / f"{key}.json",
                                        include_normalizers=spec["include_normalizers"])
            row.update(detail)
            row.update(status="fit_ok", fit_seconds=time.monotonic() - started)
            row["scores"] = {}
            for part in ("train", "validation"):
                indices = split[part]
                pred = model.best_.tree.evaluate(Xm[indices], model.best_.constants) * scales["y_model_to_native"]
                row["scores"][part] = metrics(y[indices], pred, float(np.std(y[split["train"]])))
        except (ValueError, RuntimeError, AssertionError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       fit_seconds=time.monotonic() - started)
        write_json(path, row)
    return key, row


def run(args):
    torch.set_num_threads(1)
    p = protocol(args)
    spec = p["study_spec"]
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                   "platform": platform.platform(), "torch_threads": 1}
    all_data = {}
    jobs = []
    # This entire loop has no test scoring. Each completed fit is independently resumable.
    for context in contexts(args):
        name, seed, Xm, ym, y, split, scales, info, saved = context
        all_data[name] = info
        write_json(args.output / "splits" / f"{name}_{seed}.json", saved)
        for variant, config in p["variants"].items():
            key = f"{name}_{seed}_{variant}"
            path = args.output / "runs" / f"{key}.json"
            if path.exists():
                assert read(path)["protocol_fingerprint"] == p["fingerprint"]
                print(f"RESUME skip {key}", flush=True)
                continue
            jobs.append((context, variant))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_job, args, p, environment, context, variant) for context, variant in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            key, row = future.result()
            print(f"FIT {done}/{len(jobs)} {key}: {row['status']}, "
                  f"validation R2={row.get('scores', {}).get('validation', {}).get('r2')}; "
                  f"{row['fit_seconds']:.1f}s", flush=True)
    write_json(args.output / "datasets.json", all_data)
    print("All fits complete; now scoring frozen test sets.", flush=True)
    for name, seed, Xm, ym, y, split, scales, info, saved in contexts(args):
        for variant in p["variants"]:
            path = args.output / "runs" / f"{name}_{seed}_{variant}.json"
            row = read(path)
            if row["status"] == "failed":
                continue
            model = DimenSRRegressor.load(args.output / row["model_file"])
            row["physical_checks"] = {}
            for part, indices in split.items():
                pred = model.best_.tree.evaluate(Xm[indices], model.best_.constants) * scales["y_model_to_native"]
                row["scores"][part] = metrics(y[indices], pred, float(np.std(y[split["train"]])))
                row["physical_checks"][part] = physical(pred, indices, spec["physical_bounds"][name])
                if part == "test":
                    row["test_predictions"] = [float(v) if np.isfinite(v) else None for v in pred]
            row["status"] = "ok"
            write_json(path, row)
            print(f"TEST {name} seed={seed} {variant}: {row['scores']['test']}", flush=True)
    summarize(args, p)


def summarize(args, p):
    rows = [read(path) for path in sorted((args.output / "runs").glob("*.json"))]
    rows += [read(path) for path in sorted((args.baseline / "runs").glob("*.json"))
             if read(path)["method"] in ("full", "ridge", "random_forest")]
    summaries = []
    for name in DATASETS:
        for method in ["full", "ridge", "random_forest"] + list(p["variants"]):
            subset = [r for r in rows if r["dataset"] == name and r["method"] == method]
            valid = [r for r in subset if r["status"] == "ok" and r["scores"]["test"]["status"] == "ok"]
            entry = {"dataset": name, "method": method, "runs": len(subset), "valid_test_runs": len(valid)}
            if valid:
                for part in ("train", "validation", "test"):
                    entry[part] = {}
                    for metric in ("r2", "rmse", "mae"):
                        values = [r["scores"][part][metric] for r in valid]
                        entry[part][metric] = {"mean": float(np.mean(values)),
                                               "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
                entry["mean_fit_seconds"] = float(np.mean([r["fit_seconds"] for r in valid]))
                if all("diagnostics" in r for r in valid):
                    entry["nodes"] = [r["nodes"] for r in valid]
                    for key in ("evaluations", "constant_residual_calls"):
                        entry[f"mean_{key}"] = float(np.mean([r["diagnostics"][key] for r in valid]))
                    entry["unit_violations"] = sum(r["unit_audit"]["violations"] for r in valid)
                if all("physical_checks" in r for r in valid):
                    entry["test_physical_violations"] = sum(r["physical_checks"]["test"]["violations"] for r in valid)
            summaries.append(entry)
    write_json(args.output / "summary.json", {"protocol_fingerprint": p["fingerprint"], "summary": summaries})


def verify(args):
    p = read(args.output / "protocol.json")
    verified = failed = candidates = 0
    for relative, expected in p["baseline_hashes"].items():
        assert json_hash(read(args.baseline / relative)) == expected
    for name, seed, Xm, ym, y, split, scales, info, saved in contexts(args):
        for variant in p["variants"]:
            row = read(args.output / "runs" / f"{name}_{seed}_{variant}.json")
            assert row["protocol_fingerprint"] == p["fingerprint"]
            assert row["split_sha256"] == json_hash(saved)
            if row["status"] == "failed":
                failed += 1
                continue
            assert row["status"] == "ok"
            model = DimenSRRegressor.load(args.output / row["model_file"])
            assert row["unit_audit"]["violations"] == 0
            assert row["unit_audit"]["unique_evaluated_trees"] == row["diagnostics"]["evaluations"]
            for candidate in model.pareto_front_:
                audit_tree(Node.from_dict(candidate["tree"]), model.feature_units_, model.target_units_, model.constant_units_)
                candidates += 1
            for part, indices in split.items():
                pred = model.best_.tree.evaluate(Xm[indices], model.best_.constants) * scales["y_model_to_native"]
                score = metrics(y[indices], pred, float(np.std(y[split["train"]])))
                for key, value in score.items():
                    if isinstance(value, str):
                        assert value == row["scores"][part][key]
                    else:
                        np.testing.assert_allclose(value, row["scores"][part][key], rtol=1e-10, atol=1e-10)
                assert physical(pred, indices, p["study_spec"]["physical_bounds"][name]) == row["physical_checks"][part]
                if part == "test":
                    np.testing.assert_allclose(pred, np.asarray(row["test_predictions"], dtype=float),
                                               rtol=1e-10, atol=1e-10, equal_nan=True)
            verified += 1
    result = {"verified_models": verified, "verified_pareto_models": candidates,
              "failed_runs": failed, "baseline_unchanged": True}
    write_json(args.output / "verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("--config", type=Path, default=Path("examples/optimized_real_data_configs.json"))
    parser.add_argument("--baseline", type=Path, default=Path("results/real_data"))
    parser.add_argument("--output", type=Path, default=Path("results/optimization_v2"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/real_data/cache"))
    parser.add_argument("--workers", type=int, default=1, help="Independent single-thread fitting processes")
    parser.add_argument("--resume-dispatch", action="store_true",
                        help="Explicitly record a runner-only amendment; algorithm/config/data must match")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    (run if args.action == "run" else verify)(args)
