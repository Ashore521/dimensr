"""Frozen 0.3 / search-only / 0.4 comparison: prepare -> run -> score -> verify.

Six reused regression tasks with fresh data and seeds; not blind task discovery
or equal-compute benchmarking. Fit workers never open holdouts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "cb520c96daf63922b31009e8382ea3a436d731cf"
OLD = ROOT / "outputs/interval_baseline_v03"
CASES = Path(__file__).with_name("interval_search_cases.json")
VARIANTS = ("v03", "search_only", "v04")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    paths = sorted((ROOT / "src/dimensr").glob("*.py")) + [Path(__file__), CASES]
    # Source line endings are normalized so a normal Git checkout is portable.
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in paths}


def ensure_baseline():
    archive = subprocess.check_output(["git", "archive", "--format=zip", BASELINE, "src"], cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            path = OLD / name
            content = z.read(name)
            if path.exists() and path.read_bytes() != content:
                raise RuntimeError("Baseline source was modified")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    return {p.relative_to(OLD).as_posix(): sha(p) for p in sorted((OLD / "src/dimensr").glob("*.py"))}


def permitted(X, case):
    import numpy as np
    if case["domain"] == "positive_total_mass":
        return (X[:, 0] >= 0) & (X[:, 1] >= 0) & (X[:, 0] + X[:, 1] > 0)
    if case["domain"] == "subluminal":
        return (X[:, 0] >= 0) & (X[:, 2] > 0) & (np.abs(X[:, 1]) < X[:, 2])
    return np.ones(len(X), dtype=bool)


def domain_points(case):
    import numpy as np
    axes = [sorted(set([lo, (lo + hi)/2, hi])) for lo, hi in case["domain_bounds"]]
    X = np.asarray(list(itertools.product(*axes)))
    return X[permitted(X, case)]


def generate(case, case_index, seed):
    import numpy as np
    import sympy as sp
    fn = sp.lambdify(sp.symbols(case["names"]), sp.sympify(case["truth"]), "numpy")
    fit, holdout = {}, {}
    for j, (split, n) in enumerate((("train", 192), ("val", 96), ("test", 768), ("ood", 768), ("stress", 512))):
        rng = np.random.default_rng(np.random.SeedSequence([20261023, case_index, seed, j]))
        bounds = np.asarray(case["ood"] if split == "ood" else case["domain_bounds"] if split == "stress" else case["bounds"])
        X = rng.uniform(bounds[:, 0], bounds[:, 1], (n, len(bounds)))
        if split == "stress":
            # Independent points on faces and near zero; no filtering by truth.
            for i in range(len(bounds)):
                rows = np.arange(i, n, len(bounds))
                X[rows[::3], i] = bounds[i, 0]
                X[rows[1::3], i] = bounds[i, 1]
                if bounds[i, 0] == 0:
                    X[rows[2::3], i] = 1e-10
            X = X[permitted(X, case)]
        dest = fit if split in ("train", "val") else holdout
        dest[f"X_{split}"] = X
        dest[f"y_{split}"] = np.broadcast_to(fn(*X.T), (len(X),)).copy()
        if not np.isfinite(dest[f"y_{split}"]).all():
            raise ValueError("Input contract produced invalid ground truth")
    fit["X_domain"] = domain_points(case)
    holdout["X_boundary"] = fit["X_domain"]
    holdout["y_boundary"] = np.broadcast_to(fn(*fit["X_domain"].T), (len(fit["X_domain"]),)).copy()
    return fit, holdout


def prepare(out):
    import numpy as np
    if (out / "protocol.json").exists():
        raise ValueError("Protocol exists; resume without preparing again")
    baseline_hashes = ensure_baseline()
    cases = read(CASES)
    jobs, hashes = [], {}
    for i, (name, case) in enumerate(cases.items()):
        for seed in (71, 82, 93):
            key = f"{name}_s{seed}"
            for group, data in zip(("fit", "holdout"), generate(case, i, seed)):
                path = out / "data" / f"{key}_{group}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **data)
                hashes[path.relative_to(out).as_posix()] = sha(path)
            for variant in VARIANTS:
                jobs.append(dict(id=f"{key}_{variant}", data_key=key, case=name, seed=seed, variant=variant))
    cfg = dict(population_size=48, epochs=8, gp_generations=2, selection_generations=6,
               max_nodes=21, max_depth=7, max_evaluations=2048, rademacher_draws=32,
               operators=["add", "sub", "mul", "div", "square", "sqrt", "sin", "cos", "exp", "log"])
    import importlib.metadata as md
    protocol = dict(frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    source_hashes=fingerprint(), baseline_commit=BASELINE, baseline_hashes=baseline_hashes,
                    dataset_hashes=hashes, config=cfg, cases=cases, jobs=jobs, timeout_seconds=240,
                    workers=3, noise=0, environment=dict(python=sys.version, platform=platform.platform(),
                        packages={p: md.version(p) for p in ("numpy", "scipy", "torch", "sympy", "scikit-learn")}),
                    comparison="Same unique-tree ceiling; count feature previews, pair screening, projections, linear fits and interval boxes separately. Not equal compute.",
                    domain_information="All variants receive identical unlabelled domain points and declared boxes spanning the interpolation/extrapolation ranges. Zero-mass faces outside the positive-mass box use explicit probes only.",
                    primary="Final raw-tree NRMSE < 1e-6 on all four holdout splits; no invalid intermediate.",
                    secondary="Any saved Pareto candidate with the same criterion; final IID R2 > .999; timings and extra work.",
                    scope="Six reused regression tasks, fresh samples and seeds, no noise. Extended input contract is supplied to all methods. No blind task or superiority claim.")
    write(out / "protocol.json", protocol)
    print(f"Frozen {len(jobs)} fits before fitting or holdout scoring", flush=True)


def checked(out):
    p = read(out / "protocol.json")
    if fingerprint() != p["source_hashes"]:
        raise RuntimeError("Frozen source changed; use a new study directory")
    if any(sha(out / name) != digest for name, digest in p["dataset_hashes"].items()):
        raise RuntimeError("Frozen data changed")
    return p


def worker(out, job_id):
    p = checked(out)
    job = next(j for j in p["jobs"] if j["id"] == job_id)
    source = OLD / "src" if job["variant"] == "v03" else ROOT / "src"
    if job["variant"] == "v03" and any(sha(OLD / k) != v for k, v in p["baseline_hashes"].items()):
        raise RuntimeError("Frozen baseline changed")
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from dimensr import DimenSRRegressor, SearchConfig
    torch.set_num_threads(1)
    case = p["cases"][job["case"]]
    cfg = dict(p["config"], random_state=job["seed"])
    if job["variant"] != "v03":
        cfg["interval_checks"] = job["variant"] == "v04"
    result = dict(job, status="error", protocol_sha256=sha(out / "protocol.json"), config=cfg)
    start = time.monotonic()
    try:
        # No holdout file or ground-truth evaluator is used by this worker.
        with np.load(out / "data" / f"{job['data_key']}_fit.npz") as data:
            domain = dict(X_domain=data["X_domain"], input_bounds=case["interval_bounds"])
            model = DimenSRRegressor(SearchConfig(**cfg), feature_names=case["names"],
                                    feature_units=case["units"], target_units=case["target"])
            model.fit(data["X_train"], data["y_train"], X_val=data["X_val"], y_val=data["y_val"], **domain)
        path = out / "models" / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        result.update(status="ok", model_sha256=sha(path), diagnostics=model.diagnostics_)
    except Exception:
        result["error"] = traceback.format_exc()
    result["elapsed_seconds"] = time.monotonic() - start
    write(out / "runs" / f"{job_id}.json", result)


def run(out):
    p = checked(out)
    if ensure_baseline() != p["baseline_hashes"]:
        raise RuntimeError("Baseline mismatch")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8")

    def one(job):
        result_path = out / "runs" / f"{job['id']}.json"
        if result_path.exists():
            return job["id"], "retained"
        log = out / "logs" / f"{job['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            try:
                proc = subprocess.run([sys.executable, __file__, "worker", "--out", str(out), "--job", job["id"]],
                                      cwd=ROOT, env=env, stdout=stream, stderr=stream, timeout=p["timeout_seconds"])
                status = "error" if proc.returncode else "missing_result"
            except subprocess.TimeoutExpired:
                status = "timeout"
        if not result_path.exists():
            write(result_path, dict(job, status=status, protocol_sha256=sha(out / "protocol.json")))
        return job["id"], read(result_path)["status"]

    with ThreadPoolExecutor(max_workers=p["workers"]) as pool:
        futures = [pool.submit(one, j) for j in p["jobs"]]
        for i, future in enumerate(as_completed(futures), 1):
            name, status = future.result()
            print(f"{i}/{len(futures)} {name}: {status}", flush=True)


def metrics(tree, constants, X, y):
    import numpy as np
    prediction = tree.evaluate(X, constants)
    invalid = int((~np.isfinite(prediction)).sum())
    if invalid:
        return dict(invalid=invalid, nrmse=None, r2=None)
    with np.errstate(all="ignore"):
        mse = float(np.mean((prediction-y)**2))
    if not np.isfinite(mse):
        return dict(invalid=0, nrmse=None, r2=None)
    scale = float(np.std(y))
    scale = scale if scale > 1e-12 else max(float(np.sqrt(np.mean(y*y))), 1.)
    return dict(invalid=0, nrmse=float(np.sqrt(mse)/scale), r2=float(1-mse/scale**2))


def score(out):
    import numpy as np
    from dimensr.expression import Node
    from dimensr.units import unit
    p = checked(out)
    if not all((out / "runs" / f"{j['id']}.json").exists() for j in p["jobs"]):
        raise RuntimeError("Complete all fits before opening holdouts")
    rows, evidence = [], {}
    for job in p["jobs"]:
        run_path = out / "runs" / f"{job['id']}.json"
        run = read(run_path)
        if run["protocol_sha256"] != sha(out / "protocol.json"):
            raise RuntimeError("Run provenance mismatch")
        evidence[run_path.relative_to(out).as_posix()] = sha(run_path)
        row = dict(job, status=run["status"])
        if run["status"] == "ok":
            model_path = out / "models" / f"{job['id']}.json"
            if sha(model_path) != run["model_sha256"]:
                raise RuntimeError("Saved model changed")
            evidence[model_path.relative_to(out).as_posix()] = sha(model_path)
            model = read(model_path)
            case = p["cases"][job["case"]]
            scores = []
            with np.load(out / "data" / f"{job['data_key']}_holdout.npz") as data:
                for candidate in [model["best"]] + model["pareto_front"]:
                    tree = Node.from_dict(candidate["tree"])
                    if tree.units != unit(case["target"]):
                        raise RuntimeError("Target unit violation")
                    for _, node in tree.walk():
                        if not node.children:
                            expected = case["units"][int(node.token[1:])] if node.token.startswith("x") else model["constant_units"][int(node.token[1:])]
                            if node.units != unit(expected):
                                raise RuntimeError("Terminal unit violation")
                    if tree.size > p["config"]["max_nodes"] or tree.depth > p["config"]["max_depth"]:
                        raise RuntimeError("Tree budget exceeded")
                    values = {s: metrics(tree, candidate["constants"], data[f"X_{s}"], data[f"y_{s}"])
                              for s in ("test", "ood", "boundary", "stress")}
                    recovered = all(v["nrmse"] is not None and v["nrmse"] < 1e-6 for v in values.values())
                    scores.append(dict(metrics=values, recovered=recovered))
            d = model["diagnostics"]
            if d["evaluations"] > p["config"]["max_evaluations"]:
                raise RuntimeError("Evaluation budget exceeded")
            row.update(final=scores[0], any_candidate_recovered=any(s["recovered"] for s in scores[1:]),
                       candidate_count=len(scores)-1, expression=model["best"]["expression"],
                       candidate_domain_failures=sum(any(v["invalid"] for v in s["metrics"].values()) for s in scores[1:]),
                       elapsed_seconds=run["elapsed_seconds"], evaluations=d["evaluations"],
                       structured_search=d.get("structured_search", {}),
                       interval_statuses=d.get("interval_statuses", {}),
                       interval_boxes_checked=d.get("interval_boxes_checked", 0))
        rows.append(row)
    result = dict(protocol_sha256=sha(out / "protocol.json"), evidence_hashes=evidence,
                  scorer="0.4 raw-tree semantics for all variants, including every intermediate",
                  unit_violations=0, rows=rows)
    destination = out / "scores.json"
    if destination.exists() and read(destination) != result:
        raise RuntimeError("Scores differ from saved results")
    write(destination, result)
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        ok = [r for r in selected if r["status"] == "ok"]
        print(variant, "fits", len(ok), "/", len(selected), "recovery", sum(r["final"]["recovered"] for r in ok),
              "domain failures", sum(any(v["invalid"] for v in r["final"]["metrics"].values()) for r in ok))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker", "score", "verify"))
    parser.add_argument("--out", type=Path, default=ROOT / "results/interval_search_v4")
    parser.add_argument("--job")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "worker":
        worker(out, args.job)
    elif args.command in ("score", "verify"):
        score(out)
    else:
        {"prepare": prepare, "run": run}[args.command](out)
