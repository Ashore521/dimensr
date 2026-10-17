"""Frozen, restartable acceptance pilot. Upstream PhySO source is never patched.

prepare -> run (isolated subprocesses, bounded timeout) -> score -> verify.
Only run reads training/validation data; test scoring is a separate command.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from fractions import Fraction
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = Path(__file__).with_name("acceptance_cases.json")
OPS = ("add", "sub", "mul", "div", "square", "sqrt", "sin", "cos", "exp", "log")
PHY_SHA = "bfbfa88d5def09d3d0cd47f3e2252b5a25836721"
VARIANTS = {"no_rnn": "use_rnn", "no_gp": "use_gp", "no_units": "use_units",
            "no_sd": "use_size_diversity", "no_rad": "use_rademacher"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    paths = list((ROOT / "src/dimensr").glob("*.py")) + [Path(__file__), CASES_FILE]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}


def config(seed, variant):
    from dimensr import SearchConfig
    values = dict(population_size=64, epochs=10, gp_generations=3, selection_generations=8,
                  max_nodes=21, max_depth=7, max_evaluations=4096, operators=OPS,
                  random_state=seed, rademacher_draws=32)
    if variant in VARIANTS:
        values[VARIANTS[variant]] = False
    return SearchConfig(**values)


def generate(case, case_id, seed, noise):
    import numpy as np
    import sympy as sp
    symbols = [sp.Symbol(n) for n in case["names"]]
    truth = sp.sympify(case["truth"], locals=dict(zip(case["names"], symbols)))
    fn = sp.lambdify(symbols, truth, "numpy")
    data = {}
    for j, (split, n) in enumerate((("train", 256), ("val", 128), ("test", 1024), ("ood", 1024))):
        rng = np.random.default_rng(np.random.SeedSequence([20261017, case_id, seed, j]))
        bounds = np.array(case["ood"] if split == "ood" else case["bounds"])
        X = rng.uniform(bounds[:, 0], bounds[:, 1], (n, len(symbols)))
        y = np.asarray(fn(*X.T), dtype=float)
        if split in ("train", "val"):
            # Same clean inputs and noise direction for both noise levels and all algorithms.
            y = y + noise * np.sqrt(np.mean(y**2)) * rng.normal(size=n)
        data[f"X_{split}"], data[f"y_{split}"] = X, y
    # Exact boundaries and zero are essential: simplification can hide raw-tree holes.
    axes = [sorted(set([lo, (lo + hi) / 2, hi, 0.])) for lo, hi in case["bounds"]]
    probes = np.array(list(itertools.product(*axes)))
    with np.errstate(all="ignore"):
        yp = np.asarray(fn(*probes.T), dtype=float)
    valid = np.isfinite(yp)
    data["X_probe"], data["y_probe"] = probes[valid], yp[valid]
    return data


def prepare(out):
    import numpy as np
    if (out / "protocol.json").exists():
        raise ValueError("Protocol exists; choose a new directory or resume run.")
    cases = read(CASES_FILE)
    jobs, hashes = [], {}
    for case_id, (name, case) in enumerate(cases.items()):
        for seed in (11, 22, 33):
            for noise in (0., .01):
                key = f"{name}_s{seed}_n{noise:g}"
                data = generate(case, case_id, seed, noise)
                for group, splits in (("fit", ("train", "val")), ("holdout", ("test", "ood", "probe"))):
                    path = out / "data" / f"{key}_{group}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **{k: v for k, v in data.items() if k.split("_")[-1] in splits})
                    hashes[str(path.relative_to(out)).replace("\\", "/")] = sha(path)
                variants = ["full", "physo"]
                if noise == .01 and name in ("Nguyen-7", "Feynman-I.18.4"):
                    variants += list(VARIANTS)
                for variant in variants:
                    jobs.append(dict(id=f"{key}_{variant}", data_key=key, case=name,
                                     seed=seed, noise=noise, variant=variant))
    protocol = dict(version=1, frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    scope="small adapted-budget pilot, not a paper reproduction or equal-compute ranking",
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_hashes=fingerprint(), dataset_hashes=hashes, cases=cases, jobs=jobs,
                    dimensr_config=asdict(config(11, "full")), physo_commit=PHY_SHA,
                    physo_overrides=dict(base="config1", batch_size=256, epochs=128, max_nodes=21,
                                         min_nodes=1, max_n_evaluations=4096, stop_reward=1.1,
                                         protected_ops=False, parallel_mode=False, cpu_threads=1),
                    seed_rule="same case/seed data across variants; independent case/seed/split RNG streams",
                    primary="final selection by validation NRMSE + 0.01*nodes/21 for both methods",
                    secondary="any stored Pareto candidate; truth is accessed only after all fits",
                    gates=dict(numeric_recovery_nrmse=1e-6, accurate_iid_r2=.999,
                               minimum_noiseless_core_recovery=1.,
                               noiseless_suite_recovery_target=.75, noisy_accurate_iid_target=.75,
                               final_domain_failures_allowed=0, dimensional_violations_allowed=0),
                    workers=4, timeout_seconds=300)
    write(out / "protocol.json", protocol)
    print(f"Frozen {len(jobs)} jobs; {len(hashes)} dataset artifacts")


def attach_node(op, children):
    from dimensr.expression import Node
    from dimensr.units import ZERO, add, sub, scale
    dims = [c.units for c in children]
    units = (add(*dims) if op == "mul" else sub(*dims) if op == "div" else
             scale(dims[0], 2) if op == "square" else scale(dims[0], Fraction(1, 2)) if op == "sqrt" else
             ZERO if op in ("sin", "cos", "exp", "log") else dims[0])
    return Node(op, tuple(children), units)


def physo_tree(program, case):
    """Export prefix tokens without simplifying away numerical domains."""
    from dimensr.expression import Node
    from dimensr.units import unit, ZERO
    tokens = iter(program.tokens)
    constants = []

    def visit():
        t = next(tokens)
        if t.arity:
            return attach_node("square" if t.name == "n2" else t.name, [visit() for _ in range(t.arity)])
        if t.is_input_var:
            return Node(f"x{t.var_id}", (), unit(case["units"][t.var_id]))
        value = program.free_consts.class_values[0, t.var_id] if t.is_class_free_const else t.fixed_const
        constants.append(float(value))
        return Node("c0", (), ZERO)

    tree = visit()
    return tree, constants


def fit_physo(X, y, case, seed):
    import copy
    import numpy as np
    import torch
    import physo
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = copy.deepcopy(physo.config.config1.config1)
    cfg["learning_config"].update(batch_size=256, max_time_step=21)
    for name, args in cfg["priors_config"]:
        if name == "HardLengthPrior":
            args.update(min_length=1, max_length=21)
    monitor = physo.learn.monitoring
    started = time.monotonic()
    _, logger = physo.SR(X.T, y, X_names=case["names"], X_units=case["units"],
                         y_name="target", y_units=case["target"],
                         fixed_consts=[1.], fixed_consts_units=[[0]*7],
                         free_consts_names=["c1", "c2"], free_consts_units=[[0]*7]*2,
                         op_names=["n2" if s == "square" else s for s in OPS],
                         use_protected_ops=False, run_config=cfg, epochs=128,
                         stop_reward=1.1, max_n_evaluations=4096, parallel_mode=False,
                         get_run_logger=lambda: monitor.RunLogger(do_save=False),
                         get_run_visualiser=lambda: monitor.RunVisualiser(do_show=False, do_prints=False, do_save=False))
    seconds = time.monotonic() - started
    _, programs, _, _ = logger.get_pareto_front()
    candidates = []
    for prog in programs:
        tree, constants = physo_tree(prog, case)
        native = np.asarray(prog(torch.as_tensor(X.T)).detach().cpu()).reshape(-1)
        exported = tree.evaluate(X, constants)
        if not np.allclose(native, exported, rtol=1e-9, atol=1e-10, equal_nan=True):
            raise AssertionError("Native/exported PhySO predictions differ")
        candidates.append(dict(tree=tree.to_dict(), constants=constants,
                               nodes=tree.size, expression=str(tree.sympy(constants, case["names"]))))
    return candidates, dict(sampled_proposals=len(logger.epochs_history)*256,
                            rewarded_evaluations=int(sum(logger.n_rewarded)),
                            physical_proposals=int(sum(logger.n_physical)),
                            epochs=len(logger.epochs_history), fit_seconds=seconds)


def worker(out, job_id):
    import numpy as np
    import torch
    from dimensr import DimenSRRegressor
    from dimensr.expression import Node
    from dimensr.units import unit
    torch.set_num_threads(1)
    protocol = read(out / "protocol.json")
    if protocol["source_hashes"] != fingerprint():
        raise RuntimeError("Frozen source changed; use a new study directory.")
    job = next(j for j in protocol["jobs"] if j["id"] == job_id)
    case = protocol["cases"][job["case"]]
    result = dict(job)
    path = out / "data" / f"{job['data_key']}_fit.npz"
    if sha(path) != protocol["dataset_hashes"][str(path.relative_to(out)).replace("\\", "/")]:
        raise RuntimeError("Fit dataset fingerprint changed")
    data = np.load(path)
    X, y = data["X_train"], data["y_train"]
    Xv, yv = data["X_val"], data["y_val"]
    started = time.monotonic()
    try:
        if job["variant"] == "physo":
            candidates, diagnostics = fit_physo(X, y, case, job["seed"])
        else:
            model = DimenSRRegressor(config(job["seed"], job["variant"]),
                                    feature_names=case["names"], feature_units=case["units"], target_units=case["target"])
            # Audit every proposed tree at the evaluator boundary, without changing search.
            import dimensr.regressor as reg
            base = reg.Evaluator
            audits = {"proposals": 0, "violations": 0}

            class AuditedEvaluator(base):
                def __call__(self, tree, *args, **kwargs):
                    audits["proposals"] += 1
                    Node.from_dict(tree.to_dict())
                    expected = [0]*7 if job["variant"] == "no_units" else case["target"]
                    audits["violations"] += int(tree.units != unit(expected))
                    return super().__call__(tree, *args, **kwargs)

            reg.Evaluator = AuditedEvaluator
            try:
                model.fit(X, y, X_val=Xv, y_val=yv)
            finally:
                reg.Evaluator = base
            model.save(out / "models" / f"{job_id}.json")
            candidates = model.pareto_front_
            diagnostics = dict(model.diagnostics_, unit_audit=audits)
            result["native_selected"] = model.best_.to_dict(case["names"])
        scores = []
        physical_violations = 0
        for c in candidates:
            tree = Node.from_dict(c["tree"])
            pred = tree.evaluate(Xv, c["constants"])
            error = float(np.sqrt(np.mean((pred-yv)**2))/np.std(y))
            c["validation_nrmse"] = error if np.isfinite(error) else None
            score = error + .01*tree.size/21
            scores.append(score if np.isfinite(score) else float("inf"))
            physical_violations += int(tree.units != unit(case["target"]))
        if not scores or not np.isfinite(min(scores)):
            raise ValueError("No validation-valid Pareto candidate")
        result.update(status="ok", candidates=candidates, selected=int(np.argmin(scores)),
                      exported_target_unit_violations=physical_violations, diagnostics=diagnostics)
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    result["wall_seconds"] = time.monotonic()-started
    write(out / "runs" / f"{job_id}.json", result)
    print(job_id, result["status"], round(result["wall_seconds"], 2), flush=True)


def run(out, workers):
    protocol = read(out / "protocol.json")
    if fingerprint() != protocol["source_hashes"]:
        raise RuntimeError("Frozen source changed")
    (out / "models").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    def one(job):
        path = out / "runs" / (job["id"] + ".json")
        if path.exists():
            return job["id"], "retained"
        start = time.monotonic()
        with (out / "logs" / (job["id"] + ".log")).open("w", encoding="utf-8") as log:
            try:
                p = subprocess.run([sys.executable, __file__, "worker", "--out", str(out), "--job", job["id"]],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=protocol["timeout_seconds"])
                if not path.exists():
                    write(path, dict(job, status="crashed", returncode=p.returncode, wall_seconds=time.monotonic()-start))
            except subprocess.TimeoutExpired:
                write(path, dict(job, status="timeout", wall_seconds=time.monotonic()-start))
        return job["id"], read(path)["status"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, j) for j in protocol["jobs"]]
        for i, f in enumerate(as_completed(futures), 1):
            print(i, "/", len(futures), *f.result(), flush=True)


def metric(tree, constants, data, split):
    import numpy as np
    pred = tree.evaluate(data[f"X_{split}"], constants)
    target = data[f"y_{split}"]
    invalid = int((~np.isfinite(pred)).sum())
    if invalid:
        return dict(invalid=invalid, nrmse=None, r2=None)
    with np.errstate(all="ignore"):
        nrmse = float(np.sqrt(np.mean((pred-target)**2))/np.std(target))
    return dict(invalid=0, nrmse=nrmse if np.isfinite(nrmse) else None,
                r2=1-nrmse**2 if np.isfinite(nrmse) else None)


def symbolic(candidate, case, digits):
    """Coefficients rationalized at a declared precision, not fitted to the truth."""
    import sympy as sp
    from dimensr.expression import Node
    tree = Node.from_dict(candidate["tree"])
    expr = tree.sympy(candidate["constants"], case["names"])
    expr = expr.xreplace({f: sp.Rational(str(round(float(f), digits))) for f in expr.atoms(sp.Float)})
    symbols = {name: sp.Symbol(name) for name in case["names"]}
    truth = sp.sympify(case["truth"], locals=symbols)
    diff = sp.simplify(expr-truth)
    exact = diff == 0
    offset = not diff.free_symbols and diff.is_finite is True
    ratio = sp.simplify(expr/truth) if not offset else sp.nan
    scale = not ratio.free_symbols and ratio.is_finite is True and ratio != 0
    return dict(exact=bool(exact), constant_difference_or_ratio=bool(offset or scale))


def score(out):
    from dimensr.expression import Node
    import numpy as np
    protocol = read(out / "protocol.json")
    if fingerprint() != protocol["source_hashes"]:
        raise RuntimeError("Frozen source changed")
    if any(not (out / "runs" / f"{j['id']}.json").exists() for j in protocol["jobs"]):
        raise RuntimeError("Finish all fits before holdout scoring")
    rows = []
    for job in protocol["jobs"]:
        row = read(out / "runs" / f"{job['id']}.json")
        if row["status"] == "ok":
            path = out / "data" / f"{job['data_key']}_holdout.npz"
            if sha(path) != protocol["dataset_hashes"][str(path.relative_to(out)).replace("\\", "/")]:
                raise RuntimeError("Holdout data changed")
            data = np.load(path)
            audits = []
            for c in row["candidates"]:
                tree = Node.from_dict(c["tree"])
                audit = {split: metric(tree, c["constants"], data, split) for split in ("test", "ood", "probe")}
                audit["numerical_recovery"] = all(audit[s]["nrmse"] is not None and audit[s]["nrmse"] < 1e-6
                                                  for s in ("test", "ood", "probe"))
                audit["accurate_iid"] = audit["test"]["r2"] is not None and audit["test"]["r2"] > .999
                # Keep numerical/domain evidence separate from algebraic simplification.
                audit["symbolic_8digits"] = symbolic(c, protocol["cases"][job["case"]], 8)
                audit["symbolic_2digits"] = symbolic(c, protocol["cases"][job["case"]], 2)
                audits.append(audit)
            row["audit"] = audits
            row["selected_audit"] = audits[row["selected"]]
            row["pareto_any_recovery"] = any(a["numerical_recovery"] for a in audits)
            row["pareto_any_accurate_iid"] = any(a["accurate_iid"] for a in audits)
            if "native_selected" in row:
                c = row["native_selected"]
                tree = Node.from_dict(c["tree"])
                row["native_selected_audit"] = {s: metric(tree, c["constants"], data, s)
                                                 for s in ("test", "ood", "probe")}
        write(out / "scored" / f"{job['id']}.json", row)
        rows.append(row)
        print("scored", job["id"], flush=True)
    write(out / "summary.json", {"protocol_sha256": sha(out / "protocol.json"), "results": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "worker", "score"])
    parser.add_argument("--out", type=Path, default=ROOT / "results/acceptance_v1")
    parser.add_argument("--job")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.mode == "prepare": prepare(args.out)
    elif args.mode == "run": run(args.out, args.workers)
    elif args.mode == "worker": worker(args.out, args.job)
    else: score(args.out)
