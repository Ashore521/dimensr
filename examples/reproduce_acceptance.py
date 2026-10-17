"""Repeat one preselected fit independently and compare all non-timing state."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from acceptance_benchmark import ROOT, fingerprint, read, sha, write
from dimensr import DimenSRRegressor

JOB = "Nguyen-1_s11_n0_full"
TIMING_KEYS = {"wall_seconds", "elapsed_seconds", "fit_seconds"}


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def state_hash(value):
    return hashlib.sha256(json.dumps(without_timing(value), sort_keys=True).encode()).hexdigest()


def reproduce():
    source = ROOT / "results/acceptance_v1"
    out = ROOT / "outputs/acceptance_repro"
    protocol = read(source / "protocol.json")
    assert protocol["source_hashes"] == fingerprint()
    assert all((source / "runs" / f"{j['id']}.json").exists() for j in protocol["jobs"])
    job = next(j for j in protocol["jobs"] if j["id"] == JOB)
    for folder in ("data", "models", "runs"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for relative in ("protocol.json", f"data/{job['data_key']}_fit.npz"):
        if not (out / relative).exists():
            shutil.copyfile(source / relative, out / relative)
        assert sha(source / relative) == sha(out / relative)
    run_path = out / "runs" / f"{JOB}.json"
    if not run_path.exists():
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        command = [sys.executable, str(Path(__file__).with_name("acceptance_benchmark.py")),
                   "worker", "--out", str(out), "--job", JOB]
        with (out / "repeat.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=300, env=env)
                if not run_path.exists():
                    write(run_path, dict(job, status="crashed", returncode=completed.returncode))
            except subprocess.TimeoutExpired:
                write(run_path, dict(job, status="timeout"))
    repeated = read(run_path)
    original = read(source / "runs" / f"{JOB}.json")
    result = dict(job=JOB, scope="one preselected deterministic fit; not a repeat of the whole study",
                  protocol_sha256=sha(source / "protocol.json"), repeat_status=repeated["status"],
                  excluded_timing_keys=sorted(TIMING_KEYS),
                  exact_run_state_match=without_timing(original) == without_timing(repeated))
    evidence = source / "repeat"
    evidence.mkdir(exist_ok=True)
    shutil.copyfile(run_path, evidence / "run.json")
    if repeated["status"] == "ok":
        relative = f"models/{JOB}.json"
        before, after = read(source / relative), read(out / relative)
        first = DimenSRRegressor.load(source / relative)
        second = DimenSRRegressor.load(out / relative)
        equal = {}
        # Training has ended before the saved models see any holdout inputs.
        for group, splits in (("fit", ("train", "val")), ("holdout", ("test", "ood", "probe"))):
            with np.load(source / "data" / f"{job['data_key']}_{group}.npz") as data:
                for split in splits:
                    equal[split] = bool(np.array_equal(first.predict(data[f"X_{split}"]),
                                                       second.predict(data[f"X_{split}"]), equal_nan=True))
        result.update(exact_model_state_match=without_timing(before) == without_timing(after),
                      exact_best_match=before["best"] == after["best"],
                      exact_pareto_match=before["pareto_front"] == after["pareto_front"],
                      exact_predictions_match=equal,
                      original_model_state_sha256=state_hash(before),
                      repeat_model_state_sha256=state_hash(after),
                      original_run_state_sha256=state_hash(original),
                      repeat_run_state_sha256=state_hash(repeated))
        shutil.copyfile(out / relative, evidence / "model.json")
    result["passed"] = bool(result["exact_run_state_match"] and result.get("exact_model_state_match")
                            and all(result.get("exact_predictions_match", {}).values()))
    result["evidence_hashes"] = {p.name: sha(p) for p in evidence.glob("*.json")}
    write(source / "reproducibility.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise SystemExit("Independent repeat differs or did not finish; see reproducibility.json")


if __name__ == "__main__":
    reproduce()
