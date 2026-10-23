"""Replay the preselected first full-new-version fit, preserving all evidence."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from dimensr import DimenSRRegressor
import interval_search_study as study


def without_time(value):
    if isinstance(value, dict):
        return {k: without_time(v) for k, v in value.items() if k != "elapsed_seconds"}
    if isinstance(value, list):
        return [without_time(v) for v in value]
    return value


def replay(out, scratch):
    protocol = study.checked(out)
    job = next(j for j in protocol["jobs"] if j["variant"] == "v04")
    scratch.mkdir(parents=True, exist_ok=False)
    shutil.copy2(out / "protocol.json", scratch / "protocol.json")
    shutil.copytree(out / "data", scratch / "data")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, str(Path(study.__file__)), "worker", "--out", str(scratch), "--job", job["id"]],
                   env=env, cwd=study.ROOT, timeout=protocol["timeout_seconds"], check=True)
    original = out / "models" / f"{job['id']}.json"
    repeated = scratch / "models" / original.name
    assert without_time(study.read(original)) == without_time(study.read(repeated))
    a, b = DimenSRRegressor.load(original), DimenSRRegressor.load(repeated)
    with np.load(out / "data" / f"{job['data_key']}_holdout.npz") as data:
        for split in ("test", "ood", "boundary", "stress"):
            np.testing.assert_array_equal(a.predict(data[f"X_{split}"]), b.predict(data[f"X_{split}"]))
    (out / "replay").mkdir(exist_ok=True)
    destination = out / "replay" / original.name
    if destination.exists() or (out / "reproducibility.json").exists():
        raise RuntimeError("Replay evidence already exists; do not overwrite")
    shutil.copy2(repeated, destination)
    result = dict(status="verified", job=job["id"], ignored_fields=["elapsed_seconds"],
                  original_sha256=study.sha(original), repeated_sha256=study.sha(repeated),
                  repeated_model=destination.relative_to(out).as_posix(),
                  replay_script_sha256=study.sha(__file__),
                  scope="Preselected one-run replay; all non-timing state and all four prediction arrays exactly identical")
    study.write(out / "reproducibility.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=study.ROOT / "results/interval_search_v4")
    parser.add_argument("--scratch", type=Path, default=study.ROOT / "outputs/interval_replay")
    args = parser.parse_args()
    replay(args.out.resolve(), args.scratch.resolve())
