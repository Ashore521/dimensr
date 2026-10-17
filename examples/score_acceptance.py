"""Bound symbolic diagnostics without changing the frozen fit or numeric scorer.

Timeout/error means indeterminate, never successful symbolic recovery. Checkpoints
retain that outcome on resume; numerical metrics still use the frozen raw trees.
"""
import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import acceptance_benchmark as benchmark


def symbolic_worker(pipe):
    # Import once; startup has a separate bound from each comparison.
    from dimensr.expression import Node  # noqa: F401
    import sympy  # noqa: F401
    pipe.send("ready")
    while True:
        request = pipe.recv()
        if request is None:
            break
        try:
            pipe.send(dict(status="ok", **benchmark.symbolic(*request)))
        except Exception as exc:
            pipe.send(dict(status="error", exact=False,
                           constant_difference_or_ratio=False,
                           error=f"{type(exc).__name__}: {exc}"))
    pipe.close()


class BoundedSymbolic:
    def __init__(self, cache_path, timeout=10.):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = benchmark.read(self.cache_path) if self.cache_path.exists() else {}
        self.timeout = timeout
        self.process = self.pipe = None
        self.calls = Counter()

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
            self.pipe.close()
            self.process.close()
            self.process = self.pipe = None

    def start(self):
        ctx = mp.get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=symbolic_worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        if not self.pipe.poll(30) or self.pipe.recv() != "ready":
            raise RuntimeError("Symbolic worker could not initialize within 30 seconds")

    def __call__(self, candidate, case, digits):
        # Only tree, fitted constants and declared truth determine this diagnostic.
        request = ({"tree": candidate["tree"], "constants": candidate["constants"]},
                   {"names": case["names"], "truth": case["truth"]}, digits)
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if key not in self.cache:
            try:
                if self.process is None:
                    self.start()
                self.pipe.send(request)
                if self.pipe.poll(self.timeout):
                    result = self.pipe.recv()
                else:
                    result = dict(status="timeout", exact=False,
                                  constant_difference_or_ratio=False)
                    self.close()
            except (EOFError, BrokenPipeError, OSError, RuntimeError) as exc:
                result = dict(status="error", exact=False,
                              constant_difference_or_ratio=False,
                              error=f"{type(exc).__name__}: {exc}")
                self.close()
            self.cache[key] = result
            benchmark.write(self.cache_path, self.cache)
        result = self.cache[key]
        self.calls[result["status"]] += 1
        return result


def score(out, timeout=10.):
    protocol = benchmark.read(out / "protocol.json")
    if benchmark.fingerprint() != protocol["source_hashes"]:
        raise RuntimeError("Frozen source changed")
    runs = [out / "runs" / f"{j['id']}.json" for j in protocol["jobs"]]
    if not all(p.exists() for p in runs):
        raise RuntimeError("Finish all fits before holdout scoring")
    amendment = dict(
        protocol_sha256=benchmark.sha(out / "protocol.json"),
        adapter_sha256=benchmark.sha(__file__), timeout_seconds=timeout,
        run_hashes={p.name: benchmark.sha(p) for p in runs},
        change="Subprocess timeout/error isolation for symbolic diagnostics only; "
               "frozen numeric scoring, fit, selection and gates unchanged. "
               "Indeterminate symbolic outcomes are retained and not counted as success.")
    path = out / "scoring_amendment.json"
    if path.exists() and benchmark.read(path) != amendment:
        raise RuntimeError("Scoring configuration or fitted records changed; use a new output directory")
    benchmark.write(path, amendment)
    bounded = BoundedSymbolic(out / "scored/symbolic_cache.json", timeout)
    original = benchmark.symbolic
    start = time.monotonic()
    try:
        benchmark.symbolic = bounded
        benchmark.score(out)
        benchmark.write(out / "symbolic_diagnostics.json", dict(
            calls=dict(bounded.calls), unique_comparisons=len(bounded.cache),
            unique_statuses=dict(Counter(r["status"] for r in bounded.cache.values())),
            wall_seconds=time.monotonic()-start,
            timeout_seconds=timeout, indeterminate_is_success=False))
    finally:
        benchmark.symbolic = original
        bounded.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=benchmark.ROOT / "results/acceptance_v1")
    args = parser.parse_args()
    score(args.out.resolve())
