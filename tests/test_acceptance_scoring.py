"""Symbolic failures must preserve numerical scoring and never become success."""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import score_acceptance as scoring


def slow_worker(pipe):
    pipe.send("ready")
    pipe.recv()
    time.sleep(5)


def test_symbolic_timeout_is_terminated_and_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "symbolic_worker", slow_worker)
    cache = tmp_path / "symbolic.json"
    bounded = scoring.BoundedSymbolic(cache, timeout=.05)
    request = ({"tree": {}, "constants": []}, {"names": ["x"], "truth": "x"}, 8)
    try:
        result = bounded(*request)
        assert result == dict(status="timeout", exact=False,
                              constant_difference_or_ratio=False)
        assert bounded.process is None
        # Resume preserves the failed diagnostic instead of trying until it succeeds.
        resumed = scoring.BoundedSymbolic(cache)
        assert resumed(*request) == result
        assert resumed.process is None
    finally:
        bounded.close()


def test_symbolic_error_does_not_poison_next_candidate(tmp_path):
    from dimensr.expression import Node
    bounded = scoring.BoundedSymbolic(tmp_path / "symbolic.json")
    case = {"names": ["x"], "truth": "x"}
    try:
        invalid = bounded({"tree": {}, "constants": []}, case, 8)
        assert invalid["status"] == "error" and not invalid["exact"]
        valid = bounded({"tree": Node("x0").to_dict(), "constants": []}, case, 8)
        assert valid["status"] == "ok" and valid["exact"]
    finally:
        bounded.close()
