"""Verify acceptance reporting using synthetic fixtures, never study results."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dimensr.expression import Node


@pytest.fixture
def reporting(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("acceptance_reporting", examples / "report_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit():
    return {**{s: dict(invalid=0, nrmse=0., r2=1.) for s in ("test", "ood", "probe")},
            "numerical_recovery": True, "accurate_iid": True}


@pytest.fixture
def saved_run(tmp_path, reporting, monkeypatch):
    # A one-node identity model provides hand-checkable validation and holdout metrics.
    job = dict(id="fixture", case="identity", data_key="fixture", seed=1, noise=0., variant="physo")
    (tmp_path / "data").mkdir()
    X = np.array([[0.], [1.], [2.]])
    for group, splits in (("fit", ("train", "val")), ("holdout", reporting.SPLITS)):
        np.savez(tmp_path / "data" / f"fixture_{group}.npz",
                 **{f"{kind}_{split}": X if kind == "X" else X[:, 0]
                    for split in splits for kind in ("X", "y")})
    protocol = dict(source_hashes={}, jobs=[job],
                    dataset_hashes={f"data/fixture_{g}.npz": reporting.sha(tmp_path / "data" / f"fixture_{g}.npz")
                                    for g in ("fit", "holdout")},
                    cases={"identity": dict(names=["x"], units=[[0]*7], target=[0]*7)})
    reporting.write(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(reporting, "fingerprint", lambda: {})
    row = dict(job, status="ok", candidates=[dict(tree=Node("x0").to_dict(), constants=[])],
               audit=[audit()], selected=0, selected_audit=audit(),
               pareto_any_recovery=True, pareto_any_accurate_iid=True)
    return tmp_path, [row], reporting.sha(tmp_path / "protocol.json")


def test_saved_numeric_metrics_and_flags_are_reverified(reporting, saved_run):
    result = reporting.verify(*saved_run)
    assert result["verified_candidate_trees"] == 1
    assert result["all_numeric_success_flags_recomputed"]
    assert result["selected_audits_match"]
    assert result["summary_protocol_hash_matches"]


@pytest.mark.parametrize("field", ["numerical_recovery", "accurate_iid", "pareto_any_recovery",
                                    "pareto_any_accurate_iid", "selected_audit"])
def test_tampered_success_flags_are_rejected(reporting, saved_run, field):
    out, rows, digest = saved_run
    row = rows[0]
    if field in ("numerical_recovery", "accurate_iid"):
        row["audit"][0][field] = False
    elif field == "selected_audit":
        row[field]["test"]["r2"] = .5
    else:
        row[field] = False
    with pytest.raises(AssertionError):
        reporting.verify(out, rows, digest)


def test_summary_must_match_the_frozen_protocol(reporting, saved_run):
    out, rows, _ = saved_run
    with pytest.raises(AssertionError):
        reporting.verify(out, rows, "incorrect-protocol-digest")


def test_numeric_domain_gate_requires_all_planned_runs(reporting):
    good = dict(status="ok", selected_audit=audit())
    failed = dict(status="timeout")
    assert reporting.numeric_domain_gate([good])
    assert not reporting.numeric_domain_gate([])
    assert not reporting.numeric_domain_gate([failed])
    assert not reporting.numeric_domain_gate([good, failed])


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf])
def test_nonfinite_metrics_never_pass_numeric_gates(reporting, value):
    bad = audit()
    bad["test"].update(nrmse=value, r2=value)
    assert reporting.numeric_flags(bad) == dict(numerical_recovery=False, accurate_iid=False)
    assert not reporting.numeric_domain_gate([dict(status="ok", selected_audit=bad)])
    native = dict(status="ok", native_selected_audit=bad)
    assert reporting.count([native], "native") == 0
    assert reporting.count([native], "native_accurate") == 0


def test_nonfinite_metrics_are_excluded_with_visible_denominator(reporting):
    good = dict(status="ok", selected_audit=audit())
    bad = copy.deepcopy(good)
    bad["selected_audit"]["test"]["nrmse"] = np.inf
    assert reporting.median([good, bad, dict(status="failed")]) == "0 (1/3)"
