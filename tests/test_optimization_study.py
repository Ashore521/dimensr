"""The scheduling amendment must never permit a changed algorithm or configuration."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_dispatch_resume_preserves_frozen_configuration(tmp_path):
    location = Path(__file__).parents[1] / "examples" / "compare_optimization.py"
    spec = importlib.util.spec_from_file_location("optimization_study", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    config = tmp_path / "config.json"
    definition = {"common": {}, "variants": {"small": {"max_nodes": 7}}}
    config.write_text(json.dumps(definition), encoding="utf-8")
    args = SimpleNamespace(config=config, baseline=baseline, output=tmp_path / "output",
                           resume_dispatch=False, workers=2)
    original = module.protocol(args)
    original["runner_sha256"] = "earlier_dispatch_only_runner"
    original["fingerprint"] = module.json_hash({k: v for k, v in original.items()
                                               if k not in ("fingerprint", "frozen_at")})
    module.write_json(args.output / "protocol.json", original)
    with pytest.raises(ValueError, match="changed"):
        module.protocol(args)
    args.resume_dispatch = True
    resumed = module.protocol(args)
    assert resumed == original
    assert (args.output / "dispatch_amendment.json").exists()
    definition["variants"]["small"]["max_nodes"] = 9
    config.write_text(json.dumps(definition), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        module.protocol(args)
