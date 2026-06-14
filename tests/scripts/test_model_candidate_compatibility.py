from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path("scripts/checks/check_model_candidate_compatibility.py")


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_model_candidate_compatibility", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_model_candidate_compatibility"] = module
    spec.loader.exec_module(module)
    return module


def test_model_candidate_compatibility_accepts_current_contracts() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "invarlock/model-candidate-compatibility-audit-v1"
    assert payload["ok"] is True
    assert payload["finding_count"] == 0


def test_model_candidate_compatibility_catches_multimodal_auto_route_drift(
    monkeypatch,
) -> None:
    mod = _load_script_module()
    real_resolver = mod.resolve_auto_adapter

    def fake_resolver(model_id: str) -> str:
        if model_id == "Qwen/Qwen3.5-4B":
            return "hf_causal"
        return real_resolver(model_id)

    monkeypatch.setattr(mod, "resolve_auto_adapter", fake_resolver)

    findings = mod.audit()

    assert any(
        finding.scope == "support-matrix-backlog-gpu:qwen_qwen3_5_4b"
        and "adapter:auto resolves 'Qwen/Qwen3.5-4B' to 'hf_causal'" in finding.message
        for finding in findings
    )
