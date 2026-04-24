from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch.nn as nn


def test_error_injection_set_includes_weight_tying_break() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    scenarios_path = repo_root / "scripts/evidence_packs/scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenario_ids = {entry.get("id") for entry in scenarios.get("scenarios", [])}

    assert "weight_tying_break" in scenario_ids
    assert "zero_layer" not in scenario_ids

    # Ensure the harness is wired to the manifest (avoid drift between task graph and verdict).
    queue_manager = (
        repo_root / "scripts/evidence_packs/lib/queue_manager.sh"
    ).read_text(encoding="utf-8")
    assert "scenarios.json" in queue_manager


class _FakeTextLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeTextLayer(), _FakeTextLayer()])


class _FakeContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeLanguageModel()


class _FakeConditionalGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeContainer()
        self.config = SimpleNamespace(num_hidden_layers=2)


def test_missing_tensors_injects_nested_language_model_layers(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = (
        repo_root
        / "scripts/evidence_packs/python/create_error_model_basic_injections.py"
    )
    spec = importlib.util.spec_from_file_location(
        "create_error_model_basic_injections", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    baseline_path = tmp_path / "baseline"
    baseline_path.mkdir()
    (baseline_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 2}), encoding="utf-8"
    )

    model = _FakeConditionalGeneration()
    error_info: dict[str, object] = {}

    module._inject_missing_tensors(
        model=model, baseline_path=baseline_path, error_info=error_info
    )

    assert error_info["injected"] is True
    assert error_info["arch"] == "language_model_layers"
    assert error_info["layers_before"] == 2
    assert error_info["layers_after"] == 1
    assert len(model.model.language_model.layers) == 1
    assert model.config.num_hidden_layers == 1
