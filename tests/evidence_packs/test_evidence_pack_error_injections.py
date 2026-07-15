from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch.nn as nn

from scripts.evidence_packs.python.error_model.common import _save_error_model


def test_error_injection_set_includes_weight_tying_break() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    scenarios_path = repo_root / "scripts/evidence_packs/scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenario_ids = {entry.get("id") for entry in scenarios.get("scenarios", [])}

    assert "weight_tying_break" in scenario_ids
    assert "zero_layer" not in scenario_ids


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
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            sliding_window=None,
        )

    def save_pretrained(self, output_path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        (output_path / "config.json").write_text(
            json.dumps(vars(self.config), sort_keys=True), encoding="utf-8"
        )


class _FakeTokenizer:
    def save_pretrained(self, output_path: Path) -> None:
        (output_path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def test_missing_tensors_injects_nested_language_model_layers(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = (
        repo_root / "scripts/evidence_packs/python/error_model/basic_injections.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scripts.evidence_packs.python.error_model.basic_injections", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script_path.parents[1]))
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


def test_missing_tensors_saved_output_already_has_consistent_layer_config(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline"
    baseline_path.mkdir()
    (baseline_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 2, "sliding_window": 131072}),
        encoding="utf-8",
    )
    model = _FakeConditionalGeneration()
    error_info: dict[str, object] = {}

    from scripts.evidence_packs.python.error_model.basic_injections import (
        _inject_missing_tensors,
    )

    _inject_missing_tensors(
        model=model, baseline_path=baseline_path, error_info=error_info
    )
    output_path = tmp_path / "error_missing_tensors"
    _save_error_model(
        model=model,
        tokenizer=_FakeTokenizer(),
        output_path=output_path,
        error_info=error_info,
        use_gpu=False,
    )

    saved = json.loads((output_path / "config.json").read_text())
    assert saved["num_hidden_layers"] == 1
    assert saved["layer_types"] == ["full_attention"]
    assert saved["sliding_window"] == 131072
