from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATIONS_DIR = REPO_ROOT / "examples" / "integrations"
PEFT_DIR = INTEGRATIONS_DIR / "peft_lora"
FINE_TUNE_DIR = INTEGRATIONS_DIR / "fine_tune"
MAGNITUDE_PRUNE_DIR = INTEGRATIONS_DIR / "magnitude_prune"
TORCHAO_DIR = INTEGRATIONS_DIR / "torchao_int8_runtime"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_local_fixture(
    summary: dict[str, object],
    tmp_path: Path,
    *,
    expected_model_id: str,
    expected_format_version: str,
) -> None:
    data_path = Path(str(summary["data_path"]))
    preset_path = Path(str(summary["preset_path"]))
    assert data_path.exists()
    assert preset_path.exists()
    assert (tmp_path / "fixture_summary.json").exists()

    rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    preset = preset_path.read_text(encoding="utf-8")
    assert 'kind: "local_jsonl"' in preset
    assert f'id: "{expected_model_id}"' in preset
    assert f'file: "{data_path}"' in preset
    assert "preview_n: 3" in preset
    assert summary["format_version"] == expected_format_version


def test_peft_lora_helper_writes_local_jsonl_and_preset(tmp_path: Path) -> None:
    helper = _load_module(
        PEFT_DIR / "materialize_tiny_peft_lora_subject.py",
        "peft_lora_example",
    )
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-gpt2-baseline",
        rows=6,
        terms_per_row=5,
        seq_len=32,
        preview_n=3,
        final_n=3,
    )

    _assert_local_fixture(
        summary,
        tmp_path,
        expected_model_id="/tmp/tiny-gpt2-baseline",
        expected_format_version="peft-lora-fixture-v1",
    )


def test_fine_tune_helper_writes_local_jsonl_and_preset(tmp_path: Path) -> None:
    helper = _load_module(
        FINE_TUNE_DIR / "materialize_tiny_fine_tune_subject.py",
        "fine_tune_example",
    )
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-gpt2-baseline",
        rows=6,
        terms_per_row=5,
        seq_len=32,
        preview_n=3,
        final_n=3,
    )

    _assert_local_fixture(
        summary,
        tmp_path,
        expected_model_id="/tmp/tiny-gpt2-baseline",
        expected_format_version="tiny-fine-tune-fixture-v1",
    )


def test_magnitude_prune_helper_writes_local_jsonl_and_preset(
    tmp_path: Path,
) -> None:
    helper = _load_module(
        MAGNITUDE_PRUNE_DIR / "materialize_tiny_magnitude_prune_subject.py",
        "magnitude_prune_example",
    )
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-gpt2-baseline",
        rows=6,
        terms_per_row=5,
        seq_len=32,
        preview_n=3,
        final_n=3,
    )

    _assert_local_fixture(
        summary,
        tmp_path,
        expected_model_id="/tmp/tiny-gpt2-baseline",
        expected_format_version="tiny-magnitude-prune-fixture-v1",
    )


def test_peft_lora_helper_isolates_dense_lora_from_quantized_dispatch(
    monkeypatch,
) -> None:
    helper = _load_module(
        PEFT_DIR / "materialize_tiny_peft_lora_subject.py",
        "peft_lora_example_dispatch",
    )

    class DenseModel:
        config = object()
        is_quantized = False

    calls = {"count": 0}

    def fake_get_peft_model(_model, _config):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ImportError(
                "cannot import name 'AwqGEMMQuantLinear' from "
                "'gptqmodel.nn_modules.qlinear.gemm_awq'"
            )
        return "peft-model"

    monkeypatch.setattr(
        helper,
        "_disable_quantized_peft_dispatch_for_dense_example",
        lambda: True,
    )

    assert (
        helper._get_dense_peft_model(DenseModel(), object(), fake_get_peft_model)
        == "peft-model"
    )
    assert calls["count"] == 2


def test_torchao_helper_writes_local_jsonl_and_preset(tmp_path: Path) -> None:
    helper = _load_module(
        TORCHAO_DIR / "prepare_tiny_hf_torchao_fixture.py",
        "torchao_example",
    )
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-llama-baseline",
        rows=6,
        terms_per_row=5,
        seq_len=32,
        preview_n=3,
        final_n=3,
    )

    _assert_local_fixture(
        summary,
        tmp_path,
        expected_model_id="/tmp/tiny-llama-baseline",
        expected_format_version="torchao-fixture-v1",
    )


def test_torchao_helper_prefers_non_deprecated_config_version() -> None:
    helper = _load_module(
        TORCHAO_DIR / "prepare_tiny_hf_torchao_fixture.py",
        "torchao_config_helper",
    )

    class _ModernConfig:
        def __init__(self, *, version=None):
            self.version = version

    class _LegacyConfig:
        def __init__(self, **kwargs):
            if kwargs:
                raise TypeError("unexpected keyword")
            self.version = 1

    modern = helper._torchao_int8_weight_only_config(_ModernConfig)
    legacy = helper._torchao_int8_weight_only_config(_LegacyConfig)

    assert modern.version == 2
    assert legacy.version == 1
