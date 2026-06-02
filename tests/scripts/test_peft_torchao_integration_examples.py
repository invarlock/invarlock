from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PEFT_DIR = REPO_ROOT / "examples" / "integrations" / "peft_lora"
TORCHAO_DIR = REPO_ROOT / "examples" / "integrations" / "torchao_int8_export"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_peft_lora_runner_wires_local_fixture() -> None:
    runner = PEFT_DIR / "run_tiny_peft_lora.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)

    text = runner.read_text(encoding="utf-8")
    assert "--fixture-dir" in text
    assert "--preset" in text
    assert "fixture_summary.json" in text


def test_shared_compare_wrapper_checks_report_materialization() -> None:
    wrapper = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "run_invarlock_compare.sh"
    )
    subprocess.run(["bash", "-n", str(wrapper)], check=True)

    text = wrapper.read_text(encoding="utf-8")
    assert "Evaluate completed but did not write the expected report" in text
    assert '[[ ! -s "$report_json" ]]' in text


def test_torchao_runner_wires_local_fixture() -> None:
    runner = TORCHAO_DIR / "run_tiny_torchao_int8_export.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)

    text = runner.read_text(encoding="utf-8")
    assert "--fixture-dir" in text
    assert "--preset" in text
    assert "fixture_summary.json" in text


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

    data_path = Path(summary["data_path"])
    preset_path = Path(summary["preset_path"])
    assert data_path.exists()
    assert preset_path.exists()
    assert (tmp_path / "fixture_summary.json").exists()

    rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    preset = preset_path.read_text(encoding="utf-8")
    assert 'kind: "local_jsonl"' in preset
    assert 'id: "/tmp/tiny-gpt2-baseline"' in preset
    assert f'file: "{data_path}"' in preset
    assert "preview_n: 3" in preset
    assert summary["format_version"] == "peft-lora-fixture-v1"


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
        TORCHAO_DIR / "materialize_tiny_torchao_int8_subject.py",
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

    data_path = Path(summary["data_path"])
    preset_path = Path(summary["preset_path"])
    assert data_path.exists()
    assert preset_path.exists()
    assert (tmp_path / "fixture_summary.json").exists()

    rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    preset = preset_path.read_text(encoding="utf-8")
    assert 'kind: "local_jsonl"' in preset
    assert 'id: "/tmp/tiny-llama-baseline"' in preset
    assert f'file: "{data_path}"' in preset
    assert "preview_n: 3" in preset
    assert summary["format_version"] == "torchao-fixture-v1"
