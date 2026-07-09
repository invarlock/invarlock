from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "integrations" / "quanto"
RUNNER = EXAMPLE_DIR / "run_tiny_hf_quanto.sh"
HELPER = EXAMPLE_DIR / "prepare_tiny_hf_quanto_fixture.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("quanto_example", HELPER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quanto_runner_has_expected_adapter_contract() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    text = RUNNER.read_text(encoding="utf-8")
    assert "--baseline-adapter hf_causal" in text
    assert "--subject-adapter hf_quanto" in text
    assert "--edit-label quanto_runtime_quantization" in text
    assert "prepare_tiny_hf_quanto_fixture.py" in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert "--lane cuda" in text
    assert "adapter_runtime_summary.json" in text
    assert "integration_default_host_device" in text
    assert "integration_preflight_host_cuda_device" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text


def test_quanto_helper_writes_local_jsonl_and_preset(tmp_path: Path) -> None:
    helper = _load_helper_module()
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-llama-quanto-baseline",
        rows=6,
        terms_per_row=5,
        seq_len=32,
        preview_n=3,
        final_n=3,
    )

    data_path = Path(summary["data_path"])
    preset_path = Path(summary["preset_path"])
    summary_path = tmp_path / "fixture_summary.json"

    assert data_path.exists()
    assert preset_path.exists()
    assert summary_path.exists()

    rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    assert all(isinstance(row["text"], str) and row["text"] for row in rows)

    preset = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    assert preset["model"]["id"] == "/tmp/tiny-llama-quanto-baseline"
    assert preset["dataset"]["provider"]["kind"] == "local_jsonl"
    assert preset["dataset"]["provider"]["file"] == str(data_path)
    assert preset["dataset"]["seq_len"] == 32
    assert preset["dataset"]["preview_n"] == 3
    assert preset["dataset"]["final_n"] == 3

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["format_version"] == "quanto-fixture-v1"
    assert persisted["data_sha256"] == summary["data_sha256"]


def test_quanto_helper_writes_runtime_metadata(tmp_path: Path) -> None:
    helper = _load_helper_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    model_summary = {"format_version": "quanto-model-v1", "model_path": str(model_dir)}
    runtime_summary = helper.write_adapter_runtime_summary(
        model_dir,
        subject_adapter="hf_quanto",
        weights="int8",
        model_summary=model_summary,
    )

    assert runtime_summary["subject_adapter"] == "hf_quanto"
    assert runtime_summary["runtime_quantization"] == {
        "quant_method": "quanto",
        "weights": "int8",
    }
    persisted = json.loads(
        (model_dir / "adapter_runtime_summary.json").read_text(encoding="utf-8")
    )
    assert persisted == runtime_summary
