from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "integrations" / "compressed_tensors"
RUNNER = EXAMPLE_DIR / "run_tiny_hf_ct.sh"
HELPER = EXAMPLE_DIR / "prepare_tiny_hf_ct_fixture.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("compressed_tensors_example", HELPER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compressed_tensors_runner_has_expected_adapter_contract() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    text = RUNNER.read_text(encoding="utf-8")
    assert "--baseline-adapter hf_causal" in text
    assert "--subject-adapter hf_ct" in text
    assert "--edit-label compressed_tensors_checkpoint_load" in text
    assert "prepare_tiny_hf_ct_fixture.py" in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert "--lane cuda" in text
    assert "adapter_runtime_summary.json" in text
    assert "integration_default_host_device" in text
    assert "integration_preflight_host_cuda_device" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text


def test_compressed_tensors_helper_writes_local_jsonl_and_preset(
    tmp_path: Path,
) -> None:
    helper = _load_helper_module()
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-llama-hf-ct-baseline",
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
    assert preset["model"]["id"] == "/tmp/tiny-llama-hf-ct-baseline"
    assert preset["dataset"]["provider"]["kind"] == "local_jsonl"
    assert preset["dataset"]["provider"]["file"] == str(data_path)
    assert preset["dataset"]["seq_len"] == 32
    assert preset["dataset"]["preview_n"] == 3
    assert preset["dataset"]["final_n"] == 3

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["format_version"] == "compressed-tensors-fixture-v1"
    assert persisted["data_sha256"] == summary["data_sha256"]


def test_compressed_tensors_helper_writes_runtime_metadata(tmp_path: Path) -> None:
    helper = _load_helper_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    model_summary = {
        "format_version": "compressed-tensors-model-v1",
        "model_path": str(model_dir),
        "compressed_tensors": {
            "format": "pack-quantized",
            "packed_tensor_count": 2,
        },
    }
    runtime_summary = helper.write_adapter_runtime_summary(
        model_dir,
        subject_adapter="hf_ct",
        model_summary=model_summary,
    )

    assert runtime_summary["subject_adapter"] == "hf_ct"
    assert runtime_summary["checkpoint_quantization"] == {
        "format": "pack-quantized",
        "quant_method": "compressed-tensors",
        "weights": {
            "num_bits": 8,
            "strategy": "tensor",
            "symmetric": True,
        },
    }
    persisted = json.loads(
        (model_dir / "adapter_runtime_summary.json").read_text(encoding="utf-8")
    )
    assert persisted == runtime_summary


def test_compressed_tensors_helper_writes_checkpoint_refs(tmp_path: Path) -> None:
    helper = _load_helper_module()
    baseline = tmp_path / "baseline"
    subject = tmp_path / "subject"
    baseline.mkdir()
    subject.mkdir()
    (baseline / "config.json").write_text("{}", encoding="utf-8")
    (subject / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "compressed-tensors"}}),
        encoding="utf-8",
    )

    refs = helper.write_checkpoint_refs(
        subject, baseline_dir=baseline, subject_dir=subject
    )

    assert refs["baseline_adapter"] == "hf_causal"
    assert refs["subject_adapter"] == "hf_ct"
    assert refs["format_version"] == "compressed-tensors-checkpoint-refs-v1"
    assert "config.json" in refs["baseline_files"]
    assert "config.json" in refs["subject_files"]
