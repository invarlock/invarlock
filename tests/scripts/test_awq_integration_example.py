from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "integrations" / "awq"
README = EXAMPLE_DIR / "README.md"
RUNNER = EXAMPLE_DIR / "run_tiny_awq.sh"
HELPER = EXAMPLE_DIR / "materialize_tiny_awq_subject.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("awq_example", HELPER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_awq_runner_has_expected_adapter_contract() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    text = RUNNER.read_text(encoding="utf-8")
    assert "--baseline-adapter hf_causal" in text
    assert "--subject-adapter hf_awq" in text
    assert "--edit-label gptqmodel_awq_4bit" in text
    assert "materialize_tiny_awq_subject.py" in text
    assert 'execution_mode="host"' in text
    assert 'assurance="off"' in text
    assert 'device="cuda"' in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert 'awq_backend="torch_awq"' in text
    assert "--awq-backend" in text
    assert "torch.cuda.is_available()" in text
    assert "integration_default_host_device" in text
    assert "integration_preflight_host_cuda_device" in text
    assert "integration_preflight_gptqmodel_host_runtime" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text
    assert "--require-backend-inventory" in text
    assert "--require-runtime-quantization-proof" in text
    assert "AWQ lanes in this example are CUDA-only" in text
    assert '[[ "$effective_device" != cuda* ]]' in text
    assert '[[ "$quantize_device" != cuda* ]]' in text
    assert "require_gptqmodel_runtime" in text
    assert "_patch_gptqmodel_transformers_hub_compat" not in text


def test_awq_runner_rejects_cpu_lane_before_materialization(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            str(RUNNER),
            "--baseline-dir",
            str(tmp_path / "baseline"),
            "--subject-dir",
            str(tmp_path / "subject"),
            "--fixture-dir",
            str(tmp_path / "fixture"),
            "--report-out",
            str(tmp_path / "reports"),
            "--device",
            "cpu",
            "--materialize-only",
        ],
        check=False,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 2
    assert "AWQ lanes in this example are CUDA-only" in result.stderr
    assert not (tmp_path / "baseline").exists()
    assert not (tmp_path / "subject").exists()
    assert not (tmp_path / "fixture").exists()


def test_awq_readme_scopes_strict_evidence_to_tiny_runtime() -> None:
    text = README.read_text(encoding="utf-8")

    assert "`cuda-container-strict` result requires" in text
    assert "strict container evidence is verified" not in text
    assert "scoped to the configured tiny AWQ checkpoint" in text
    assert "shared integration evidence" in text
    assert "`cuda-host-off` | `--lane host --device cuda`" in text
    assert "The shell runner relies on InvarLock report persistence to emit" in text
    assert "`backend_inventory.json` when adapter provenance is available" in text
    assert "`runtime_quantization_proof.json`" in text


def test_awq_helper_defaults_are_awq_compatible() -> None:
    helper = _load_helper_module()
    parser = helper.build_parser()
    args = parser.parse_args(
        [
            "--baseline-dir",
            "/tmp/baseline",
            "--subject-dir",
            "/tmp/subject",
            "--fixture-dir",
            "/tmp/fixture",
        ]
    )

    assert args.awq_backend == "torch_awq"
    assert args.hidden_size == 256
    assert args.intermediate_size == 256
    assert args.max_position_embeddings == 256


def test_awq_helper_writes_local_jsonl_and_preset(tmp_path: Path) -> None:
    helper = _load_helper_module()
    helper_text = HELPER.read_text(encoding="utf-8")
    assert "import_gptqmodel" in helper_text
    assert "require_jit_toolchain=True" in helper_text
    assert "_patch_gptqmodel_transformers_hub_compat" not in helper_text

    summary = helper.write_text_fixture(
        tmp_path,
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
    assert preset["dataset"]["provider"]["kind"] == "local_jsonl"
    assert preset["dataset"]["provider"]["file"] == str(data_path)
    assert preset["dataset"]["seq_len"] == 32
    assert preset["dataset"]["preview_n"] == 3
    assert preset["dataset"]["final_n"] == 3

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["format_version"] == "awq-fixture-v1"
    assert persisted["data_sha256"] == summary["data_sha256"]


def test_awq_helper_pins_transformers_backend(tmp_path: Path) -> None:
    helper = _load_helper_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization_config": {
                    "bits": 4,
                    "group_size": 16,
                    "quant_method": "awq",
                    "version": "gemm",
                },
            }
        ),
        encoding="utf-8",
    )

    quant_cfg = helper._pin_awq_backend(config_path, backend="torch_awq")

    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert quant_cfg["backend"] == "torch_awq"
    assert persisted["quantization_config"]["backend"] == "torch_awq"
