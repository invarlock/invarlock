from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "integrations" / "awq"
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
    assert 'awq_backend="torch_awq"' in text
    assert "--awq-backend" in text
    assert "torch.cuda.is_available()" in text


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

    preset = preset_path.read_text(encoding="utf-8")
    assert 'kind: "local_jsonl"' in preset
    assert f'file: "{data_path}"' in preset
    assert "seq_len: 32" in preset
    assert "preview_n: 3" in preset
    assert "final_n: 3" in preset

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
