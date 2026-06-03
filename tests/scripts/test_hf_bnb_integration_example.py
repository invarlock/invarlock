from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "integrations" / "hf_bnb"
README = EXAMPLE_DIR / "README.md"
RUNNER = EXAMPLE_DIR / "run_tiny_hf_bnb_8bit.sh"
HELPER = EXAMPLE_DIR / "prepare_tiny_hf_bnb_fixture.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("hf_bnb_fixture", HELPER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hf_bnb_runner_has_expected_adapter_contract() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    text = RUNNER.read_text(encoding="utf-8")
    assert "--baseline-adapter hf_causal" in text
    assert "--subject-adapter hf_bnb" in text
    assert "--edit-label hf_bnb_8bit_runtime_load" in text
    assert "prepare_tiny_hf_bnb_fixture.py" in text
    assert "--model-dir" in text
    assert 'execution_mode="host"' in text
    assert 'assurance="off"' in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert "integration_default_host_device" in text
    assert "integration_preflight_host_cuda_device" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text


def test_hf_bnb_readme_scopes_strict_evidence_to_tiny_runtime() -> None:
    text = README.read_text(encoding="utf-8")

    assert "strict container evidence is verified" in text
    assert "this tiny\nbitsandbytes runtime-load example" in text
    assert "scoped to the configured tiny runtime-loaded bitsandbytes" in text
    assert "shared integration evidence" in text
    assert "The shell runner relies on InvarLock report persistence to emit" in text
    assert "`backend_inventory.json` when adapter provenance is available" in text


def test_prepare_tiny_hf_bnb_fixture_writes_local_jsonl_and_preset(
    tmp_path: Path,
) -> None:
    helper = _load_helper_module()
    summary = helper.write_fixture(
        tmp_path,
        model_id="/tmp/local-tiny-llama",
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
    assert 'id: "/tmp/local-tiny-llama"' in preset
    assert f'file: "{data_path}"' in preset
    assert "seq_len: 32" in preset
    assert "preview_n: 3" in preset
    assert "final_n: 3" in preset

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["format_version"] == "hf-bnb-fixture-v1"
    assert persisted["data_sha256"] == summary["data_sha256"]
