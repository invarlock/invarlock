from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME
from tests.integration.packaging._support_installed_wheel import (
    InstalledWheelEnv,
    _ensure_hf_smoke_dependencies,
    _run,
    _write_jsonl,
    _write_local_jsonl_preset,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
@pytest.mark.slow
def test_wheel_install_runs_front_door_evaluate_verify_report_html_outside_repo_tree(
    installed_wheel_env: InstalledWheelEnv, tmp_path: Path
) -> None:
    _ensure_hf_smoke_dependencies(installed_wheel_env)

    data_file = tmp_path / "smoke.jsonl"
    _write_jsonl(
        data_file,
        [
            "installed wheel front door sample one",
            "installed wheel front door sample two",
            "installed wheel front door sample three",
            "installed wheel front door sample four",
        ],
    )
    preset_path = tmp_path / "preset.yaml"
    _write_local_jsonl_preset(preset_path, data_file)

    hf_home = tmp_path / "hf-home"
    hf_home.mkdir(parents=True, exist_ok=True)
    smoke_env = {
        "HF_HOME": str(hf_home),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "INVARLOCK_ALLOW_NETWORK": "1",
        "INVARLOCK_DEDUP_TEXTS": "1",
        "INVARLOCK_TINY_RELAX": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_NO_TORCHVISION": "1",
    }

    prefetch = _run(
        installed_wheel_env.python_exe,
        [
            "-c",
            (
                "from transformers import AutoModelForCausalLM, AutoTokenizer; "
                "model_id='sshleifer/tiny-gpt2'; "
                "AutoTokenizer.from_pretrained(model_id, trust_remote_code=False); "
                "AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)"
            ),
        ],
        cwd=tmp_path,
        env=smoke_env,
        timeout=900,
    )
    assert prefetch.returncode == 0, prefetch.stdout + prefetch.stderr

    report_dir = tmp_path / "front-door-report"
    evaluate = _run(
        installed_wheel_env.cli_exe,
        [
            "evaluate",
            "--allow-network",
            "--execution-mode",
            "host",
            "--baseline",
            "sshleifer/tiny-gpt2",
            "--subject",
            "sshleifer/tiny-gpt2",
            "--adapter",
            "auto",
            "--profile",
            "dev",
            "--preset",
            str(preset_path),
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "runs"),
            "--report-out",
            str(report_dir),
            "--quiet",
            "--no-banner",
            "--no-progress",
            "--no-color",
        ],
        cwd=tmp_path,
        env=smoke_env,
        timeout=1800,
    )
    assert evaluate.returncode == 0, evaluate.stdout + evaluate.stderr

    report_path = report_dir / "evaluation.report.json"
    assert report_path.is_file()
    assert (report_dir / RUNTIME_MANIFEST_FILENAME).is_file()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["schema_version"] == "v1"

    verify = _run(
        installed_wheel_env.cli_exe,
        [
            "verify",
            "--runtime-provenance",
            "host",
            "--json",
            str(report_path),
        ],
        cwd=tmp_path,
        env=smoke_env,
        timeout=300,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    verify_payload = json.loads(verify.stdout.strip().splitlines()[-1])
    assert verify_payload["format_version"] == "verify-v1"
    assert verify_payload["summary"]["ok"] is True

    html_path = tmp_path / "front-door.html"
    render_html = _run(
        installed_wheel_env.cli_exe,
        [
            "report",
            "html",
            "-i",
            str(report_path),
            "-o",
            str(html_path),
            "--force",
        ],
        cwd=tmp_path,
        env=smoke_env,
        timeout=300,
    )
    assert render_html.returncode == 0, render_html.stdout + render_html.stderr
    assert html_path.is_file()
    assert "<html" in html_path.read_text(encoding="utf-8").lower()
