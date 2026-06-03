from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME
from tests.integration.packaging._support_installed_wheel import (
    InstalledWheelEnv,
    _build_strict_report,
    _ensure_hf_smoke_dependencies,
    _output_indicates_network_unavailable,
    _prefetch_hf_smoke_model,
    _resolve_hf_smoke_env,
    _run,
    _write_json,
    _write_jsonl,
    _write_local_jsonl_preset,
    _write_runtime_manifest,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("SKIP_BUILD_TESTS") == "1", reason="skip build tests")
def test_wheel_install_verifies_strict_report_bundle_outside_repo_tree(
    installed_wheel_env: InstalledWheelEnv, tmp_path: Path
) -> None:
    report_dir = tmp_path / "strict-fixture"
    report_path = report_dir / "evaluation.report.json"
    _write_json(report_path, _build_strict_report())
    _write_runtime_manifest(report_path)

    verify = _run(
        installed_wheel_env.cli_exe,
        [
            "verify",
            "--assurance",
            "strict",
            "--profile",
            "ci",
            str(report_path),
        ],
        cwd=tmp_path,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "VERIFY OK" in verify.stdout

    html_path = tmp_path / "strict-fixture.html"
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
    )
    assert render_html.returncode == 0, render_html.stdout + render_html.stderr
    assert html_path.is_file()
    assert "<html" in html_path.read_text(encoding="utf-8").lower()


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

    smoke_env, local_cache_ready = _resolve_hf_smoke_env(
        installed_wheel_env.python_exe, tmp_path
    )

    if not local_cache_ready:
        prefetch = _prefetch_hf_smoke_model(
            installed_wheel_env.python_exe,
            cwd=tmp_path,
            env=smoke_env,
        )
        if prefetch.returncode != 0 and _output_indicates_network_unavailable(
            f"{prefetch.stdout}{prefetch.stderr}"
        ):
            pytest.skip(
                "Network unavailable and no local tiny-gpt2 cache for installed-wheel front-door smoke."
            )
        assert prefetch.returncode == 0, prefetch.stdout + prefetch.stderr

    report_dir = tmp_path / "front-door-report"
    evaluate = _run(
        installed_wheel_env.cli_exe,
        [
            "evaluate",
            "--assurance",
            "off",
            "--allow-network",
            "--execution-mode",
            "host",
            "--baseline",
            "sshleifer/tiny-gpt2",
            "--subject",
            "sshleifer/tiny-gpt2",
            "--baseline-adapter",
            "auto",
            "--subject-adapter",
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
