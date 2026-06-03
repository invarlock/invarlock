from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import invarlock.runtime_security as runtime_security

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.manual,
]


def _require_manual_container_smoke() -> None:
    if os.environ.get("INVARLOCK_CONTAINER_DEFAULT_SMOKE") != "1":
        pytest.skip(
            "set INVARLOCK_CONTAINER_DEFAULT_SMOKE=1 to run the container-default smoke"
        )
    if os.environ.get("INVARLOCK_ALLOW_NETWORK") != "1":
        pytest.skip("set INVARLOCK_ALLOW_NETWORK=1 for the model-download smoke")
    engine = runtime_security.resolve_container_engine()
    if engine is None:
        pytest.skip("docker/podman is required for the default runtime-container smoke")
    image = runtime_security.resolve_runtime_image()
    if not runtime_security.container_image_available_locally(image, engine=engine):
        pytest.skip(f"runtime image {image!r} is not available locally")


def _write_smoke_dataset(path: Path) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"text": text})
            for text in [
                "container default smoke sample one",
                "container default smoke sample two",
                "container default smoke sample three",
                "container default smoke sample four",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_smoke_preset(path: Path, data_file: Path) -> None:
    path.write_text(
        textwrap.dedent(
            f"""
            dataset:
              provider:
                kind: local_jsonl
              file: {data_file}
              split: validation
              seq_len: 16
              stride: 16
              preview_n: 2
              final_n: 2
              seed: 42
            guards:
              order: []
            eval:
              metric: {{kind: ppl_causal}}
              loss: {{type: auto}}
            """
        ),
        encoding="utf-8",
    )


def _write_smoke_edit(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            edit:
              name: quant_rtn
              plan:
                bitwidth: 8
                per_channel: true
                clamp_ratio: 0.005
                scope: attn
                max_modules: 12
                seed: 42
            """
        ),
        encoding="utf-8",
    )


def _write_smoke_profile(config_root: Path) -> None:
    (config_root / "runtime" / "profiles").mkdir(parents=True)
    (config_root / "runtime" / "profiles" / "smoke_ext.yaml").write_text(
        "model:\n  device: cpu\n",
        encoding="utf-8",
    )


def _build_smoke_env(
    repo_root: Path, config_root: Path, tmpdir: Path
) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["INVARLOCK_ALLOW_NETWORK"] = "1"
    env.pop("INVARLOCK_ALLOW_HOST_EXECUTION", None)
    env["INVARLOCK_CONFIG_ROOT"] = str(config_root)
    env["TMPDIR"] = str(tmpdir)
    return env


def test_evaluate_container_default_smoke_with_external_runtime_inputs(
    tmp_path: Path,
) -> None:
    _require_manual_container_smoke()
    repo_root = Path(__file__).resolve().parents[2]

    data_file = tmp_path / "smoke.jsonl"
    _write_smoke_dataset(data_file)
    preset_path = tmp_path / "preset.yaml"
    _write_smoke_preset(preset_path, data_file)
    edit_path = tmp_path / "edit.yaml"
    _write_smoke_edit(edit_path)
    config_root = tmp_path / "config-root"
    _write_smoke_profile(config_root)
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    out_dir = tmp_path / "runs"
    report_dir = tmp_path / "report"

    env = _build_smoke_env(repo_root, config_root, tmpdir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "invarlock",
            "evaluate",
            "--baseline",
            "sshleifer/tiny-gpt2",
            "--subject",
            "sshleifer/tiny-gpt2",
            "--baseline-adapter",
            "hf_causal",
            "--subject-adapter",
            "hf_causal",
            "--preset",
            str(preset_path),
            "--edit-config",
            str(edit_path),
            "--profile",
            "smoke_ext",
            "--assurance",
            "off",
            "--allow-network",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
            "--report-out",
            str(report_dir),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (report_dir / "evaluation.report.json").is_file()
    assert (report_dir / runtime_security.RUNTIME_MANIFEST_FILENAME).is_file()


def test_container_default_front_door_smoke_runs_evaluate_verify_and_report_html(
    tmp_path: Path,
) -> None:
    _require_manual_container_smoke()
    repo_root = Path(__file__).resolve().parents[2]

    data_file = tmp_path / "front-door.jsonl"
    _write_smoke_dataset(data_file)
    preset_path = tmp_path / "preset.yaml"
    _write_smoke_preset(preset_path, data_file)
    config_root = tmp_path / "config-root"
    _write_smoke_profile(config_root)
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    out_dir = tmp_path / "runs"
    report_dir = tmp_path / "report"
    report_path = report_dir / "evaluation.report.json"
    html_path = tmp_path / "report.html"

    env = _build_smoke_env(repo_root, config_root, tmpdir)

    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "invarlock",
            "evaluate",
            "--baseline",
            "sshleifer/tiny-gpt2",
            "--subject",
            "sshleifer/tiny-gpt2",
            "--baseline-adapter",
            "auto",
            "--subject-adapter",
            "auto",
            "--preset",
            str(preset_path),
            "--profile",
            "smoke_ext",
            "--assurance",
            "off",
            "--allow-network",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
            "--report-out",
            str(report_dir),
            "--quiet",
            "--no-banner",
            "--no-progress",
            "--no-color",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )

    assert evaluate.returncode == 0, evaluate.stdout + evaluate.stderr
    assert report_path.is_file()
    manifest_path = report_dir / runtime_security.RUNTIME_MANIFEST_FILENAME
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("execution_mode") == "container"

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "invarlock",
            "verify",
            "--assurance",
            "off",
            "--json",
            str(report_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert verify.returncode == 0, verify.stdout + verify.stderr
    verify_payload = json.loads(verify.stdout.strip().splitlines()[-1])
    assert verify_payload["format_version"] == "verify-v1"
    assert verify_payload["summary"]["ok"] is True

    render_html = subprocess.run(
        [
            sys.executable,
            "-m",
            "invarlock",
            "report",
            "html",
            "-i",
            str(report_path),
            "-o",
            str(html_path),
            "--force",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert render_html.returncode == 0, render_html.stdout + render_html.stderr
    assert html_path.is_file()
    assert "<html" in html_path.read_text(encoding="utf-8").lower()
