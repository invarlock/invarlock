from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli import runtime_verify
from invarlock.runtime_verify import RuntimeVerifyResult


def test_runtime_verify_cli_json_success(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(
        runtime_verify,
        "verify_runtime_manifest",
        lambda *_: RuntimeVerifyResult(
            ok=True,
            errors=(),
            report=str(tmp_path / "evaluation.report.json"),
            manifest=str(tmp_path / "runtime.manifest.json"),
        ),
    )
    exit_code = runtime_verify.main(
        [
            "--report",
            str(tmp_path / "evaluation.report.json"),
            "--manifest",
            str(tmp_path / "runtime.manifest.json"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload == {
        "format_version": "runtime-verify-v1",
        "ok": True,
        "errors": [],
        "report": str(tmp_path / "evaluation.report.json"),
        "manifest": str(tmp_path / "runtime.manifest.json"),
    }


def test_runtime_verify_cli_plain_failure(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(
        runtime_verify,
        "verify_runtime_manifest",
        lambda *_: RuntimeVerifyResult(
            ok=False,
            errors=("bad digest", "missing runtime"),
            report=str(tmp_path / "evaluation.report.json"),
            manifest=str(tmp_path / "runtime.manifest.json"),
        ),
    )

    exit_code = runtime_verify.main(
        [
            "--report",
            str(tmp_path / "evaluation.report.json"),
            "--manifest",
            str(tmp_path / "runtime.manifest.json"),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Runtime manifest verification failed" in output
    assert str(tmp_path / "evaluation.report.json") in output
    assert str(tmp_path / "runtime.manifest.json") in output
    assert "bad digest" in output
    assert "missing runtime" in output
