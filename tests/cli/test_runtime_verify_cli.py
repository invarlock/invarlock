from __future__ import annotations

import builtins
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from click.termui import strip_ansi
from click.testing import CliRunner

from invarlock.cli.commands import verify as runtime_verify
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


def test_runtime_verify_cli_plain_success(monkeypatch, tmp_path: Path, capsys) -> None:
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
        ]
    )

    assert exit_code == 0
    assert "Runtime manifest verification passed" in capsys.readouterr().out


def test_runtime_verify_cli_help_surface() -> None:
    result = CliRunner().invoke(runtime_verify.runtime_verify_app, ["--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "COMMAND [ARGS]..." not in out
    assert "--report" in out
    assert "--manifest" in out
    assert "--version" in out


def test_runtime_verify_cli_version(capsys) -> None:
    exit_code = runtime_verify.main(
        [
            "--report",
            "evaluation.report.json",
            "--manifest",
            "runtime.manifest.json",
            "--version",
        ]
    )

    assert exit_code == 0
    assert "InvarLock runtime verifier" in capsys.readouterr().out


def test_runtime_verify_version_uses_package_metadata(monkeypatch, capsys) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "9.9.9")

    runtime_verify._emit_version(SimpleNamespace(print=print))

    assert "InvarLock runtime verifier 9.9.9" in capsys.readouterr().out


def test_runtime_verify_version_falls_back_to_module_version(monkeypatch, capsys):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError("missing")
        ),
    )

    runtime_verify._emit_version(SimpleNamespace(print=print))

    assert "InvarLock runtime verifier" in capsys.readouterr().out


def test_runtime_verify_version_unknown_when_imports_fail(monkeypatch, capsys):
    original_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):  # noqa: ANN001
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if name == "importlib.metadata":
            raise ImportError("metadata unavailable")
        if name == "invarlock" and fromlist:
            raise ImportError("version unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    runtime_verify._emit_version(SimpleNamespace(print=print))

    assert "version unknown" in capsys.readouterr().out


def test_runtime_verify_main_returns_click_exit_code() -> None:
    exit_code = runtime_verify.main(["--help"])
    assert exit_code == 0


def test_runtime_verify_main_catches_click_exit(monkeypatch) -> None:
    class _Command:
        def main(self, **_kwargs):  # noqa: ANN001
            raise runtime_verify.click.exceptions.Exit(7)

    monkeypatch.setattr(runtime_verify, "build_click_command", lambda: _Command())

    assert runtime_verify.main(["--help"]) == 7


def test_runtime_verify_build_click_command() -> None:
    assert runtime_verify.build_click_command() is runtime_verify.runtime_verify_app
