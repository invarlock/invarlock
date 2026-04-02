from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _load_remote_setup_smoke():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "proof_packs" / "python" / "remote_setup_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "proof_pack_remote_setup_smoke", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_setup_smoke_detects_missing_modules(monkeypatch) -> None:
    remote_setup_smoke = _load_remote_setup_smoke()

    def fake_import_module(name: str):
        if name == "sentencepiece":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(
        remote_setup_smoke.importlib, "import_module", fake_import_module
    )

    assert remote_setup_smoke.check_modules(("torch", "sentencepiece")) == [
        "sentencepiece"
    ]


def test_remote_setup_smoke_detects_cli_failures(monkeypatch) -> None:
    remote_setup_smoke = _load_remote_setup_smoke()

    monkeypatch.setattr(
        remote_setup_smoke.shutil, "which", lambda name: "/tmp/invarlock"
    )

    def fake_run(cmd, check, capture_output, text):
        return subprocess.CompletedProcess(
            cmd,
            2,
            stdout="",
            stderr="help failed",
        )

    monkeypatch.setattr(remote_setup_smoke.subprocess, "run", fake_run)

    err = remote_setup_smoke.check_cli("invarlock")
    assert err is not None
    assert "advanced proof-pack --help" in err


def test_remote_setup_smoke_main_succeeds_when_modules_and_cli_are_ready(
    monkeypatch,
) -> None:
    remote_setup_smoke = _load_remote_setup_smoke()

    monkeypatch.setattr(remote_setup_smoke, "check_modules", lambda modules: [])
    monkeypatch.setattr(remote_setup_smoke, "check_cli", lambda cli_name: None)

    assert remote_setup_smoke.main([]) == 0
