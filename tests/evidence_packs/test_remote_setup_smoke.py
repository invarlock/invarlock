from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _load_remote_setup_smoke():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "evidence_packs" / "python" / "runtime_tools.py"
    spec = importlib.util.spec_from_file_location(
        "evidence_pack_runtime_tools_remote_smoke", script
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
    assert "advanced evidence-pack --help" in err


def test_remote_setup_smoke_main_succeeds_when_modules_and_cli_are_ready(
    monkeypatch,
) -> None:
    remote_setup_smoke = _load_remote_setup_smoke()

    monkeypatch.setattr(remote_setup_smoke, "check_modules", lambda modules: [])
    monkeypatch.setattr(remote_setup_smoke, "check_cli", lambda cli_name: None)
    monkeypatch.setattr(remote_setup_smoke, "check_repo_root", lambda repo_root: None)
    monkeypatch.setattr(remote_setup_smoke, "check_runtime_provenance", lambda: None)

    assert remote_setup_smoke.main(["remote-setup-smoke"]) == 0


def test_remote_setup_smoke_detects_runtime_provenance_gap(monkeypatch) -> None:
    remote_setup_smoke = _load_remote_setup_smoke()

    monkeypatch.setattr(remote_setup_smoke, "check_modules", lambda modules: [])
    monkeypatch.setattr(remote_setup_smoke, "check_cli", lambda cli_name: None)
    monkeypatch.setattr(remote_setup_smoke, "check_repo_root", lambda repo_root: None)
    monkeypatch.setattr(
        remote_setup_smoke,
        "check_runtime_provenance",
        lambda: (
            "runtime image 'ghcr.io/invarlock/invarlock-runtime:latest' is not provenance-ready"
        ),
    )

    assert remote_setup_smoke.main(["remote-setup-smoke"]) == 1


def test_remote_setup_smoke_only_runtime_provenance_mode_skips_module_and_cli_checks(
    monkeypatch,
) -> None:
    remote_setup_smoke = _load_remote_setup_smoke()

    monkeypatch.setattr(
        remote_setup_smoke,
        "check_modules",
        lambda modules: (_ for _ in ()).throw(
            AssertionError("module check should be skipped")
        ),
    )
    monkeypatch.setattr(
        remote_setup_smoke,
        "check_cli",
        lambda cli_name: (_ for _ in ()).throw(
            AssertionError("cli check should be skipped")
        ),
    )
    monkeypatch.setattr(remote_setup_smoke, "check_runtime_provenance", lambda: None)

    assert (
        remote_setup_smoke.main(["remote-setup-smoke", "--only-runtime-provenance"])
        == 0
    )


def test_remote_setup_smoke_repo_root_requires_entrypoints(tmp_path: Path) -> None:
    remote_setup_smoke = _load_remote_setup_smoke()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    err = remote_setup_smoke.check_repo_root(str(repo_root))
    assert err is not None
    assert "missing required entrypoint" in err


def test_remote_setup_smoke_main_checks_repo_root(monkeypatch, tmp_path: Path) -> None:
    remote_setup_smoke = _load_remote_setup_smoke()

    monkeypatch.setattr(remote_setup_smoke, "check_modules", lambda modules: [])
    monkeypatch.setattr(remote_setup_smoke, "check_cli", lambda cli_name: None)
    monkeypatch.setattr(remote_setup_smoke, "check_runtime_provenance", lambda: None)

    checked: list[str] = []

    def fake_check_repo_root(repo_root: str) -> str | None:
        checked.append(repo_root)
        return None

    monkeypatch.setattr(remote_setup_smoke, "check_repo_root", fake_check_repo_root)

    repo_root = tmp_path / "repo"
    assert (
        remote_setup_smoke.main(["remote-setup-smoke", "--repo-root", str(repo_root)])
        == 0
    )
    assert checked == [str(repo_root)]
