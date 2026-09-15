"""Exercise shared installed-wheel consumer staging and failure handling."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.release import core_wheel_consumers as consumers


@pytest.mark.parametrize("failure_index", [None, 0, 1, 2, 3, 4, 5])
def test_stages_and_runs_all_consumers_with_clean_imports(monkeypatch, failure_index):
    monkeypatch.setenv("PYTHONPATH", str(consumers.ROOT / "src"))
    monkeypatch.setenv("PYTHONSAFEPATH", "0")
    monkeypatch.setattr(consumers.shutil, "which", lambda _: "/candidate/bin/invarlock")
    calls = []

    def run(command, *, cwd, env, check):
        calls.append((command, cwd))
        assert not cwd.is_relative_to(consumers.ROOT)
        assert "PYTHONPATH" not in env
        assert env["PYTHONSAFEPATH"] == env["PYTHONNOUSERSITE"] == "1"
        assert check is True
        assert command[0] == sys.executable
        if command[1] == "-c":
            assert "sysconfig.get_path('purelib')" in command[2]
        else:
            assert (cwd / command[1]).is_file()
        if command[1] == "scorer-wheel-smoke.py":
            assert command[2:] == [
                "--fixture",
                "judge",
                "--cli",
                "/candidate/bin/invarlock",
            ]
            for name in consumers.JUDGE_FILES:
                assert (cwd / "judge" / name).read_bytes() == (
                    consumers.ROOT / "examples/judge-measurements" / name
                ).read_bytes()
            assert (cwd / "judge/collection.json").is_file()
        if command[1] == "review/verify_deployment_receipt.py":
            assert (cwd / "incoming/evidence/manifest.json").is_file()
            assert (cwd / "incoming/verification.receipt.json").is_file()
            assert (cwd / "review/policy/acceptance.json").is_file()
        if failure_index == len(calls) - 1:
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(consumers.subprocess, "run", run)
    if failure_index is None:
        consumers.run_consumers("invarlock")
        assert [command[1] for command, _ in calls] == [
            "-c",
            "run.py",
            "captured-wheel-smoke.py",
            "judge/wheel_smoke.py",
            "scorer-wheel-smoke.py",
            "review/verify_deployment_receipt.py",
        ]
    else:
        with pytest.raises(subprocess.CalledProcessError) as error:
            consumers.run_consumers("invarlock")
        assert error.value.returncode == 7
        assert len(calls) == failure_index + 1
    assert all(not cwd.exists() for _, cwd in calls)


def test_missing_cli_fails_before_staging(monkeypatch):
    monkeypatch.setattr(consumers.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Install the candidate wheel"):
        consumers.run_consumers("missing")


def test_checkout_temporary_directory_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(consumers.shutil, "which", lambda _: "/candidate/bin/invarlock")
    monkeypatch.setattr(consumers, "ROOT", tmp_path)
    monkeypatch.setattr(consumers.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(RuntimeError, match="outside the checkout"):
        consumers.run_consumers("invarlock")
    assert list(tmp_path.iterdir()) == []


def test_missing_fixture_fails_closed_and_cleans_up(monkeypatch, tmp_path):
    monkeypatch.setattr(consumers.shutil, "which", lambda _: "/candidate/bin/invarlock")
    monkeypatch.setattr(consumers, "ROOT", tmp_path / "checkout")
    monkeypatch.setattr(consumers.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        consumers.run_consumers("invarlock")
    assert list(tmp_path.iterdir()) == []


def test_main_passes_cli(monkeypatch):
    observed = []
    monkeypatch.setattr(consumers.signal, "signal", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["consumers", "--cli", "/candidate/bin/invarlock"])
    monkeypatch.setattr(consumers, "run_consumers", observed.append)
    consumers.main()
    assert observed == ["/candidate/bin/invarlock"]


def test_script_entrypoint_requires_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["consumers"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(consumers.__file__)), run_name="__main__")
    assert error.value.code == 2


def test_editable_source_import_rejected_in_real_interpreter(tmp_path):
    import os
    import venv

    candidate = tmp_path / "candidate"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(candidate)
    source = tmp_path / "source"
    source.mkdir()
    (source / "invarlock.py").write_text("__version__ = 'source-fallback'\n")
    site = next(candidate.glob("lib/python*/site-packages"))
    (site / "editable.pth").write_text(str(source) + "\n")
    cli = candidate / "bin/invarlock"
    cli.write_text("#!/bin/sh\nexit 99\n")
    cli.chmod(0o755)
    staging = tmp_path / "staging"
    staging.mkdir()
    result = subprocess.run(
        [str(candidate / "bin/python"), consumers.__file__, "--cli", str(cli)],
        env={**os.environ, "TMPDIR": str(staging), "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert "PASS signed evidence verified" not in result.stdout
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize(
    "signum",
    [consumers.signal.SIGHUP, consumers.signal.SIGINT, consumers.signal.SIGTERM],
)
def test_signal_handler_exits_for_context_cleanup(signum):
    with pytest.raises(SystemExit) as error:
        consumers.exit_on_signal(signum, None)
    assert error.value.code == 128 + signum
