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
        if command[1] == "-I":
            assert command[2:] == [
                str(Path(consumers.__file__).resolve()),
                "--mode",
                "check-core",
                "--cli",
                "/candidate/bin/invarlock",
            ]
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
            "-I",
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


def test_missing_optional_cli_fails_before_staging(monkeypatch):
    monkeypatch.setattr(consumers.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Install the candidate wheel"):
        consumers.run_optional_consumers("missing")


def test_optional_consumers_run_from_clean_installed_environment(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(consumers.ROOT / "src"))
    monkeypatch.setenv("PYTHONSAFEPATH", "0")
    monkeypatch.setattr(consumers.shutil, "which", lambda _: "/candidate/bin/invarlock")
    calls = []

    def run(command, *, cwd, env, check):
        calls.append((command, cwd))
        assert command[0] == sys.executable
        assert not cwd.is_relative_to(consumers.ROOT)
        assert "PYTHONPATH" not in env
        assert env["PYTHONSAFEPATH"] == env["PYTHONNOUSERSITE"] == "1"
        assert check is True

    monkeypatch.setattr(consumers.subprocess, "run", run)
    consumers.run_optional_consumers("invarlock")

    assert len(calls) == 3
    assert "import PIL, numpy, invarlock" in calls[0][0][2]
    assert "spectral_observation" in calls[1][0][2]
    assert calls[2][0][1:] == (
        "-m",
        "invarlock.runtime_providers.hf_vision_text_conformance",
    )
    assert all(not cwd.exists() for _, cwd in calls)


@pytest.mark.parametrize("failure_kind", [None, "spectral", "rmt", "variance"])
def test_optional_probe_exercises_each_diagnostic(monkeypatch, failure_kind):
    import invarlock.diagnostics as diagnostics

    monkeypatch.setattr(consumers.shutil, "which", lambda _: "/candidate/bin/invarlock")
    commands = []
    monkeypatch.setattr(
        consumers.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    consumers.run_optional_consumers("invarlock")
    if failure_kind is not None:
        original = getattr(diagnostics, f"{failure_kind}_observation")

        def incorrect(values):
            return {**original(values), "status": "incorrect"}

        monkeypatch.setattr(diagnostics, f"{failure_kind}_observation", incorrect)
    if failure_kind is None:
        exec(commands[1][2], {})
    else:
        with pytest.raises(AssertionError):
            exec(commands[1][2], {})


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


def test_main_routes_optional_mode(monkeypatch):
    observed = []
    monkeypatch.setattr(consumers.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["consumers", "--mode", "optional", "--cli", "/candidate/bin/invarlock"],
    )
    monkeypatch.setattr(consumers, "run_optional_consumers", observed.append)
    consumers.main()
    assert observed == ["/candidate/bin/invarlock"]


def test_main_routes_core_boundary_mode(monkeypatch):
    observed = []
    monkeypatch.setattr(consumers.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["consumers", "--mode", "check-core", "--cli", "/candidate/bin/invarlock"],
    )
    monkeypatch.setattr(consumers, "check_core_install", observed.append)
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


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "version",
        "numpy",
        "PIL",
        "judge_origin",
        "provider_origin",
        "provider_name",
        "provider_abi",
        "provider_package",
        "provider_version",
        "entry_point",
        "missing_entry",
        "optional_import",
    ],
)
def test_core_boundary_rejects_corrupted_installed_surface(monkeypatch, failure):
    import importlib.metadata
    import importlib.util
    import sysconfig
    from types import SimpleNamespace

    import invarlock
    import invarlock.judge_measurements as judge
    from invarlock.core.registry import CoreRegistry

    site = Path(invarlock.__file__).resolve().parent.parent
    monkeypatch.setattr(sysconfig, "get_path", lambda _name: str(site))
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: (
            ("incorrect" if failure == "version" else invarlock.__version__)
            if name == "invarlock"
            else original_version(name)
        ),
    )
    names = ["hf_transformers", "hf_vision_text", "llama_cpp", "tensorrt_llm"]
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _name: SimpleNamespace(
            entry_points=[
                SimpleNamespace(name=name, group="invarlock.runtime_providers")
                for name in (names[1:] if failure == "missing_entry" else names)
            ]
        ),
    )
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            (object() if failure == name else None)
            if name in {"numpy", "PIL"}
            else original_find_spec(name)
        ),
    )
    for optional in (
        "torch",
        "transformers",
        "llama_cpp",
        "tensorrt_llm",
        "inspect_ai",
        "openai",
    ):
        monkeypatch.delitem(sys.modules, optional, raising=False)
    if failure == "optional_import":
        monkeypatch.setitem(sys.modules, "inspect_ai", SimpleNamespace())
    if failure == "judge_origin":
        monkeypatch.setattr(judge, "__file__", "/outside/judge.py")
    original_provider = CoreRegistry.get_runtime_provider

    def provider(registry, name):
        value = original_provider(registry, name)
        if failure == "provider_name":
            monkeypatch.setattr(value, "name", "incorrect")
        elif failure == "provider_abi":
            monkeypatch.setattr(value, "abi_version", "incorrect")
        elif failure == "provider_origin":
            monkeypatch.setattr(
                sys.modules[value.__class__.__module__],
                "__file__",
                "/outside/provider.py",
            )
        return value

    monkeypatch.setattr(CoreRegistry, "get_runtime_provider", provider)
    original_info = CoreRegistry.get_plugin_info

    def info(registry, name, kind):
        value = original_info(registry, name, kind)
        if failure in {"provider_package", "provider_version", "entry_point"}:
            field = {
                "provider_package": "package",
                "provider_version": "version",
                "entry_point": "entry_point",
            }[failure]
            value = {**value, field: "incorrect"}
        return value

    monkeypatch.setattr(CoreRegistry, "get_plugin_info", info)
    commands = []
    monkeypatch.setattr(
        consumers.subprocess, "run", lambda command, **_kwargs: commands.append(command)
    )
    if failure is not None:
        with pytest.raises(AssertionError):
            consumers.check_core_install("/candidate/bin/invarlock")
        assert commands == []
    else:
        consumers.check_core_install("/candidate/bin/invarlock")
        assert commands == [
            ["/candidate/bin/invarlock", "--help"],
            [
                sys.executable,
                "-I",
                "-m",
                "invarlock.runtime_providers.llama_cpp_conformance",
            ],
            [
                sys.executable,
                "-I",
                "-m",
                "invarlock.runtime_providers.tensorrt_llm_conformance",
            ],
        ]
