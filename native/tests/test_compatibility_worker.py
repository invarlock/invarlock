"""Explicit worker selection, protocol, deadlines and process-group cancellation."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ["CARGO_TARGET_DIR"]) / "debug/invarlock-native"


def run(*args, env=None):
    return subprocess.run(
        [str(BIN), *map(str, args)], capture_output=True, text=True, env=env, timeout=20
    )


def fake(tmp_path, body):
    path = tmp_path / "interpreter"
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o700)
    return path


def test_worker_is_explicit_and_preserves_arguments(tmp_path):
    python = fake(
        tmp_path,
        "import sys,json\nr=json.load(sys.stdin)\nassert r['format']=='invarlock/python-worker-v1'\nprint(json.dumps(r['argv']))\nsys.exit(7)\n",
    )
    result = run(
        "evaluate", "a space.yaml", "--python-worker", python, "--unsigned", "--json"
    )
    assert result.returncode == 7
    assert json.loads(result.stdout) == [
        "evaluate",
        "a space.yaml",
        "--unsigned",
        "--json",
    ]
    assert run("evaluate", "missing.yaml", "--python-worker", "python").returncode == 2
    assert run("evaluate", "missing.yaml", env={"PATH": str(tmp_path)}).returncode != 0


@pytest.mark.parametrize("option", ["--worker-timeout-seconds", "--python-worker"])
def test_repeated_worker_controls_rejected(tmp_path, option):
    python = fake(tmp_path, "raise AssertionError('must not execute')\n")
    value = "1" if option == "--worker-timeout-seconds" else str(python)
    result = run(
        "evaluate", "request", "--python-worker", python, option, value, option, value
    )
    assert result.returncode == 2
    assert "duplicate" in result.stderr


def test_deadline_terminates_worker(tmp_path):
    python = fake(tmp_path, "import time\ntime.sleep(30)\n")
    started = time.monotonic()
    result = run(
        "evaluate",
        "request",
        "--python-worker",
        python,
        "--worker-timeout-seconds",
        "1",
    )
    assert result.returncode == 2
    assert time.monotonic() - started < 10
    assert "tim" in result.stderr.lower()


def test_cancellation_terminates_descendant(tmp_path):
    child_pid = tmp_path / "child.pid"
    python = fake(
        tmp_path,
        f"import subprocess,sys,time,pathlib\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\npathlib.Path({str(child_pid)!r}).write_text(str(p.pid))\ntime.sleep(60)\n",
    )
    proc = subprocess.Popen(
        [str(BIN), "evaluate", "request", "--python-worker", str(python)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 8
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid.exists()
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=8)
        assert proc.returncode == 128 + signal.SIGTERM
        child = int(child_pid.read_text())
        # A reaped child disappears; an orphan zombie is terminated, awaiting init.
        if sys.platform == "linux":
            try:
                state = (
                    Path(f"/proc/{child}/stat")
                    .read_text()
                    .rpartition(")")[2]
                    .split()[0]
                )
            except FileNotFoundError:
                state = ""
        else:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child)], capture_output=True, text=True
            ).stdout.strip()
        assert not state or state.startswith("Z"), state
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


@pytest.mark.parametrize(
    "protocol",
    [
        {},
        {
            "format": "invarlock/python-worker-v1",
            "argv": ["evaluate", "--backend=python"],
        },
        {"format": "invarlock/python-worker-v1", "argv": ["shell"]},
    ],
)
def test_python_protocol_rejects_invalid_input(protocol):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "invarlock.native_compat_worker"],
        input=json.dumps(protocol),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "Python compatibility failed" in result.stderr


def test_actual_python_worker_captured_preflight(tmp_path):
    import importlib

    sys.path.insert(0, str(ROOT / "experiments/rust/python"))
    fixtures = importlib.import_module("invarlock_rust_experiment.full_fixtures")
    # The installed extension must be available; no fallback is permitted.
    import invarlock_native  # noqa: F401

    root = tmp_path.resolve() / "captured"
    fixtures.captured(root, 4, "pass")
    path = root / "request.json"
    result = run(
        "evaluate",
        path,
        "--unsigned",
        "--output",
        root / "worker-output",
        "--preflight",
        "--json",
        "--python-worker",
        sys.executable,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["ok"] is True


def test_installed_scorer_worker_publication_matches_python(tmp_path, monkeypatch):
    import yaml
    from typer.testing import CliRunner

    from invarlock.cli.app import app
    from tests.cli.test_import_journey import (
        _install_casefold_scorer_entry_point,
        _key,
        _materialize_request,
        _text_scorer_registry_and_binding,
    )

    root = tmp_path.resolve()
    registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        root, scorer_binding=binding, scorer_registry=registry
    )
    _install_casefold_scorer_entry_point(root, monkeypatch)
    key, _ = _key(root / "signer.pem")
    path = Path(material["request"])
    request = yaml.safe_load(path.read_text())
    request["output"]["evidence"] = "python-out"
    path.write_text(yaml.safe_dump(request))
    args = [
        "evaluate",
        str(path),
        "--signing-key",
        str(key),
        "--allow-installed-scorers",
        "--json",
    ]
    reference = CliRunner().invoke(app, args)
    assert reference.exit_code == 0, reference.output
    request["output"]["evidence"] = "rust-out"
    path.write_text(yaml.safe_dump(request))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(ROOT / "src"),
                str(ROOT),
                str(root / "installed-scorer"),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
    }
    result = run(*args, "--python-worker", sys.executable, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = json.loads(reference.stdout)
    expected["evidence"] = str(root / "rust-out")
    assert json.loads(result.stdout) == expected

    def files(p):
        return {
            str(x.relative_to(p)): x.read_bytes() for x in p.rglob("*") if x.is_file()
        }

    assert files(root / "python-out") == files(root / "rust-out")
    refused = run(*args, env={"PATH": "/nonexistent"})
    assert refused.returncode != 0
