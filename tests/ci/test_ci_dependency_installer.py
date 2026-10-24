"""Fast CI installation must retain the hash-locked environment contract."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.scripts.test_installed_pip_audit import _archive, _record, _wheel

ROOT = Path(__file__).resolve().parents[2]
JOBS = (
    "verify-checks",
    "verify-packages",
    "evaluator-parity",
    "minimum-python-packages",
    "minimum-python-parity",
    "minimum-python-tests",
    "coverage-tests",
)


def _jobs():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]


@pytest.mark.parametrize("name", JOBS)
def test_standard_setup_preserves_pins_bytecode_and_selected_python(name):
    steps = _jobs()[name]["steps"]
    setup = next(step for step in steps if step["name"] == "Set up uv")
    assert (
        setup["uses"] == "astral-sh/setup-uv@1c37ad07a6a961277cf70c0d37d6f313000f5884"
    )
    assert setup["with"] == {"version": "0.13.4", "enable-cache": False}
    installs = [step for step in steps if "uv pip install" in step.get("run", "")]
    assert installs
    for install in installs:
        assert steps.index(setup) < steps.index(install)
        for line in install["run"].splitlines():
            if "uv pip install" in line:
                assert line.startswith(
                    'uv pip install --python "$(command -v python)" '
                    "--compile-bytecode --require-hashes "
                )
                assert " -r requirements/workflows/" in line
    install = next(step for step in steps if step["name"] == "Install dependencies")
    assert install["run"].splitlines()[0] == (
        "python scripts/security/build_hardened_accelerate_wheel.py bootstrap"
    )
    assert "--find-links runtime/wheels" in install["run"]
    assert "python -m build --wheel --no-isolation" in install["run"]
    assert (
        "python -m pip install --no-deps --force-reinstall dist/*.whl" in install["run"]
    )


def _run(command, *, env, check=True):
    result = subprocess.run(
        command, env=env, check=False, capture_output=True, text=True, timeout=60
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def installation(tmp_path, monkeypatch):
    # The CI runner exposes the checkout through PYTHONPATH. Unrelated metadata
    # on that path must not enter the fresh installer comparison environments.
    ambient = tmp_path / "ambient"
    metadata = ambient / "unrelated-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unrelated\nVersion: 1.0\n"
        "Requires-Dist: deliberately-missing-dependency==1.0\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(ambient))
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith(("COVERAGE_", "COV_CORE_", "UV_", "PIP_")):
            env.pop(key)
    env.update(
        UV_CACHE_DIR=str(tmp_path / "cache"),
        UV_PYTHON_DOWNLOADS="never",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_CONFIG_FILE=os.devnull,
    )
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    paths = {}
    for name, version in (
        ("ci_install_dependency", "1.0"),
        ("ci_install_dependency", "2.0"),
        ("ci_install_parent", "1.0"),
    ):
        path = wheels / f"{name}-{version}-py3-none-any.whl"
        files = _wheel(path, name, version)
        if name == "ci_install_parent":
            dist = f"{name}-{version}.dist-info"
            files[f"{dist}/METADATA"] += b"Requires-Dist: ci-install-dependency==1.0\n"
            files[f"{dist}/RECORD"] = _record(files, dist)
            _archive(path, files)
        paths[(name, version)] = path
    lock = tmp_path / "locked.txt"
    lock.write_text(
        "".join(
            f"{name}==1.0 --hash=sha256:{hashlib.sha256(paths[name, '1.0'].read_bytes()).hexdigest()}\n"
            for name in ("ci_install_parent", "ci_install_dependency")
        )
    )
    interpreters = {}
    for name in ("pip", "uv"):
        _run([sys.executable, "-m", "venv", str(tmp_path / name)], env=env)
        interpreters[name] = str(tmp_path / name / "bin/python")
    return env, wheels, paths, lock, interpreters


def _candidate_command(python, wheels, lock):
    step = next(
        s
        for s in _jobs()["coverage-tests"]["steps"]
        if s["name"] == "Install dependencies"
    )
    line = next(line for line in step["run"].splitlines() if "uv pip install" in line)
    command = shlex.split(line)
    executable = shutil.which("uv")
    assert executable is not None, "the pinned uv installer is required"
    command[0] = executable
    command[command.index("--python") + 1] = python
    command[command.index("--find-links") + 1] = str(wheels)
    command[command.index("-r") + 1] = str(lock)
    return [*command, "--no-index"]


def test_real_installer_matches_pip_closure_and_downgrade(installation):
    env, wheels, paths, lock, interpreters = installation
    for python in interpreters.values():
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                str(paths["ci_install_dependency", "2.0"]),
            ],
            env=env,
        )
    _run(
        [
            interpreters["pip"],
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "--require-hashes",
            "-r",
            str(lock),
        ],
        env=env,
    )
    _run(_candidate_command(interpreters["uv"], wheels, lock), env=env)
    inventories = []
    for python in interpreters.values():
        _run([python, "-m", "pip", "check"], env=env)
        probe = _run(
            [
                python,
                "-I",
                "-c",
                "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ('ci_install_parent','ci_install_dependency')}))",
            ],
            env=env,
        )
        inventories.append(json.loads(probe.stdout))
    assert (
        inventories
        == [{"ci_install_parent": "1.0", "ci_install_dependency": "1.0"}] * 2
    )
    site = Path(interpreters["uv"]).parent.parent / "lib"
    assert list(site.glob("python*/site-packages/ci_install_parent/__pycache__/*.pyc"))


@pytest.mark.parametrize("fault", ["wrong_hash", "missing_hash", "missing_dependency"])
def test_real_installer_rejects_unverified_closure(installation, fault):
    env, wheels, paths, lock, interpreters = installation
    lines = lock.read_text().splitlines()
    if fault == "wrong_hash":
        lines[0] = lines[0].split("--hash=")[0] + "--hash=sha256:" + "0" * 64
    elif fault == "missing_hash":
        lines[0] = lines[0].split("--hash=")[0]
    else:
        lines = lines[:1]
    lock.write_text("\n".join(lines) + "\n")
    for command in (
        [
            interpreters["pip"],
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "--require-hashes",
            "-r",
            str(lock),
        ],
        _candidate_command(interpreters["uv"], wheels, lock),
    ):
        assert _run(command, env=env, check=False).returncode != 0
