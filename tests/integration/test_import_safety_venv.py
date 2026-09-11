from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _create_venv(tmp_path: Path) -> tuple[Path, Path]:
    env_dir = tmp_path / "venv"

    preferred_python = shutil.which("python3.12")
    current_python = Path(sys.executable)
    candidate_pythons: list[Path] = []
    if preferred_python:
        candidate_pythons.append(Path(preferred_python))
    if current_python not in candidate_pythons:
        candidate_pythons.append(current_python)

    creation_errors: list[str] = []
    for python_exe in candidate_pythons:
        result = subprocess.run(
            [str(python_exe), "-m", "venv", str(env_dir)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            break
        creation_errors.append(
            f"{python_exe} -> {result.returncode}\n{result.stdout}{result.stderr}"
        )
        shutil.rmtree(env_dir, ignore_errors=True)
    else:
        builder = venv.EnvBuilder(with_pip=True)
        try:
            builder.create(env_dir)
        except subprocess.CalledProcessError as exc:
            combined_errors = "\n\n".join(creation_errors)
            raise AssertionError(
                "failed to create isolated venv for import-safety test\n"
                f"{combined_errors}\n\n"
                f"fallback builder failed: {exc}"
            ) from exc

    if os.name == "nt":
        python_exe = env_dir / "Scripts" / "python.exe"
    else:
        python_exe = env_dir / "bin" / "python"
    return env_dir, python_exe


def _run(python: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [str(python), *args]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        cmd,
        cwd=python.parent.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_import_and_cli_help_without_torch(tmp_path: Path):
    # Create an isolated virtual environment and install the project.
    env_dir, python_exe = _create_venv(tmp_path)
    project_root = Path(__file__).resolve().parents[2]
    source_root = tmp_path / "src-copy"
    shutil.copytree(
        project_root,
        source_root,
        ignore=shutil.ignore_patterns(
            ".*",
            "*.egg-info",
            "__pycache__",
            "artifacts",
            "build",
            "custom-runs",
            "logs",
            "dist",
            "evidence_pack_runs",
            "reports",
            "runs",
            "site",
            "out",
            "tmp",
            "node_modules",
            "target",
        ),
    )

    distribution_root = tmp_path / "dist"
    build = _run(
        Path(sys.executable),
        [
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(distribution_root),
            str(source_root),
        ],
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheel = next(distribution_root.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any(
            "invarlock/pipeline/" in name or "pipeline_" in name for name in names
        )
        entry_points = archive.read(
            next(name for name in names if name.endswith("/entry_points.txt"))
        ).decode()
        assert "invarlock = invarlock.cli.app:app" in entry_points
        assert "invarlock-qualify-evaluator =" in entry_points
        assert "invarlock-pipeline" not in entry_points
        for schema in (
            "evidence_pack_v2",
            "evidence_verification_receipt_v3",
            "trust_inputs_v2",
            "evaluation_request_v2",
        ):
            assert f"invarlock/_data/contracts/{schema}.schema.json" in names
    with tarfile.open(next(distribution_root.glob("*.tar.gz"))) as archive:
        assert not any(
            "/src/invarlock/pipeline/" in name or "/pipeline_" in name
            for name in archive.getnames()
        )

    install = _run(python_exe, ["-m", "pip", "install", str(wheel)])
    if install.returncode != 0:
        combined = f"{install.stdout}{install.stderr}"
        if "requires a different Python" in combined or "not in '>=3.12'" in combined:
            pytest.skip("Requires Python 3.12+ to install invarlock in a venv.")
        if any(
            marker in combined
            for marker in (
                "Failed to establish a new connection",
                "NewConnectionError",
                "Temporary failure in name resolution",
                "Name or service not known",
                "nodename nor servname provided",
            )
        ):
            pytest.skip(
                "Network unavailable to install runtime dependencies into an isolated venv."
            )
        assert install.returncode == 0, combined

    # Ensure torch/transformers are not present in the venv.
    _run(python_exe, ["-m", "pip", "uninstall", "-y", "torch", "transformers"])

    # Plain import of the package root must succeed and expose __version__.
    res_import = _run(
        python_exe,
        ["-c", "import invarlock; print(invarlock.__version__)"],
    )
    assert res_import.returncode == 0, res_import.stderr
    assert res_import.stdout.strip()

    # CLI help via `python -m invarlock --help` must be torch-free.
    res_help = _run(python_exe, ["-m", "invarlock", "--help"])
    assert res_help.returncode == 0, res_help.stderr
    assert "Usage:" in res_help.stdout

    # Version flag must also work without torch installed.
    res_version = _run(python_exe, ["-m", "invarlock", "--version"])
    assert res_version.returncode == 0, res_version.stderr
    assert "InvarLock" in res_version.stdout

    consumer = tmp_path / "consumer"
    consumer.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    probe = subprocess.run(
        [
            str(python_exe),
            "-I",
            "-c",
            "from importlib.metadata import distribution, distributions; "
            "from pathlib import Path; import invarlock, sysconfig; "
            "assert Path(invarlock.__file__).is_relative_to(sysconfig.get_path('purelib')); "
            "assert not any(d.metadata['Name'].startswith('invarlock-') for d in distributions()); "
            "from invarlock.cli.app import app; "
            "assert {c.name for c in app.registered_commands} == {'evaluate', 'verify', 'report'}; "
            "assert {e.name for e in distribution('invarlock').entry_points if e.group == 'console_scripts'} == {'invarlock', 'invarlock-qualify-evaluator'}",
        ],
        cwd=consumer,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    shutil.copytree(
        project_root / "examples/acceptance-handoff/golden", consumer / "golden"
    )
    for relative, arguments in (
        ("examples/quickstart/run.py", ["--fixture", "golden"]),
        (
            "examples/captured-results/wheel_smoke.py",
            ["--cli", str(env_dir / "bin/invarlock")],
        ),
    ):
        script = shutil.copy2(project_root / relative, consumer / Path(relative).name)
        result = subprocess.run(
            [str(python_exe), str(script), *arguments],
            cwd=consumer,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
