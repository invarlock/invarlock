from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.integrations import local_registry

ROOT = Path(__file__).resolve().parents[2]


def test_flat_script_entrypoints_resolve_their_sibling_helpers(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(ROOT / "src"),
            str(ROOT / "examples/integrations"),
        )
    )
    registry = subprocess.run(
        [
            sys.executable,
            "-c",
            "import local_registry; print(local_registry.REGISTRY_IMAGE)",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert registry.returncode == 0, registry.stderr
    assert registry.stdout.strip() == local_registry.REGISTRY_IMAGE

    runner = subprocess.run(
        [sys.executable, str(ROOT / "examples/integrations/run.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert runner.returncode == 0, runner.stderr
    assert "--runtime-image-digest" in runner.stdout


@pytest.mark.parametrize("entrypoint", ("run.py", "showcase.py"))
def test_tensorrt_entrypoints_start_as_direct_scripts(entrypoint: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(ROOT / "src"),
            str(ROOT / "addins/tensorrt_llm/src"),
            str(ROOT),
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/integrations/tensorrt-llm" / entrypoint),
            "--help",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
