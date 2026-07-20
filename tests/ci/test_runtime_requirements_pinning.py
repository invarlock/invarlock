from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_REQUIREMENTS = ROOT / "requirements" / "workflows"
REFRESH_SCRIPT = ROOT / "scripts" / "security" / "refresh_pinned_requirements.sh"


def test_refresh_pinned_requirements_generates_canonical_runtime_locks() -> None:
    text = REFRESH_SCRIPT.read_text(encoding="utf-8")

    assert (
        '"${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-py312.txt"'
    ) in text
    assert (
        '"${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-py312-aarch64.txt"'
    ) in text
    assert (
        '"${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-py312-cu128.txt"'
    ) in text
    assert text.count("--torch-backend cpu") == 3
    assert text.count("--torch-backend cu128") == 2
    assert (
        '"${WORKFLOW_DIR}/multimodal-runtime.in" \\\n'
        '    "${WORKFLOW_DIR}/multimodal-runtime-py312.txt"'
    ) in text
    assert (
        '"${WORKFLOW_DIR}/lm-evaluation-harness.in" \\\n'
        '    "${WORKFLOW_DIR}/lm-evaluation-harness-py312.txt"'
    ) in text
    assert "--constraints requirements/workflows/runtime-image.in" in text
    assert "--no-deps" in text


def test_refresh_surface_excludes_retired_runtime_profiles() -> None:
    text = REFRESH_SCRIPT.read_text(encoding="utf-8")
    retired = (
        "advanced-py313",
        "assurance-ci",
        "evidence-packs",
        "runtime-image-quant",
        "training-profile",
    )

    for marker in retired:
        assert marker not in text


def test_runtime_image_locks_are_cpu_only() -> None:
    runtime_locks = (
        "runtime-image-py312.txt",
        "runtime-image-py312-aarch64.txt",
    )

    for filename in runtime_locks:
        text = (WORKFLOW_REQUIREMENTS / filename).read_text(encoding="utf-8")
        assert "torch==2.11.0+cpu" in text
        assert "+cu" not in text
        assert "cu13" not in text


def test_cuda_runtime_image_lock_is_separate_and_backend_pinned() -> None:
    text = (WORKFLOW_REQUIREMENTS / "runtime-image-py312-cu128.txt").read_text(
        encoding="utf-8"
    )

    assert "torch==2.11.0+cu128" in text
    assert "nvidia-cuda-runtime-cu12==" in text
    assert "--hash=sha256:" in text
    assert "bitsandbytes==" not in text
    assert "gptqmodel==" not in text


def test_runtime_wheel_build_lock_is_retained() -> None:
    lock = WORKFLOW_REQUIREMENTS / "runtime-wheel-build-py312.txt"

    assert lock.is_file()
    assert "--hash=sha256:" in lock.read_text(encoding="utf-8")


def test_multimodal_runtime_lock_pins_cuda_matched_torchvision() -> None:
    text = (WORKFLOW_REQUIREMENTS / "multimodal-runtime-py312.txt").read_text(
        encoding="utf-8"
    )

    assert "torchvision==0.26.0+cu128" in text
    assert "pillow==12.3.0" in text
    assert "torch==" not in text
    assert "--hash=sha256:" in text


def test_lm_evaluation_harness_lock_is_complete_and_cpu_aligned() -> None:
    text = (WORKFLOW_REQUIREMENTS / "lm-evaluation-harness-py312.txt").read_text(
        encoding="utf-8"
    )

    assert "lm-eval==0.4.12" in text
    assert "torch==2.11.0+cpu" in text
    assert "transformers==5.14.1" in text
    assert "+cu" not in text
    assert "--hash=sha256:" in text


def test_declared_typer_floor_matches_the_maintained_runtime_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    typer_requirement = next(
        dependency for dependency in dependencies if dependency.startswith("typer")
    )
    runtime_input = (WORKFLOW_REQUIREMENTS / "runtime-image.in").read_text(
        encoding="utf-8"
    )
    locked_typer = next(
        line.removeprefix("typer==")
        for line in runtime_input.splitlines()
        if line.startswith("typer==")
    )

    assert typer_requirement == f"typer>={locked_typer}"


def test_refresh_pinned_requirements_help_is_side_effect_free() -> None:
    result = subprocess.run(
        ["bash", str(REFRESH_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "all or workflows" in result.stdout
    assert "uv pip compile" not in result.stderr
