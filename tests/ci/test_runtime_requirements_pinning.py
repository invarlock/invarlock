from __future__ import annotations

import subprocess
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
    assert text.count("--torch-backend cpu") == 2
    assert text.count("--torch-backend cu128") == 1


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
