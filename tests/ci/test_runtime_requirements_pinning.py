from __future__ import annotations

import subprocess
from pathlib import Path


def test_refresh_pinned_requirements_generates_runtime_locks() -> None:
    script = Path.cwd() / "scripts" / "security" / "refresh_pinned_requirements.sh"
    text = script.read_text(encoding="utf-8")

    assert (
        '"${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-py312.txt"'
    ) in text
    assert (
        '"${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-py312-cu128.txt"'
    ) in text
    assert (
        '"${WORKFLOW_DIR}/runtime-image-quant.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-quant-py312-cu128.txt"'
    ) in text
    assert (
        '"${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-py312-aarch64.txt"'
    ) in text
    assert text.count("--torch-backend cpu") == 2
    assert text.count("--torch-backend cu128") == 2


def test_refresh_pinned_requirements_help_is_side_effect_free() -> None:
    script = Path.cwd() / "scripts" / "security" / "refresh_pinned_requirements.sh"

    result = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "uv pip compile" not in result.stderr
