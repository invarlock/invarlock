from __future__ import annotations

from pathlib import Path


def test_refresh_pinned_requirements_generates_runtime_locks() -> None:
    text = (
        Path.cwd() / "scripts" / "security" / "refresh_pinned_requirements.sh"
    ).read_text(encoding="utf-8")

    assert (
        'compile_req_platform \\\n  "${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '  "${WORKFLOW_DIR}/runtime-image-py312.txt"'
    ) in text
    assert (
        'compile_req_platform \\\n  "${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '  "${WORKFLOW_DIR}/runtime-image-py312-cu128.txt"'
    ) in text
    assert (
        'compile_req_platform \\\n  "${WORKFLOW_DIR}/runtime-image-quant.in" \\\n'
        '  "${WORKFLOW_DIR}/runtime-image-quant-py312-cu128.txt"'
    ) in text
    assert (
        'compile_req_platform \\\n  "${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '  "${WORKFLOW_DIR}/runtime-image-py312-aarch64.txt"'
    ) in text
    assert text.count("--torch-backend cpu") == 2
    assert text.count("--torch-backend cu128") == 2
