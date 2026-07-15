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
    assert (
        '"${WORKFLOW_DIR}/training-profile.in" \\\n'
        '    "${WORKFLOW_DIR}/training-profile-py312.txt"'
    ) in text
    assert (
        '"${EVIDENCE_PACK_DIR}/accelerate.in" \\\n'
        '    "${EVIDENCE_PACK_DIR}/accelerate.txt" \\\n'
        "    --no-deps"
    ) in text
    assert (
        '"${EVIDENCE_PACK_DIR}/cuda-nvcc.in" \\\n'
        '    "${EVIDENCE_PACK_DIR}/cuda-nvcc.txt" \\\n'
        "    --no-deps"
    ) in text
    assert (
        '"${EVIDENCE_PACK_DIR}/flash-attn.in" \\\n'
        '    "${EVIDENCE_PACK_DIR}/flash-attn.txt" \\\n'
        "    --no-deps"
    ) in text
    assert text.count("--torch-backend cpu") == 3
    assert text.count("--torch-backend cu128") == 2


def test_evidence_pack_helper_locks_do_not_select_torch_cuda_backend() -> None:
    req_dir = Path.cwd() / "requirements" / "evidence-packs"
    forbidden = (
        "torch==",
        "torchvision==",
        "torchao==",
        "triton==",
        "bitsandbytes==",
        "optimum-quanto==",
        "cuda-toolkit",
        "cuda-bindings",
        "nvidia-",
        "cu13",
    )
    allowed_backend_specific = {"cuda-nvcc.txt"}

    for path in sorted(req_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        assert str(Path.cwd()) not in text, f"{path.name} must use repo-relative paths"
        if path.name in allowed_backend_specific:
            continue
        for token in forbidden:
            assert token not in text, f"{path.name} must not pin {token!r}"


def test_runtime_image_locks_are_the_explicit_torch_backend_surface() -> None:
    workflow_dir = Path.cwd() / "requirements" / "workflows"
    runtime_locks = {
        "runtime-image-py312.txt": ("2.13.0+cpu", "0.28.0+cpu"),
        "runtime-image-py312-aarch64.txt": ("2.11.0+cpu", "0.26.0+cpu"),
        "runtime-image-py312-cu128.txt": ("2.11.0+cu128", "0.26.0+cu128"),
        "runtime-image-quant-py312-cu128.txt": (
            "2.11.0+cu128",
            "0.26.0+cu128",
        ),
        "training-profile-py312.txt": ("2.11.0+cpu", "0.26.0+cpu"),
    }

    for filename, (torch_version, torchvision_version) in runtime_locks.items():
        text = (workflow_dir / filename).read_text(encoding="utf-8")
        assert f"torch=={torch_version}" in text
        assert f"torchvision=={torchvision_version}" in text
        assert "cu13" not in text


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
