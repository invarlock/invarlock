from __future__ import annotations

import re
from pathlib import Path


def test_runtime_dockerfile_installs_hf_stack() -> None:
    text = (Path.cwd() / "runtime" / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "ARG RUNTIME_BASE_IMAGE=python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4"
        in text
    )
    assert "FROM ${RUNTIME_BASE_IMAGE}" in text
    assert "ARG TARGETARCH" in text
    assert "COPY requirements/workflows/runtime-image-py312.txt" in text
    assert "COPY requirements/workflows/runtime-image-py312-aarch64.txt" in text
    assert "COPY requirements/workflows/runtime-image-py312-cu128.txt" in text
    assert "COPY requirements/workflows/runtime-image-quant-py312-cu128.txt" in text
    assert "ARG RUNTIME_REQUIREMENTS_AMD64" in text
    assert "ARG RUNTIME_REQUIREMENTS_ARM64" in text
    assert "ARG RUNTIME_CUDA_HOME" in text
    assert "ARG RUNTIME_KEEP_BUILD_TOOLCHAIN=0" in text
    assert "ARG RUNTIME_KEEP_BUILD_TOOLCHAIN" in text
    assert "ARG RUNTIME_PATH_PREFIX" in text
    assert "ARG PYTORCH_EXTRA_INDEX_URL" in text
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in text
    assert "CUDA_HOME=${RUNTIME_CUDA_HOME}" in text
    assert "PATH=${RUNTIME_PATH_PREFIX}${PATH}" in text
    assert '--extra-index-url "${PYTORCH_EXTRA_INDEX_URL}"' in text
    assert 'amd64) echo "/opt/invarlock/${RUNTIME_REQUIREMENTS_AMD64}"' in text
    assert 'arm64) echo "/opt/invarlock/${RUNTIME_REQUIREMENTS_ARM64}"' in text
    assert "apt-get install -y --no-install-recommends build-essential" in text
    assert "python3 python3-pip python3-venv python-is-python3" in text
    assert 'if [ "${RUNTIME_KEEP_BUILD_TOOLCHAIN}" = "1" ]' in text
    assert "apt-get install -y --no-install-recommends python3-dev" in text
    assert 'if [ "${RUNTIME_KEEP_BUILD_TOOLCHAIN}" != "1" ]' in text
    assert "apt-get purge -y --auto-remove build-essential" in text
    assert "python -m pip install" in text
    assert "--require-hashes" in text
    assert "python -m pip install --no-deps -e /opt/invarlock" not in text
    assert (
        "pip install --index-url https://download.pytorch.org/whl/cpu torch" not in text
    )
    assert "PYTHONPATH=/opt/invarlock/src" in text


def test_runtime_dockerfile_copied_requirement_files_exist() -> None:
    root = Path.cwd()
    text = (root / "runtime" / "Dockerfile").read_text(encoding="utf-8")
    copied_requirements = re.findall(r"COPY (requirements/workflows/[^ ]+) ", text)

    assert copied_requirements
    for relpath in copied_requirements:
        assert (root / relpath).is_file(), f"missing Dockerfile input: {relpath}"


def test_runtime_dockerignore_keeps_runtime_inputs() -> None:
    text = (Path.cwd() / ".dockerignore").read_text(encoding="utf-8")

    assert "**" in text
    assert "!README.md" in text
    assert "!pyproject.toml" in text
    assert "!contracts/**" in text
    assert "!runtime/Dockerfile" in text
    assert "!requirements/workflows/runtime-image-py312.txt" in text
    assert "!requirements/workflows/runtime-image-py312-aarch64.txt" in text
    assert "!requirements/workflows/runtime-image-py312-cu128.txt" in text
    assert "!requirements/workflows/runtime-image-quant-py312-cu128.txt" in text
    assert "!src/**" in text


def test_runtime_image_x86_requirements_are_hash_locked_cpu_only() -> None:
    text = (
        Path.cwd() / "requirements" / "workflows" / "runtime-image-py312.txt"
    ).read_text(encoding="utf-8")

    assert "torch==" in text
    assert "+cpu" in text
    assert "--hash=sha256:" in text
    assert "nvidia-cublas-cu12" not in text
    assert "nvidia-cuda-runtime-cu12" not in text
    assert "triton==" not in text


def test_runtime_image_aarch64_requirements_are_hash_locked() -> None:
    text = (
        Path.cwd() / "requirements" / "workflows" / "runtime-image-py312-aarch64.txt"
    ).read_text(encoding="utf-8")

    assert "torch==" in text
    assert "+cpu" in text
    assert "--hash=sha256:" in text
    assert "nvidia-cublas-cu12" not in text
    assert "nvidia-cuda-runtime-cu12" not in text
    assert "triton==" not in text


def test_runtime_image_cuda_requirements_are_hash_locked() -> None:
    text = (
        Path.cwd() / "requirements" / "workflows" / "runtime-image-py312-cu128.txt"
    ).read_text(encoding="utf-8")

    assert "torch==" in text
    assert "+cu128" in text
    assert "--hash=sha256:" in text
    assert "nvidia-cublas-cu12" in text
    assert "nvidia-cuda-runtime-cu12" in text
    assert "triton==" in text


def test_runtime_image_quant_cuda_requirements_are_hash_locked() -> None:
    text = (
        Path.cwd()
        / "requirements"
        / "workflows"
        / "runtime-image-quant-py312-cu128.txt"
    ).read_text(encoding="utf-8")

    assert "torch==" in text
    assert "+cu128" in text
    assert "--hash=sha256:" in text
    assert "bitsandbytes==" in text
    assert "gptqmodel==" in text
    assert "autoawq==" not in text


def test_current_quant_dependency_surfaces_do_not_pin_autoawq() -> None:
    root = Path.cwd()
    surfaces = (
        root / "pyproject.toml",
        root / "requirements" / "workflows" / "advanced-py313.txt",
        root / "requirements" / "workflows" / "runtime-image-quant.in",
        root / "requirements" / "workflows" / "runtime-image-quant-py312-cu128.txt",
    )

    for surface in surfaces:
        assert "autoawq" not in surface.read_text(encoding="utf-8").lower()
