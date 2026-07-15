from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

STABLE_IMAGE = (
    "nvcr.io/nvidia/tensorrt-llm/release:1.2.1@"
    "sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9"
)


def test_tensorrt_llm_image_pins_official_stable_multiarch_manifest() -> None:
    text = (Path.cwd() / "runtime" / "Dockerfile.tensorrt-llm").read_text(
        encoding="utf-8"
    )

    assert f"FROM {STABLE_IMAGE}" in text
    assert "ARG TENSORRT_LLM_BASE_IMAGE" not in text
    assert "FROM ${TENSORRT_LLM_BASE_IMAGE}" not in text
    assert "1.2.0rc" not in text
    assert "1.3.0rc" not in text
    assert "release:latest" not in text
    assert "376f7e1bd8ed543f75014309e3fd4b237e9b0e73" in text
    assert (
        "https://catalog.ngc.nvidia.com/orgs/nvidia/tensorrt-llm/containers/"
        "release/1.2.1"
    ) in text
    assert "https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1" in text
    assert 'dev.invarlock.runtime-provider="tensorrt_llm"' in text


def test_tensorrt_llm_layer_installs_only_the_exact_local_wheel() -> None:
    text = (Path.cwd() / "runtime" / "Dockerfile.tensorrt-llm").read_text(
        encoding="utf-8"
    )

    assert "runtime-wheel-build-py312.txt" in text
    assert "set -- /opt/invarlock/artifacts/invarlock-*.whl" in text
    assert "--no-compile" in text
    assert "--no-deps" in text
    assert "--no-index" in text
    assert "PIP_NO_INDEX=1" in text
    assert "INVARLOCK_ALLOW_NETWORK=0" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "runtime-image-py312-cu128.txt" not in text
    assert "runtime-image-quant-py312-cu128.txt" not in text
    assert "pip install invarlock[" not in text
    assert (
        "INVARLOCK_TENSORRT_LLM_RUNNER=/opt/invarlock/bin/tensorrt-llm-runner" in text
    )
    assert "install -m 0555" in text
    assert "/opt/invarlock/bin/tensorrt-llm-runner" in text

    project = (Path.cwd() / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        "invarlock-tensorrt-llm-runner = "
        '"invarlock.runtime_providers.tensorrt_llm_runner:main"'
    ) in project


def test_tensorrt_image_isolates_cli_from_vendor_backend_environment() -> None:
    dockerfile = (Path.cwd() / "runtime" / "Dockerfile.tensorrt-llm").read_text(
        encoding="utf-8"
    )
    boundary = (
        Path.cwd() / "scripts" / "release" / "tensorrt_llm_runtime_fixture_boundary.py"
    ).read_text(encoding="utf-8")

    assert "FROM ${WHEEL_BUILD_BASE} AS cli-dependencies" in dockerfile
    assert "-r /tmp/core-py312.txt" in dockerfile
    assert "-m venv /opt/invarlock/cli-venv" in dockerfile
    assert 'ln -s "${vendor_python}" /opt/invarlock/bin/vendor-python' in dockerfile
    assert (
        "/opt/invarlock/cli-venv/bin/invarlock advanced runtime-behavior --help"
        in dockerfile
    )
    assert (
        "/opt/invarlock/cli-venv/bin/invarlock advanced plugins "
        "runtime-providers --json" in dockerfile
    )
    assert "/opt/invarlock/bin/vendor-python -c" in boundary
    assert (
        "/opt/invarlock/cli-venv/bin/invarlock advanced runtime-behavior --help"
        in boundary
    )


def test_makefile_uses_the_safe_dual_flow_for_public_tensorrt_targets() -> None:
    text = (Path.cwd() / "Makefile").read_text(encoding="utf-8")

    assert "TENSORRT_LLM_BASE_IMAGE" not in text
    assert "runtime-image-tensorrt-llm: runtime-image-tensorrt-llm-dual" in text
    assert "runtime-canary-tensorrt-llm: runtime-canary-tensorrt-llm-dual" in text
    assert (
        "$(PYTHON) scripts/release/tensorrt_llm_runtime_fixture.py smoke-image" in text
    )
    assert "--build-arg TENSORRT_LLM_BASE_IMAGE" not in text


def test_makefile_dual_gpu_flow_builds_fixture_before_stable_promotion() -> None:
    text = (Path.cwd() / "Makefile").read_text(encoding="utf-8")

    exports = (
        "CONTAINER_ENGINE",
        "IMAGE",
        "STABLE_TAG",
        "GPU_0",
        "GPU_1",
        "SMOKE_GPU",
        "MODEL",
        "FIXTURE_ROOT",
        "MODEL_INVENTORY_SHA256",
        "SOURCE_DATE_EPOCH",
    )
    for name in exports:
        assert f"export INVARLOCK_TENSORRT_LLM_{name} :=" in text

    candidate = text.index("runtime-image-tensorrt-llm-candidate:")
    fixture = text.index("runtime-fixture-tensorrt-llm:", candidate)
    qualification = text.index("runtime-canary-tensorrt-llm-dual:", fixture)
    flow = text.index("runtime-image-tensorrt-llm-dual:", qualification)
    public_alias = text.index("runtime-image-tensorrt-llm:", flow)
    blocks = (
        text[candidate:fixture],
        text[fixture:qualification],
        text[qualification:flow],
        text[flow:public_alias],
    )
    assert "tensorrt_llm_runtime_fixture.py build-image" in blocks[0]
    assert "tensorrt_llm_runtime_fixture.py build-fixture" in blocks[1]
    assert "tensorrt_llm_runtime_fixture.py qualify-two-gpu" in blocks[2]
    assert "tensorrt_llm_runtime_fixture.py promote" in blocks[3]
    ordered = (
        "tensorrt_llm_runtime_fixture.py preflight",
        "runtime-image-tensorrt-llm-candidate",
        "tensorrt_llm_runtime_fixture.py smoke-image",
        "runtime-fixture-tensorrt-llm",
        "runtime-canary-tensorrt-llm-dual",
        "tensorrt_llm_runtime_fixture.py promote",
    )
    positions = tuple(blocks[3].index(value) for value in ordered)
    assert positions == tuple(sorted(positions))
    unsafe = (
        "CONTAINER_ENGINE",
        "RUNTIME_IMAGE_TENSORRT_LLM",
        "TENSORRT_LLM_PRIMARY_DOCKER_GPUS",
        "TENSORRT_LLM_SECONDARY_DOCKER_GPUS",
        "TENSORRT_LLM_FIXTURE_MODEL_DIR",
        "TENSORRT_LLM_FIXTURE_MODEL_INVENTORY_SHA256",
        "TENSORRT_LLM_FIXTURE_DIR",
        "RUNTIME_SOURCE_DATE_EPOCH",
    )
    for recipe in blocks:
        for name in unsafe:
            assert f"$({name})" not in recipe


def test_dual_flow_invalid_inputs_fail_before_docker_or_candidate_build(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "docker-started"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        f"#!/bin/sh\ntouch {marker}\nexit 99\n",
        encoding="utf-8",
    )
    docker.chmod(0o700)
    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    result = subprocess.run(
        [
            "make",
            "runtime-image-tensorrt-llm-dual",
            f"PYTHON={sys.executable}",
            f"TENSORRT_LLM_FIXTURE_MODEL_DIR={tmp_path / 'missing-model'}",
            f"TENSORRT_LLM_FIXTURE_DIR={tmp_path / 'fixture'}",
            f"TENSORRT_LLM_FIXTURE_MODEL_INVENTORY_SHA256={'1' * 64}",
            "RUNTIME_SOURCE_DATE_EPOCH=1784073600",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert result.returncode != 0
    assert "preflight" in result.stdout
    assert "runtime-image-tensorrt-llm-candidate" not in result.stdout
    assert not marker.exists()


def test_make_override_cannot_escape_before_python_validation(tmp_path: Path) -> None:
    marker = tmp_path / "escaped"
    payload = f'candidate:tag"; touch {marker}; #'
    result = subprocess.run(
        [
            "make",
            "runtime-image-tensorrt-llm-candidate",
            f"PYTHON={sys.executable}",
            f"RUNTIME_IMAGE_TENSORRT_LLM_BUILD={payload}",
            "RUNTIME_SOURCE_DATE_EPOCH=1784073600",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert not marker.exists()
    assert "candidate tag is invalid" in result.stderr


def test_tensorrt_qualification_docs_require_reviewed_repeatable_fixture() -> None:
    text = (Path.cwd() / "docs" / "reference" / "runtime-providers.md").read_text(
        encoding="utf-8"
    )

    assert "TENSORRT_LLM_FIXTURE_MODEL_INVENTORY_SHA256" in text
    assert "TENSORRT_LLM_PRIMARY_DOCKER_GPUS" in text
    assert "TENSORRT_LLM_SECONDARY_DOCKER_GPUS" in text
    assert "older single-GPU target" not in text
    assert "two single-rank scores" in text
    assert "fresh provider sessions" in text
    assert "byte-identical canonical observations and provider receipts" in text
    assert "`FORCE_DETERMINISTIC=1`" in text
