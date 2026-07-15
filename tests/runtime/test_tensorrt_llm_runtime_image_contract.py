from __future__ import annotations

from pathlib import Path

STABLE_IMAGE = (
    "nvcr.io/nvidia/tensorrt-llm/release:1.2.1@"
    "sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9"
)


def test_tensorrt_llm_image_pins_official_stable_multiarch_manifest() -> None:
    text = (Path.cwd() / "runtime" / "Dockerfile.tensorrt-llm").read_text(
        encoding="utf-8"
    )

    assert f"ARG TENSORRT_LLM_BASE_IMAGE={STABLE_IMAGE}" in text
    assert "FROM ${TENSORRT_LLM_BASE_IMAGE}" in text
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
    makefile = (Path.cwd() / "Makefile").read_text(encoding="utf-8")

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
    assert "/opt/invarlock/bin/vendor-python -c" in makefile
    assert (
        "/opt/invarlock/cli-venv/bin/invarlock advanced runtime-behavior --help"
        in makefile
    )


def test_makefile_qualifies_candidate_before_stable_tag() -> None:
    text = (Path.cwd() / "Makefile").read_text(encoding="utf-8")

    assert f"TENSORRT_LLM_BASE_IMAGE ?= {STABLE_IMAGE}" in text
    assert "runtime-image-tensorrt-llm:" in text
    assert "runtime-image-tensorrt-llm: runtime-image" not in text
    assert "-f runtime/Dockerfile.tensorrt-llm" in text
    assert "RUNTIME_IMAGE_TENSORRT_LLM_BUILD ?=" in text
    assert "runtime-smoke-tensorrt-llm:" in text
    assert (
        "$(MAKE) runtime-smoke-tensorrt-llm "
        "RUNTIME_IMAGE_TENSORRT_LLM=$(RUNTIME_IMAGE_TENSORRT_LLM_BUILD)"
    ) in text
    assert (
        "$(MAKE) runtime-canary-tensorrt-llm "
        "RUNTIME_IMAGE_TENSORRT_LLM=$(RUNTIME_IMAGE_TENSORRT_LLM_BUILD)"
    ) in text
    build = text.index("runtime-image-tensorrt-llm:")
    preflight_engine = text.index(
        "Set TENSORRT_LLM_CANARY_ENGINE_BUNDLE before building", build
    )
    preflight_tokenizer = text.index(
        "Set TENSORRT_LLM_CANARY_TOKENIZER_CONTRACT before building", build
    )
    preflight_engine_digest = text.index(
        "Set TENSORRT_LLM_CANARY_ENGINE_TREE_SHA256", build
    )
    preflight_tokenizer_digest = text.index(
        "Set TENSORRT_LLM_CANARY_TOKENIZER_SHA256", build
    )
    preflight_output_digest = text.index(
        "Set TENSORRT_LLM_CANARY_EXPECTED_OUTPUT_SHA256", build
    )
    image_build = text.index("-f runtime/Dockerfile.tensorrt-llm", build)
    smoke = text.index("$(MAKE) runtime-smoke-tensorrt-llm", build)
    canary = text.index("$(MAKE) runtime-canary-tensorrt-llm", smoke)
    stable_tag = text.index("image tag $(RUNTIME_IMAGE_TENSORRT_LLM_BUILD)", canary)
    assert build < preflight_engine < image_build
    assert build < preflight_tokenizer < image_build
    assert build < preflight_engine_digest < image_build
    assert build < preflight_tokenizer_digest < image_build
    assert build < preflight_output_digest < image_build
    assert image_build < smoke < canary < stable_tag
    assert "TENSORRT_LLM_DOCKER_GPUS ?= all" in text
    assert '--gpus "$(TENSORRT_LLM_DOCKER_GPUS)"' in text
    assert "--network none" in text
    assert "--read-only" in text
    assert "--tmpfs /tmp:rw,nosuid,nodev,noexec" in text
    assert "torch.cuda.is_available()" in text
    assert "NVIDIA_VISIBLE_DEVICES" in text
    assert "m.version(" in text
    assert "tensorrt_llm" in text
    assert "1.2.1" in text
    assert (
        "/opt/invarlock/bin/tensorrt-llm-runner --invarlock-runtime-info-v1"
    ) in text
    assert "runtime-canary-tensorrt-llm:" in text
    assert "image inspect --format '{{.Id}}'" in text
    assert '-e INVARLOCK_RUNTIME_IMAGE_DIGEST="$$image_digest"' in text
    assert '-e INVARLOCK_RUNTIME_IMAGE="$$image_digest"' in text
    assert "-m invarlock.runtime_providers.tensorrt_llm_canary" in text
    assert "--entrypoint /opt/invarlock/cli-venv/bin/python" in text
    assert "--engine-bundle /opt/invarlock/canary/engine" in text
    assert "--tokenizer-contract /opt/invarlock/canary/tokenizer.json" in text
    assert "--runner /opt/invarlock/bin/tensorrt-llm-runner" in text
    assert (
        '--expected-engine-tree-sha256 "$(TENSORRT_LLM_CANARY_ENGINE_TREE_SHA256)"'
        in text
    )
    assert (
        '--expected-tokenizer-sha256 "$(TENSORRT_LLM_CANARY_TOKENIZER_SHA256)"' in text
    )
    assert (
        "--expected-output-sha256 "
        '"$(TENSORRT_LLM_CANARY_EXPECTED_OUTPUT_SHA256)"' in text
    )
    assert "reviewed engine, tokenizer, and fixed-output digests" in text
    assert "stable qualification requires an authenticated real-engine canary" in text


def test_tensorrt_qualification_docs_require_reviewed_repeatable_fixture() -> None:
    text = (Path.cwd() / "docs" / "reference" / "runtime-providers.md").read_text(
        encoding="utf-8"
    )

    assert "TENSORRT_LLM_CANARY_ENGINE_TREE_SHA256" in text
    assert "TENSORRT_LLM_CANARY_TOKENIZER_SHA256" in text
    assert "TENSORRT_LLM_CANARY_EXPECTED_OUTPUT_SHA256" in text
    assert "two single-rank scores" in text
    assert "fresh provider sessions" in text
    assert "byte-identical canonical observations and receipts" in text
    assert "`FORCE_DETERMINISTIC=1`" in text
