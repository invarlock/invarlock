from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path.cwd()


def test_every_maintained_runtime_image_rejects_unbound_source_identity() -> None:
    for relative in (
        "runtime/Dockerfile",
        "runtime/Dockerfile.cuda",
        "addins/gguf/runtime/Dockerfile",
        "addins/multimodal/runtime/Dockerfile",
        "addins/tensorrt_llm/runtime/Dockerfile",
    ):
        text = ROOT.joinpath(relative).read_text(encoding="utf-8")
        assert "INVARLOCK_SOURCE_BUNDLE_SHA256=unbound" not in text
        assert "INVARLOCK_SOURCE_COMMIT=unbound" not in text
        assert "invalid INVARLOCK_SOURCE_COMMIT" in text
        assert "invalid INVARLOCK_SOURCE_BUNDLE_SHA256" in text


def test_runtime_dockerfile_builds_and_installs_one_final_wheel() -> None:
    text = ROOT.joinpath("runtime", "Dockerfile").read_text(encoding="utf-8")

    assert "FROM ${RUNTIME_BUILD_BASE_IMAGE} AS public-wheel" in text
    assert "--wheel-dir /wheelhouse" in text
    assert "--no-build-isolation" in text
    assert "FROM ${RUNTIME_BASE_IMAGE}" in text
    assert '/opt/invarlock/artifacts/$(basename "$1")' in text
    assert "--no-deps" in text
    assert 'ENTRYPOINT ["python", "-m", "invarlock"]' in text
    assert "pip install --no-deps -e" not in text
    assert "PYTHONPATH=/" not in text
    assert 'org.opencontainers.image.revision="${INVARLOCK_SOURCE_COMMIT}"' in text
    assert (
        'dev.invarlock.source-bundle-sha256="${INVARLOCK_SOURCE_BUNDLE_SHA256}"' in text
    )


def test_runtime_dockerfile_has_one_hf_runtime_dependency_surface() -> None:
    text = ROOT.joinpath("runtime", "Dockerfile").read_text(encoding="utf-8")

    copied_requirements = re.findall(r"COPY (requirements/workflows/[^ ]+) ", text)
    assert copied_requirements == [
        "requirements/workflows/runtime-image-py312.txt",
        "requirements/workflows/runtime-image-py312-aarch64.txt",
        "requirements/workflows/runtime-wheel-build-py312.txt",
    ]
    assert "RUNTIME_REQUIREMENTS_AMD64" in text
    assert "RUNTIME_REQUIREMENTS_ARM64" in text
    assert "runtime-image-py312-cu" not in text
    assert "runtime-image-quant" not in text
    assert "gptq" not in text.lower()
    assert "CUDA_HOME" not in text
    assert "RUNTIME_KEEP_BUILD_TOOLCHAIN" not in text
    assert "RUNTIME_PATH_PREFIX" not in text
    for relative in copied_requirements:
        assert ROOT.joinpath(relative).is_file()


def test_runtime_dockerfile_is_offline_by_default_and_hash_locked() -> None:
    text = ROOT.joinpath("runtime", "Dockerfile").read_text(encoding="utf-8")

    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "--require-hashes" in text
    assert '--extra-index-url "${PYTORCH_EXTRA_INDEX_URL}"' in text
    assert "ARG SOURCE_DATE_EPOCH" in text
    assert 'touch -h -d "@${SOURCE_DATE_EPOCH}"' in text


def test_runtime_input_is_the_core_plus_canonical_hf_dependencies() -> None:
    text = ROOT.joinpath("requirements", "workflows", "runtime-image.in").read_text(
        encoding="utf-8"
    )

    required = {
        "typer",
        "click",
        "cryptography",
        "rich",
        "pyyaml",
        "jsonschema",
        "accelerate",
        "torch",
        "transformers",
        "safetensors",
        "protobuf",
        "sentencepiece",
        "tiktoken",
    }
    observed = {
        line.split("=", 1)[0].split(">", 1)[0].split("!", 1)[0]
        for line in text.splitlines()
        if line and not line.startswith("#")
    }
    assert observed == required
    for retired in (
        "autoawq",
        "bitsandbytes",
        "compressed-tensors",
        "datasets",
        "gptqmodel",
        "hqq",
        "optimum-quanto",
        "peft",
        "torchao",
        "torchvision",
    ):
        assert retired not in text.lower()


def test_hf_extra_declares_fp8_runtime_support() -> None:
    project = tomllib.loads(ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    requirements = project["project"]["optional-dependencies"]["hf"]

    assert any(str(item).startswith("accelerate>=1.14.0") for item in requirements)
    assert any(str(item).startswith("safetensors>=0.8.0") for item in requirements)


def test_runtime_smokes_assert_the_supported_hf_stack() -> None:
    makefile = ROOT.joinpath("Makefile").read_text(encoding="utf-8")

    for expected in (
        "import accelerate, safetensors, torch, transformers",
        "accelerate.__version__ == '1.14.0'",
        "safetensors.__version__ == '0.8.0'",
        "transformers.__version__ == '5.14.1'",
    ):
        assert makefile.count(expected) == 2


def test_runtime_platform_locks_are_cpu_only_and_hash_locked() -> None:
    locks = (
        ROOT.joinpath("requirements", "workflows", "runtime-image-py312.txt"),
        ROOT.joinpath("requirements", "workflows", "runtime-image-py312-aarch64.txt"),
    )
    for path in locks:
        text = path.read_text(encoding="utf-8")
        assert "torch==2.13.0+cpu" in text
        assert "--hash=sha256:" in text
        assert "transformers==" in text
        assert "safetensors==" in text
        assert "nvidia-cublas-cu12" not in text
        assert "nvidia-cuda-runtime-cu12" not in text
        assert "triton==" not in text
        assert "gptqmodel==" not in text
        assert "bitsandbytes==" not in text


def test_cuda_runtime_is_a_separate_minimal_hf_image() -> None:
    text = ROOT.joinpath("runtime", "Dockerfile.cuda").read_text(encoding="utf-8")

    assert "FROM ${RUNTIME_BUILD_BASE_IMAGE} AS public-wheel" in text
    assert "FROM ${RUNTIME_BASE_IMAGE}" in text
    assert "runtime-image-py312-cu128.txt" in text
    assert "https://download.pytorch.org/whl/cu128" in text
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in text
    assert "NVIDIA_VISIBLE_DEVICES=all" not in text
    assert 'test "${TARGETARCH:-amd64}" = amd64' in text
    assert "--wheel-dir /wheelhouse" in text
    assert "--require-hashes" in text
    assert "--no-deps" in text
    assert 'ENTRYPOINT ["python", "-m", "invarlock"]' in text
    assert "runtime-image-quant" not in text
    assert "bitsandbytes" not in text
    assert "gptq" not in text.lower()
    assert 'org.opencontainers.image.revision="${INVARLOCK_SOURCE_COMMIT}"' in text
    assert (
        'dev.invarlock.source-bundle-sha256="${INVARLOCK_SOURCE_BUNDLE_SHA256}"' in text
    )


def test_cuda_runtime_lock_is_hash_locked_and_cuda_specific() -> None:
    text = ROOT.joinpath(
        "requirements", "workflows", "runtime-image-py312-cu128.txt"
    ).read_text(encoding="utf-8")

    assert "torch==2.11.0+cu128" in text
    assert "nvidia-cublas-cu12==" in text
    assert "nvidia-cuda-runtime-cu12==" in text
    assert "triton==" in text
    assert "--hash=sha256:" in text
    assert "accelerate==1.14.0" in text
    assert "transformers==5.14.1" in text
    assert "safetensors==0.8.0" in text
    assert "bitsandbytes==" not in text
    assert "gptqmodel==" not in text


def test_dockerignore_exposes_only_the_canonical_runtime_inputs() -> None:
    text = ROOT.joinpath(".dockerignore").read_text(encoding="utf-8")

    assert "**" in text
    for item in (
        "!README.md",
        "!pyproject.toml",
        "!contracts/**",
        "!runtime/Dockerfile",
        "!requirements/workflows/runtime-image-py312.txt",
        "!requirements/workflows/runtime-image-py312-aarch64.txt",
        "!requirements/workflows/runtime-image-py312-cu128.txt",
        "!requirements/workflows/runtime-wheel-build-py312.txt",
        "!runtime/Dockerfile.cuda",
        "!LICENSE",
        "!MANIFEST.in",
        "!src/**",
    ):
        assert item in text
    assert "runtime-image-quant" not in text


def test_make_exposes_separate_cuda_build_and_gpu_smoke_targets() -> None:
    text = ROOT.joinpath("Makefile").read_text(encoding="utf-8")

    assert "RUNTIME_IMAGE_CUDA ?= invarlock-runtime:hf-cuda-local" in text
    assert "runtime-image-cuda:" in text
    assert "scripts/authenticated_runtime_build.py" in text
    assert "--dockerfile runtime/Dockerfile.cuda" in text
    assert '--image "$(RUNTIME_IMAGE_CUDA)"' in text
    assert '--source-bundle "$(RUNTIME_SOURCE_BUNDLE)"' in text
    assert "--platform linux/amd64" in text
    assert "runtime-smoke-cuda:" in text
    assert "$(RUNTIME_CUDA_DEVICE_ARGS)" in text
    assert "--device nvidia.com/gpu=all,--gpus all" in text
    assert "assert torch.version.cuda == '12.8'" in text
    assert "assert torch.cuda.is_available()" in text
    assert "runtime-image-cuda-quant" not in text
