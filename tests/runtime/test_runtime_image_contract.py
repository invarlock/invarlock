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
    assert "COPY requirements/workflows/runtime-wheel-build-py312.txt" in text
    assert "ARG RUNTIME_REQUIREMENTS_AMD64" in text
    assert "ARG RUNTIME_REQUIREMENTS_ARM64" in text
    assert "ARG RUNTIME_CUDA_HOME" in text
    assert "ARG RUNTIME_KEEP_BUILD_TOOLCHAIN=0" in text
    assert "ARG RUNTIME_KEEP_BUILD_TOOLCHAIN" in text
    assert "ARG RUNTIME_PATH_PREFIX" in text
    assert "ARG PYTORCH_EXTRA_INDEX_URL" in text
    assert "ARG SOURCE_DATE_EPOCH" in text
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in text
    assert "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" in text
    assert "CUDA_HOME=${RUNTIME_CUDA_HOME}" in text
    assert "PATH=${RUNTIME_PATH_PREFIX}${PATH}" in text
    assert '--extra-index-url "${PYTORCH_EXTRA_INDEX_URL}"' in text
    assert 'amd64) echo "/opt/invarlock/${RUNTIME_REQUIREMENTS_AMD64}"' in text
    assert 'arm64) echo "/opt/invarlock/${RUNTIME_REQUIREMENTS_ARM64}"' in text
    assert "apt_get install -y --no-install-recommends build-essential" in text
    assert 'apt-get -o "APT::Snapshot=${RUNTIME_APT_SNAPSHOT}"' in text
    assert "python3 python3-pip python3-venv python-is-python3" in text
    assert 'if [ "${RUNTIME_KEEP_BUILD_TOOLCHAIN}" = "1" ]' in text
    assert "apt_get install -y --no-install-recommends python3-dev" in text
    assert 'if [ "${RUNTIME_KEEP_BUILD_TOOLCHAIN}" != "1" ]' in text
    assert "apt_get purge -y --auto-remove build-essential" in text
    assert "python -m pip install" in text
    assert "--require-hashes" in text
    assert "python -m pip install --no-deps -e /opt/invarlock" not in text
    assert (
        "pip install --index-url https://download.pytorch.org/whl/cpu torch" not in text
    )
    assert "FROM ${RUNTIME_BUILD_BASE_IMAGE} AS public-wheel" in text
    assert "--wheel-dir /wheelhouse" in text
    assert "--no-build-isolation" in text
    assert "--no-compile" in text
    assert "/opt/invarlock/artifacts/${wheel_name}" in text
    assert "PYTHONPATH=/opt/invarlock/src" not in text


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
    assert "!requirements/workflows/runtime-wheel-build-py312.txt" in text
    assert "!LICENSE" in text
    assert "!MANIFEST.in" in text
    assert "!src/**" in text
    assert "src/**/__pycache__/" in text
    assert "src/*.egg-info/**" in text


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
    assert "hqq==" in text
    assert "optimum-quanto==" in text
    assert "compressed-tensors==" in text
    assert "torchao==" in text
    assert "peft==" in text
    assert "gptqmodel==6.0.3" in text
    assert "autoawq==" not in text


def test_runtime_dockerfile_bootstraps_gptqmodel_source_metadata_from_locked_torch() -> (
    None
):
    text = (Path.cwd() / "runtime" / "Dockerfile").read_text(encoding="utf-8")

    assert "torch_bootstrap_path=/tmp/invarlock-torch-bootstrap.txt" in text
    assert "setuptools_bootstrap_path=/tmp/invarlock-setuptools-bootstrap.txt" in text
    assert "pypcre_bootstrap_path=/tmp/invarlock-pypcre-bootstrap.txt" in text
    assert "gptqmodel_bootstrap_path=/tmp/invarlock-gptqmodel-bootstrap.txt" in text
    assert "grep -q '^gptqmodel=='" in text
    assert "awk '/^torch==/{capture=1}" in text
    assert "awk '/^setuptools==/{capture=1}" in text
    assert "awk '/^pypcre==/{capture=1}" in text
    assert "awk '/^gptqmodel==/{capture=1}" in text
    assert "BUILD_CUDA_EXT=0 python -m pip install" in text
    assert "--no-build-isolation" in text
    assert "--no-deps" in text


def test_runtime_dockerfile_builds_a_deterministic_final_filesystem_layer() -> None:
    text = (Path.cwd() / "runtime" / "Dockerfile").read_text(encoding="utf-8")
    final_stage = text.split("FROM ${RUNTIME_BASE_IMAGE}", maxsplit=1)[1]

    assert "FROM scratch AS runtime-inputs" in text
    assert (
        "RUN --mount=type=bind,from=runtime-inputs,source=/,"
        "target=/mnt/invarlock-inputs,ro"
    ) in final_stage
    assert (
        "--mount=type=bind,from=public-wheel,source=/wheelhouse,"
        "target=/mnt/invarlock-wheel,ro"
    ) in final_stage
    assert "COPY " not in final_stage
    assert re.search(
        r'CFLAGS="-g0" python -m pip install.*?'
        r'-r "\$\{pypcre_bootstrap_path\}"',
        final_stage,
        flags=re.DOTALL,
    )
    assert not re.search(
        r'CFLAGS="-g0" python -m pip install.*?'
        r'-r "\$\{setuptools_bootstrap_path\}"',
        final_stage,
        flags=re.DOTALL,
    )
    assert "Dir::Cache::archives=/tmp/invarlock-apt-cache/archives" in final_stage
    assert "Dir::Log=/tmp/invarlock-apt-log" in final_stage
    assert "Dir::State::lists=/tmp/invarlock-apt-state/lists" in final_stage
    assert "Dpkg::Options::=--log=/tmp/invarlock-dpkg.log" in final_stage
    assert ": > /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list" in final_stage
    assert (
        "rm -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list" not in final_stage
    )
    assert "/tmp/invarlock-preserved-state/var/cache/ldconfig/aux-cache" in final_stage
    assert "/tmp/invarlock-preserved-state/var/log/alternatives.log" in final_stage
    assert "apt_get purge -y --auto-remove build-essential" in final_stage
    assert "/var/lib/apt/lists/*" not in final_stage
    assert "/var/log/apt/*" not in final_stage
    assert "/tmp/invarlock-*-bootstrap.txt" in final_stage
    assert 'touch -h -d "@${SOURCE_DATE_EPOCH}"' in final_stage
    assert "find / -xdev" in final_stage
    assert "-path /mnt/invarlock-inputs" in final_stage
    assert "-path /mnt/invarlock-wheel" in final_stage
    assert '-type d -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +' in final_stage
    for normalized_path in (
        "/usr/local/lib/python3.12",
        "/usr/local/bin",
        "/usr/lib/python3",
        "/usr/lib/python3.12",
        "/var/lib/apt",
        "/var/lib/dpkg",
        "/var/cache/debconf",
        "/etc/apt",
        "/etc/ssl",
        "/usr/share/python3",
    ):
        assert normalized_path in final_stage


def test_cuda_quant_runtime_smoke_covers_supported_quant_adapters() -> None:
    root = Path.cwd()
    makefile_text = (root / "Makefile").read_text(encoding="utf-8")
    smoke_text = (
        root
        / "examples"
        / "integrations"
        / "_runtime_images"
        / "quant_runtime_image_smoke.py"
    ).read_text(encoding="utf-8")

    assert (
        "examples/integrations/_runtime_images/quant_runtime_image_smoke.py"
        in makefile_text
    )
    for adapter in (
        "hf_bnb",
        "hf_awq",
        "hf_gptq",
        "hf_torchao",
        "hf_hqq",
        "hf_quanto",
        "hf_ct",
    ):
        assert adapter in smoke_text
    for backend in (
        "bitsandbytes",
        "gptqmodel",
        "torchao",
        "hqq",
        "optimum.quanto",
        "compressed_tensors",
    ):
        assert backend in smoke_text
    assert '"peft"' in smoke_text
    assert "require_gptqmodel_runtime" in smoke_text
    assert "_require_gptqmodel_runtime(" in smoke_text
    assert "_check_peft_awq_dispatcher(" in smoke_text
    assert "dispatch_awq(" in smoke_text
    assert "torch.nn.Linear(1, 1)" in smoke_text
    assert "_patch_gptqmodel_transformers_hub_compat" not in smoke_text


def test_cuda_runtime_smokes_require_a_selected_gpu_and_execute_cuda_work() -> None:
    root = Path.cwd()
    makefile_text = (root / "Makefile").read_text(encoding="utf-8")
    smoke_text = (
        root
        / "examples"
        / "integrations"
        / "_runtime_images"
        / "quant_runtime_image_smoke.py"
    ).read_text(encoding="utf-8")

    assert "RUNTIME_CUDA_DOCKER_GPUS ?= all" in makefile_text
    assert '--gpus "$(RUNTIME_CUDA_DOCKER_GPUS)"' in makefile_text
    assert "CUDA tensor execution ok" in makefile_text
    assert "--require-cuda-toolchain --require-gpu" in makefile_text
    assert 'device="cuda"' in smoke_text
    assert "torch.cuda.synchronize()" in smoke_text


def test_quant_runtime_input_pins_compatible_gptqmodel_dependencies() -> None:
    root = Path.cwd()
    text = (root / "requirements" / "workflows" / "runtime-image-quant.in").read_text(
        encoding="utf-8"
    )

    assert "gptqmodel==6.0.3" in text
    assert "kernels==0.14.1" in text


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
