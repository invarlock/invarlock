ARG TARGETARCH
ARG RUNTIME_BASE_IMAGE=python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
ARG RUNTIME_REQUIREMENTS_AMD64=requirements/workflows/runtime-image-py312.txt
ARG RUNTIME_REQUIREMENTS_ARM64=requirements/workflows/runtime-image-py312-aarch64.txt
ARG RUNTIME_CUDA_HOME=
ARG RUNTIME_KEEP_BUILD_TOOLCHAIN=0
ARG RUNTIME_PATH_PREFIX=
ARG PYTORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
FROM ${RUNTIME_BASE_IMAGE}
ARG TARGETARCH
ARG RUNTIME_REQUIREMENTS_AMD64
ARG RUNTIME_REQUIREMENTS_ARM64
ARG RUNTIME_CUDA_HOME
ARG RUNTIME_KEEP_BUILD_TOOLCHAIN
ARG RUNTIME_PATH_PREFIX
ARG PYTORCH_EXTRA_INDEX_URL

ENV PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/invarlock/src \
    CUDA_HOME=${RUNTIME_CUDA_HOME} \
    PATH=${RUNTIME_PATH_PREFIX}${PATH}

WORKDIR /opt/invarlock

COPY requirements/workflows/runtime-image-py312.txt /opt/invarlock/requirements/workflows/runtime-image-py312.txt
COPY requirements/workflows/runtime-image-py312-aarch64.txt /opt/invarlock/requirements/workflows/runtime-image-py312-aarch64.txt
COPY requirements/workflows/runtime-image-py312-cu128.txt /opt/invarlock/requirements/workflows/runtime-image-py312-cu128.txt
COPY requirements/workflows/runtime-image-quant-py312-cu128.txt /opt/invarlock/requirements/workflows/runtime-image-quant-py312-cu128.txt
# Install only the container runtime execution stack from the pinned
# requirements surface, then execute the repo source directly via PYTHONPATH.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && if ! command -v python >/dev/null 2>&1; then \
      apt-get install -y --no-install-recommends ca-certificates python3 python3-pip python3-venv python-is-python3; \
    fi \
    && requirements_path="$(case "${TARGETARCH:-amd64}" in \
      amd64) echo "/opt/invarlock/${RUNTIME_REQUIREMENTS_AMD64}" ;; \
      arm64) echo "/opt/invarlock/${RUNTIME_REQUIREMENTS_ARM64}" ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac)" \
    && python -m pip install \
      --require-hashes \
      --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}" \
      -r "${requirements_path}" \
    && if [ "${RUNTIME_KEEP_BUILD_TOOLCHAIN}" != "1" ]; then \
      apt-get purge -y --auto-remove build-essential; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY src /opt/invarlock/src

ENTRYPOINT ["python", "-m", "invarlock"]
