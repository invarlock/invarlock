ARG TARGETARCH
ARG RUNTIME_BASE_IMAGE=python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
ARG RUNTIME_CUDA_HOME=
ARG RUNTIME_KEEP_BUILD_TOOLCHAIN=0
ARG RUNTIME_PATH_PREFIX=
ARG PYTORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
FROM ${RUNTIME_BASE_IMAGE}
ARG TARGETARCH
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

COPY requirements.txt /opt/invarlock/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && if ! command -v python >/dev/null 2>&1; then \
      apt-get install -y --no-install-recommends ca-certificates python3 python3-pip python3-venv python-is-python3; \
    fi \
    && if [ "${RUNTIME_KEEP_BUILD_TOOLCHAIN}" = "1" ]; then \
      apt-get install -y --no-install-recommends python3-dev; \
    fi \
    && python -m pip install \
      --require-hashes \
      --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}" \
      -r /opt/invarlock/requirements.txt \
    && if [ "${RUNTIME_KEEP_BUILD_TOOLCHAIN}" != "1" ]; then \
      apt-get purge -y --auto-remove build-essential; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY src /opt/invarlock/src

ENTRYPOINT ["python", "-m", "invarlock"]
