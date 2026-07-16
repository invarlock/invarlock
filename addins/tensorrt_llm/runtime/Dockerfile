# Runtime image for the optional TensorRT-LLM provider distribution.
ARG WHEEL_BUILD_BASE=python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
ARG SOURCE_DATE_EPOCH=0

FROM scratch AS runtime-inputs
COPY requirements/workflows/core-py312.txt /requirements/workflows/core-py312.txt
COPY requirements/workflows/runtime-wheel-build-py312.txt /requirements/workflows/runtime-wheel-build-py312.txt
COPY pyproject.toml README.md LICENSE MANIFEST.in /project/
COPY src /project/src
COPY addins/tensorrt_llm/pyproject.toml addins/tensorrt_llm/README.md /addin/
COPY addins/tensorrt_llm/src /addin/src

FROM ${WHEEL_BUILD_BASE} AS public-wheel
ARG SOURCE_DATE_EPOCH
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
COPY --from=runtime-inputs /requirements/workflows/runtime-wheel-build-py312.txt /tmp/runtime-wheel-build.txt
COPY --from=runtime-inputs /project /project
COPY --from=runtime-inputs /addin /addin
RUN python -m pip install \
      --require-hashes \
      --no-deps \
      -r /tmp/runtime-wheel-build.txt \
    && python -m pip wheel \
      --no-build-isolation \
      --no-deps \
      --wheel-dir /wheelhouse \
      /project \
    && python -m pip wheel \
      --no-build-isolation \
      --no-deps \
      --wheel-dir /wheelhouse \
      /addin \
    && set -- /wheelhouse/invarlock-*.whl \
    && test "$#" -eq 1 \
    && set -- /wheelhouse/invarlock_runtime_tensorrt_llm-*.whl \
    && test "$#" -eq 1 \
    && touch -h -d "@${SOURCE_DATE_EPOCH}" /wheelhouse/*.whl

FROM ${WHEEL_BUILD_BASE} AS cli-dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=runtime-inputs /requirements/workflows/core-py312.txt /tmp/core-py312.txt
RUN python -m pip download \
      --require-hashes \
      --only-binary=:all: \
      --dest /dependency-wheels \
      -r /tmp/core-py312.txt

# NVIDIA publishes this multi-architecture manifest for the stable TensorRT-LLM
# 1.2.1 release. Digest pinning prevents a mutable tag from changing the runtime
# beneath an authenticated provider receipt.
FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.1@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9

ARG SOURCE_DATE_EPOCH
ARG TENSORRT_LLM_VERSION=1.2.1
ARG TENSORRT_LLM_SOURCE_COMMIT=376f7e1bd8ed543f75014309e3fd4b237e9b0e73
ARG TENSORRT_LLM_BASE_DIGEST=sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9

ENV HF_HUB_OFFLINE=1 \
    INVARLOCK_ALLOW_NETWORK=0 \
    INVARLOCK_TENSORRT_LLM_RUNNER=/opt/invarlock/bin/tensorrt-llm-runner \
    PATH="/opt/invarlock/cli-venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_INDEX=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    TRANSFORMERS_OFFLINE=1

LABEL dev.invarlock.runtime-provider="tensorrt_llm" \
      dev.invarlock.runtime-runner="/opt/invarlock/bin/tensorrt-llm-runner" \
      dev.invarlock.tensorrt-llm.base-digest="${TENSORRT_LLM_BASE_DIGEST}" \
      dev.invarlock.tensorrt-llm.base-url="https://catalog.ngc.nvidia.com/orgs/nvidia/tensorrt-llm/containers/release/1.2.1" \
      dev.invarlock.tensorrt-llm.source-commit="${TENSORRT_LLM_SOURCE_COMMIT}" \
      dev.invarlock.tensorrt-llm.source-url="https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1" \
      dev.invarlock.tensorrt-llm.version="${TENSORRT_LLM_VERSION}"

COPY --from=runtime-inputs /requirements/workflows/core-py312.txt /opt/invarlock/requirements/core-py312.txt
COPY --from=public-wheel /wheelhouse /opt/invarlock/artifacts
COPY --from=cli-dependencies /dependency-wheels /opt/invarlock/dependency-wheels

# TensorRT-LLM's release image owns its CUDA/PyTorch dependency closure. Keep the
# process runner in that vendor environment, but install the public CLI and its
# hash-pinned core closure in an isolated virtual environment.
RUN set -- /opt/invarlock/artifacts/invarlock-*.whl \
    && test "$#" -eq 1 \
    && core_wheel="$1" \
    && set -- /opt/invarlock/artifacts/invarlock_runtime_tensorrt_llm-*.whl \
    && test "$#" -eq 1 \
    && addin_wheel="$1" \
    && vendor_python="$(command -v python)" \
    && PIP_CONSTRAINT= python -m pip install \
      --no-compile \
      --no-deps \
      --no-index \
      "${core_wheel}" "${addin_wheel}" \
    && "${vendor_python}" -c "import importlib.metadata as m; import invarlock; assert m.version('invarlock') == invarlock.__version__" \
    && install -d -m 0755 /opt/invarlock/bin \
    && runner="$(command -v invarlock-tensorrt-llm-runner)" \
    && test -n "${runner}" \
    && install -m 0555 "${runner}" /opt/invarlock/bin/tensorrt-llm-runner \
    && ln -s "${vendor_python}" /opt/invarlock/bin/vendor-python \
    && test -x /opt/invarlock/bin/tensorrt-llm-runner \
    && "${vendor_python}" -c "import sys; before=set(sys.modules); from invarlock_addins.tensorrt_llm import runner as r; loaded=set(sys.modules)-before; assert callable(r.main); assert not any(name == 'torch' or name == 'transformers' or name == 'tensorrt_llm' or name.startswith(('torch.', 'transformers.', 'tensorrt_llm.')) for name in loaded)" \
    && "${vendor_python}" -m invarlock_addins.tensorrt_llm.conformance \
    && "${vendor_python}" -m venv /opt/invarlock/cli-venv \
    && /opt/invarlock/cli-venv/bin/python -m pip install \
      --no-compile \
      --no-index \
      --find-links /opt/invarlock/dependency-wheels \
      --require-hashes \
      -r /opt/invarlock/requirements/core-py312.txt \
    && /opt/invarlock/cli-venv/bin/python -m pip install \
      --no-compile \
      --no-deps \
      --no-index \
      "${core_wheel}" "${addin_wheel}" \
    && /opt/invarlock/cli-venv/bin/invarlock-tensorrt-llm-conformance \
    && test "${TRT_LLM_VERSION}" = "${TENSORRT_LLM_VERSION}"

WORKDIR /opt/invarlock

# Keep NVIDIA's runtime initialization in front of the InvarLock CLI. Arguments
# appended by `docker run IMAGE ...` or `podman run IMAGE ...` therefore become
# InvarLock arguments instead of replacing the CLI command.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/opt/invarlock/cli-venv/bin/python", "-m", "invarlock"]
