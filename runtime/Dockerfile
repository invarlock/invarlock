ARG TARGETARCH
ARG RUNTIME_BASE_IMAGE=python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
ARG RUNTIME_BUILD_BASE_IMAGE=python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
ARG RUNTIME_REQUIREMENTS_AMD64=requirements/workflows/runtime-image-py312.txt
ARG RUNTIME_REQUIREMENTS_ARM64=requirements/workflows/runtime-image-py312-aarch64.txt
ARG PYTORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG SOURCE_DATE_EPOCH=0

FROM scratch AS runtime-inputs
COPY requirements/workflows/runtime-image-py312.txt /requirements/workflows/runtime-image-py312.txt
COPY requirements/workflows/runtime-image-py312-aarch64.txt /requirements/workflows/runtime-image-py312-aarch64.txt
COPY requirements/workflows/runtime-wheel-build-py312.txt /requirements/workflows/runtime-wheel-build-py312.txt
COPY pyproject.toml README.md LICENSE MANIFEST.in /project/
COPY contracts /project/contracts
COPY src /project/src

FROM ${RUNTIME_BUILD_BASE_IMAGE} AS public-wheel
ARG SOURCE_DATE_EPOCH
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
COPY --from=runtime-inputs /requirements/workflows/runtime-wheel-build-py312.txt /tmp/runtime-wheel-build.txt
COPY --from=runtime-inputs /project /project
RUN python -m pip install \
      --require-hashes \
      --no-deps \
      -r /tmp/runtime-wheel-build.txt \
    && python -m pip wheel \
      --no-build-isolation \
      --no-deps \
      --wheel-dir /wheelhouse \
      /project \
    && set -- /wheelhouse/invarlock-*.whl \
    && test "$#" -eq 1 \
    && touch -h -d "@${SOURCE_DATE_EPOCH}" "$1"

FROM ${RUNTIME_BASE_IMAGE}
ARG TARGETARCH
ARG INVARLOCK_SOURCE_BUNDLE_SHA256
ARG INVARLOCK_SOURCE_COMMIT
ARG RUNTIME_REQUIREMENTS_AMD64
ARG RUNTIME_REQUIREMENTS_ARM64
ARG PYTORCH_EXTRA_INDEX_URL
ARG SOURCE_DATE_EPOCH

RUN python -c 'import re,sys; commit,digest=sys.argv[1:]; re.fullmatch(r"[0-9a-f]{40,64}", commit) or sys.exit("invalid INVARLOCK_SOURCE_COMMIT"); re.fullmatch(r"sha256:[0-9a-f]{64}", digest) or sys.exit("invalid INVARLOCK_SOURCE_BUNDLE_SHA256")' \
      "${INVARLOCK_SOURCE_COMMIT}" "${INVARLOCK_SOURCE_BUNDLE_SHA256}"

LABEL dev.invarlock.source-bundle-sha256="${INVARLOCK_SOURCE_BUNDLE_SHA256}" \
      org.opencontainers.image.revision="${INVARLOCK_SOURCE_COMMIT}"

ENV HF_HUB_OFFLINE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    TRANSFORMERS_OFFLINE=1

# Install the canonical Hugging Face runtime and the exact wheel built from
# this source tree. Optional providers are separate distributions and images.
RUN --mount=type=bind,from=runtime-inputs,source=/requirements,target=/mnt/requirements,ro \
    --mount=type=bind,from=public-wheel,source=/wheelhouse,target=/mnt/wheelhouse,ro \
    requirements_path="$(case "${TARGETARCH:-amd64}" in \
      amd64) echo "/mnt/requirements/workflows/runtime-image-py312.txt" ;; \
      arm64) echo "/mnt/requirements/workflows/runtime-image-py312-aarch64.txt" ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac)" \
    && python -m pip install \
      --require-hashes \
      --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}" \
      -r "${requirements_path}" \
    && set -- /mnt/wheelhouse/invarlock-*.whl \
    && test "$#" -eq 1 \
    && install -d /opt/invarlock/artifacts \
    && install -m 0644 "$1" "/opt/invarlock/artifacts/$(basename "$1")" \
    && python -m pip install \
      --no-compile \
      --no-deps \
      --no-index \
      "/opt/invarlock/artifacts/$(basename "$1")" \
    && find /opt/invarlock /usr/local/lib/python3.12 /usr/local/bin \
      -xdev -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

WORKDIR /work
ENTRYPOINT ["python", "-m", "invarlock"]
