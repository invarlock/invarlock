ARG TARGETARCH
FROM python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
ARG TARGETARCH

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/invarlock/src

WORKDIR /opt/invarlock

COPY requirements/workflows/runtime-image-py312.txt /opt/invarlock/requirements/workflows/runtime-image-py312.txt
COPY requirements/workflows/runtime-image-py312-aarch64.txt /opt/invarlock/requirements/workflows/runtime-image-py312-aarch64.txt
# The local runtime image is the attested CPU execution surface used by the
# smoke workflows. Install only the runtime execution stack from the pinned
# requirements surface, then execute the repo source directly via PYTHONPATH.
RUN python -m pip install \
    --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r "$(case "${TARGETARCH:-amd64}" in \
      amd64) echo /opt/invarlock/requirements/workflows/runtime-image-py312.txt ;; \
      arm64) echo /opt/invarlock/requirements/workflows/runtime-image-py312-aarch64.txt ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac)"

COPY src /opt/invarlock/src

ENTRYPOINT ["python", "-m", "invarlock"]
