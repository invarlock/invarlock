# Optional vision-text provider layered on an exact InvarLock CUDA image.
ARG RUNTIME_BASE_IMAGE
ARG WHEEL_BUILD_BASE=python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
ARG SOURCE_DATE_EPOCH=0

FROM scratch AS runtime-inputs
COPY requirements/workflows/multimodal-runtime-py312.txt /requirements/multimodal-runtime-py312.txt
COPY requirements/workflows/runtime-wheel-build-py312.txt /requirements/runtime-wheel-build-py312.txt
COPY addins/multimodal/pyproject.toml addins/multimodal/README.md /addin/
COPY addins/multimodal/src /addin/src

FROM ${WHEEL_BUILD_BASE} AS addin-wheel
ARG SOURCE_DATE_EPOCH
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
COPY --from=runtime-inputs /requirements/runtime-wheel-build-py312.txt /tmp/runtime-wheel-build.txt
COPY --from=runtime-inputs /addin /addin
RUN python -m pip install --require-hashes --no-deps -r /tmp/runtime-wheel-build.txt \
    && python -m pip wheel --no-build-isolation --no-deps --wheel-dir /wheelhouse /addin \
    && set -- /wheelhouse/invarlock_runtime_hf_vision_text-*.whl \
    && test "$#" -eq 1 \
    && touch -h -d "@${SOURCE_DATE_EPOCH}" "$1"

FROM ${RUNTIME_BASE_IMAGE}
ARG SOURCE_DATE_EPOCH
ENV HF_HUB_OFFLINE=1 \
    INVARLOCK_ALLOW_NETWORK=0 \
    INVARLOCK_ALLOW_REMOTE_CODE=0 \
    INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    TRANSFORMERS_OFFLINE=1
COPY --from=runtime-inputs /requirements/multimodal-runtime-py312.txt /opt/invarlock/requirements/multimodal-runtime-py312.txt
COPY --from=addin-wheel /wheelhouse /opt/invarlock/addin-wheels
RUN python -m pip install \
      --no-compile \
      --require-hashes \
      --only-binary=:all: \
      -r /opt/invarlock/requirements/multimodal-runtime-py312.txt \
    && set -- /opt/invarlock/addin-wheels/invarlock_runtime_hf_vision_text-*.whl \
    && test "$#" -eq 1 \
    && python -m pip install --no-compile --no-deps --no-index "$1" \
    && python -m invarlock_addins.multimodal.conformance \
    && python -c "from PIL import Image; assert Image.__version__ == '12.3.0'" \
    && find /opt/invarlock /usr/local/lib/python3.12 /usr/local/bin \
      -xdev -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +
WORKDIR /work
ENTRYPOINT ["python", "-m", "invarlock"]
