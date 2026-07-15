ARG TARGETARCH
ARG RUNTIME_BASE_IMAGE=python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
ARG RUNTIME_BUILD_BASE_IMAGE=python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
ARG RUNTIME_REQUIREMENTS_AMD64=requirements/workflows/runtime-image-py312.txt
ARG RUNTIME_REQUIREMENTS_ARM64=requirements/workflows/runtime-image-py312-aarch64.txt
ARG RUNTIME_CUDA_HOME=
ARG RUNTIME_KEEP_BUILD_TOOLCHAIN=0
ARG RUNTIME_PATH_PREFIX=
ARG PYTORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG SOURCE_DATE_EPOCH=0

FROM scratch AS runtime-inputs
COPY requirements/workflows/runtime-image-py312.txt /requirements/workflows/runtime-image-py312.txt
COPY requirements/workflows/runtime-image-py312-aarch64.txt /requirements/workflows/runtime-image-py312-aarch64.txt
COPY requirements/workflows/runtime-image-py312-cu128.txt /requirements/workflows/runtime-image-py312-cu128.txt
COPY requirements/workflows/runtime-image-quant-py312-cu128.txt /requirements/workflows/runtime-image-quant-py312-cu128.txt
COPY requirements/workflows/runtime-wheel-build-py312.txt /requirements/workflows/runtime-wheel-build-py312.txt
COPY pyproject.toml README.md LICENSE MANIFEST.in /project/
COPY src /project/src

FROM ${RUNTIME_BUILD_BASE_IMAGE} AS public-wheel
ARG SOURCE_DATE_EPOCH
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
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
ARG RUNTIME_REQUIREMENTS_AMD64
ARG RUNTIME_REQUIREMENTS_ARM64
ARG RUNTIME_CUDA_HOME
ARG RUNTIME_KEEP_BUILD_TOOLCHAIN
ARG RUNTIME_PATH_PREFIX
ARG PYTORCH_EXTRA_INDEX_URL
ARG SOURCE_DATE_EPOCH
ARG RUNTIME_APT_SNAPSHOT=

ENV PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    CUDA_HOME=${RUNTIME_CUDA_HOME} \
    PATH=${RUNTIME_PATH_PREFIX}${PATH}

# Install the container runtime execution stack and the exact wheel built from
# this source tree.  The wheel remains in the image so the private runner can
# extract and independently pin the same artifact installed in the container.
# The discarded input stage keeps context-copy timestamps out of the final
# image. The single final filesystem layer is normalized to SOURCE_DATE_EPOCH.
RUN --mount=type=bind,from=runtime-inputs,source=/,target=/mnt/invarlock-inputs,ro \
    --mount=type=bind,from=public-wheel,source=/wheelhouse,target=/mnt/invarlock-wheel,ro \
    install -d /opt/invarlock/artifacts \
    && cp -a /mnt/invarlock-inputs/requirements /opt/invarlock/requirements \
    && install -d \
      /tmp/invarlock-apt-cache/archives/partial \
      /tmp/invarlock-apt-log \
      /tmp/invarlock-apt-state/lists/partial \
      /tmp/invarlock-preserved-state/var/cache/ldconfig \
      /tmp/invarlock-preserved-state/var/log \
    && if [ -e /var/cache/ldconfig/aux-cache ]; then \
      cp -a /var/cache/ldconfig/aux-cache \
        /tmp/invarlock-preserved-state/var/cache/ldconfig/aux-cache; \
    fi \
    && if [ -e /var/log/alternatives.log ]; then \
      cp -a /var/log/alternatives.log \
        /tmp/invarlock-preserved-state/var/log/alternatives.log; \
    fi \
    && apt_get() { \
      set -- \
        -o "Dir::Cache::archives=/tmp/invarlock-apt-cache/archives" \
        -o "Dir::Log=/tmp/invarlock-apt-log" \
        -o "Dir::State::lists=/tmp/invarlock-apt-state/lists" \
        -o "Dpkg::Options::=--log=/tmp/invarlock-dpkg.log" \
        "$@"; \
      if [ -n "${RUNTIME_APT_SNAPSHOT}" ]; then \
        apt-get -o "APT::Snapshot=${RUNTIME_APT_SNAPSHOT}" "$@"; \
      else \
        apt-get "$@"; \
      fi; \
    } \
    && if [ -n "${RUNTIME_APT_SNAPSHOT}" ] \
      && [ -e /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]; then \
      : > /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list; \
    fi \
    && apt_get update \
    && apt_get install -y --no-install-recommends build-essential jq \
    && if ! command -v python >/dev/null 2>&1; then \
      apt_get install -y --no-install-recommends ca-certificates python3 python3-pip python3-venv python-is-python3; \
    fi \
    && if [ "${RUNTIME_KEEP_BUILD_TOOLCHAIN}" = "1" ]; then \
      apt_get install -y --no-install-recommends python3-dev; \
    fi \
    && requirements_path="$(case "${TARGETARCH:-amd64}" in \
      amd64) echo "/opt/invarlock/${RUNTIME_REQUIREMENTS_AMD64}" ;; \
      arm64) echo "/opt/invarlock/${RUNTIME_REQUIREMENTS_ARM64}" ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac)" \
    && if grep -q '^gptqmodel==' "${requirements_path}"; then \
      torch_bootstrap_path=/tmp/invarlock-torch-bootstrap.txt; \
      setuptools_bootstrap_path=/tmp/invarlock-setuptools-bootstrap.txt; \
      pypcre_bootstrap_path=/tmp/invarlock-pypcre-bootstrap.txt; \
      gptqmodel_bootstrap_path=/tmp/invarlock-gptqmodel-bootstrap.txt; \
      awk '/^torch==/{capture=1} capture{print} capture && /^    #/{exit}' "${requirements_path}" > "${torch_bootstrap_path}"; \
      awk '/^setuptools==/{capture=1} capture{print} capture && /^    #/{exit}' "${requirements_path}" > "${setuptools_bootstrap_path}"; \
      awk '/^pypcre==/{capture=1} capture{print} capture && /^    #/{exit}' "${requirements_path}" > "${pypcre_bootstrap_path}"; \
      awk '/^gptqmodel==/{capture=1} capture{print} capture && /^    #/{exit}' "${requirements_path}" > "${gptqmodel_bootstrap_path}"; \
      test -s "${torch_bootstrap_path}"; \
      test -s "${setuptools_bootstrap_path}"; \
      test -s "${pypcre_bootstrap_path}"; \
      test -s "${gptqmodel_bootstrap_path}"; \
      python -m pip install \
        --require-hashes \
        --no-deps \
        --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}" \
        -r "${setuptools_bootstrap_path}"; \
      CFLAGS="-g0" python -m pip install \
        --require-hashes \
        --no-deps \
        --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}" \
        -r "${pypcre_bootstrap_path}"; \
      python -m pip install \
        --require-hashes \
        --no-deps \
        --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}" \
        -r "${torch_bootstrap_path}"; \
      BUILD_CUDA_EXT=0 python -m pip install \
        --require-hashes \
        --no-build-isolation \
        --no-deps \
        --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}" \
        -r "${gptqmodel_bootstrap_path}"; \
    fi \
    && python -m pip install \
      --require-hashes \
      --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}" \
      -r "${requirements_path}" \
    && set -- /mnt/invarlock-wheel/invarlock-*.whl \
    && test "$#" -eq 1 \
    && wheel_name="$(basename "$1")" \
    && cp "$1" "/opt/invarlock/artifacts/${wheel_name}" \
    && python -m pip install \
      --disable-pip-version-check \
      --no-compile \
      --no-deps \
      --no-index \
      "/opt/invarlock/artifacts/${wheel_name}" \
    && if [ "${RUNTIME_KEEP_BUILD_TOOLCHAIN}" != "1" ]; then \
      apt_get purge -y --auto-remove build-essential; \
    fi \
    && if [ -e /tmp/invarlock-preserved-state/var/cache/ldconfig/aux-cache ]; then \
      cp -a /tmp/invarlock-preserved-state/var/cache/ldconfig/aux-cache \
        /var/cache/ldconfig/aux-cache; \
    else \
      rm -f /var/cache/ldconfig/aux-cache; \
    fi \
    && if [ -e /tmp/invarlock-preserved-state/var/log/alternatives.log ]; then \
      cp -a /tmp/invarlock-preserved-state/var/log/alternatives.log \
        /var/log/alternatives.log; \
    else \
      rm -f /var/log/alternatives.log; \
    fi \
    && rm -rf \
      /tmp/invarlock-apt-cache \
      /tmp/invarlock-apt-log \
      /tmp/invarlock-apt-state \
      /tmp/invarlock-dpkg.log \
      /tmp/invarlock-*-bootstrap.txt \
      /tmp/invarlock-preserved-state \
    && normalization_roots=" \
      /opt/invarlock \
      /usr/local/lib/python3.12 \
      /usr/local/bin \
      /usr/local/share \
      /usr/lib/python3 \
      /usr/lib/python3.12 \
      /var/lib/apt \
      /var/lib/dpkg \
      /var/lib/python \
      /var/cache/apt \
      /var/cache/debconf \
      /var/cache/ldconfig \
      /var/log \
      /etc/apt \
      /etc/ssl \
      /etc/python3 \
      /usr/share/python3 \
      /tmp \
      /etc/ca-certificates.conf \
      /etc/ld.so.cache \
      /etc/localtime \
      /etc/timezone" \
    && for path in ${normalization_roots}; do \
      if [ -e "${path}" ] || [ -L "${path}" ]; then \
        find "${path}" -xdev -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +; \
      fi; \
    done \
    && find / -xdev \( \
      -path /dev -o \
      -path /mnt/invarlock-inputs -o \
      -path /mnt/invarlock-wheel -o \
      -path /proc -o \
      -path /run -o \
      -path /sys \
    \) -prune -o -type d -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +

WORKDIR /opt/invarlock

ENTRYPOINT ["python", "-m", "invarlock"]
