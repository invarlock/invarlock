# Native CUDA13 BF16 candidate; exact platform manifest, not a mutable tag.
FROM nvidia/cuda@sha256:a85c9f5af049f0ab679c1669ae6fa8393022886739af7361e85bb96878e8cdd4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SGLANG_BUILD_RUST_EXTS=none \
    SGLANG_RUST_BUILD_MODE=never \
    SGLANG_USE_SGL_FA3_KERNEL=true \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.0.0.dev0+invarlock.k2.392841f.1 \
    MAX_JOBS=2 \
    CUDA_HOME=/usr/local/cuda

# Install the separately resolved and hashed Ubuntu security/build package bundle.
COPY os-security-pins.txt /usr/share/invarlock-k2/os-security-pins.txt
COPY apt/debs /tmp/apt-debs
COPY apt/deb-artifacts.sha256 /usr/share/invarlock-k2/deb-artifacts.sha256
COPY apt/deb-packages.tsv /usr/share/invarlock-k2/deb-packages.tsv
COPY apt/repository-metadata /usr/share/invarlock-k2/repository-metadata
RUN --network=none sed 's|/out/debs/|/tmp/apt-debs/|' /usr/share/invarlock-k2/deb-artifacts.sha256 | sha256sum -c - \
    && mkdir -p /tmp/empty-apt-lists \
    && apt-get -o Dir::Etc::sourcelist=- -o Dir::Etc::sourceparts=- \
       -o Dir::State::lists=/tmp/empty-apt-lists \
       --yes --no-install-recommends install /tmp/apt-debs/*.deb \
    && while IFS= read -r pin; do \
         name="${pin%%=*}"; required="${pin#*=}"; \
         test "$(dpkg-query -W -f='${Version}' "$name")" = "$required"; \
       done < /usr/share/invarlock-k2/os-security-pins.txt \
    && dpkg-query -W > /usr/share/invarlock-k2/os-packages.txt \
    && rm -rf /var/lib/apt/lists/* /tmp/apt-debs \
    && python3 -m venv /opt/k2

ENV PATH=/opt/k2/bin:$PATH
COPY bootstrap /usr/share/invarlock-k2/bootstrap
# Replace Ubuntu's venv seed using only the independently authenticated wheel.
RUN --network=none sha256sum -c /usr/share/invarlock-k2/bootstrap/pip-wheel.sha256 \
    && python -m pip --isolated install --no-index --no-deps --no-cache-dir \
       /usr/share/invarlock-k2/bootstrap/pip-26.2-py3-none-any.whl \
    && python -c 'import pip; assert pip.__version__ == "26.2"'
COPY requirements.txt /usr/share/invarlock-k2/requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
    -r /usr/share/invarlock-k2/requirements.txt

COPY apt/package-indexes /usr/share/invarlock-k2/package-indexes
COPY apt/package-verification.json /usr/share/invarlock-k2/package-verification.json
COPY source /tmp/sglang-source
COPY core /tmp/core
COPY examples /opt/campaign/examples
COPY native_probe.py /opt/campaign/native_probe.py
COPY build-inputs.json /usr/share/invarlock-k2/build-inputs.json
RUN --network=none cd /tmp/sglang-source/python \
    && python -m build --wheel --no-isolation --outdir /tmp/runtime-wheel \
    && python -m pip install --no-deps --no-index \
       /tmp/runtime-wheel/*.whl /tmp/core/*.whl \
    && apt-get --yes --no-download --no-auto-remove purge \
       python3-pip-whl python3.12-venv python3-venv \
    && python -m pip check \
    && dpkg-query -W > /usr/share/invarlock-k2/os-packages.txt \
    && cp /tmp/sglang-source/source-derivation.json /usr/share/invarlock-k2/ \
    && python -m pip list --format=json > /usr/share/invarlock-k2/python-packages.json \
    && sha256sum /usr/share/invarlock-k2/bootstrap/*.whl /tmp/runtime-wheel/*.whl /tmp/core/*.whl > /usr/share/invarlock-k2/wheel-artifacts.sha256 \
    && rm -rf /tmp/sglang-source /tmp/runtime-wheel /tmp/core

WORKDIR /opt/campaign
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["python", "/opt/campaign/native_probe.py"]
