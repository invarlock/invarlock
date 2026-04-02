FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/invarlock

COPY pyproject.toml README.md /opt/invarlock/
COPY src /opt/invarlock/src
# The local runtime image is the attested CPU execution surface used by the
# smoke workflows. Install only the runtime execution stack plus the editable
# package itself without its broader local-dev dependency graph.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install \
        typer>=0.15 \
        click>=8.1 \
        shellingham>=1.5.0 \
        cryptography>=46.0.0 \
        pydantic>=2.0 \
        rich>=13.0 \
        pyyaml>=6.0 \
        markdown>=3.5 \
        psutil>=5.9 \
        jsonschema>=4.0 \
        accelerate>=0.26.0 \
        transformers>=5.0.0 \
        safetensors>=0.4.0 \
        protobuf>=4.25 \
        sentencepiece>=0.2.1 \
        tiktoken>=0.9.0 \
        datasets>=3.0 \
        requests>=2.33.0 \
        numpy>=1.24 \
        huggingface_hub>=1.0.0 \
        aiohttp>=3.12.14 \
        h2>=4.3.0 \
        pillow>=11.3.0 \
    && python -m pip install --no-deps -e /opt/invarlock

ENTRYPOINT ["python", "-m", "invarlock"]
