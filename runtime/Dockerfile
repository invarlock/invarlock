FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/invarlock

COPY pyproject.toml README.md /opt/invarlock/
COPY src /opt/invarlock/src
# The local runtime image is the attested CPU execution surface used by the
# smoke workflows. Preinstall the CPU-only torch wheel so Linux builds do not
# pull the full CUDA distribution into the image.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -e /opt/invarlock[hf]

ENTRYPOINT ["python", "-m", "invarlock"]
