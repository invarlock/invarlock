FROM rust:1.81-slim AS verifier

WORKDIR /build
COPY Cargo.toml Cargo.toml
COPY crates crates
RUN cargo build --release -p invarlock-runtime-verify

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/invarlock

COPY pyproject.toml README.md /opt/invarlock/
COPY src /opt/invarlock/src
RUN python -m pip install --upgrade pip \
    && python -m pip install -e /opt/invarlock

COPY --from=verifier /build/target/release/invarlock-runtime-verify /usr/local/bin/invarlock-runtime-verify

ENTRYPOINT ["python", "-m", "invarlock"]
