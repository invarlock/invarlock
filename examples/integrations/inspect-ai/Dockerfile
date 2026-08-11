ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID
ARG SOURCE_COMMIT
ARG SOURCE_BUNDLE_SHA256
ARG EVALUATOR_LOCK_SHA256

LABEL org.invarlock.example.base-image-id="${BASE_IMAGE_ID}"
LABEL org.invarlock.example.source-commit="${SOURCE_COMMIT}"
LABEL org.invarlock.example.source-bundle-sha256="${SOURCE_BUNDLE_SHA256}"
LABEL org.invarlock.example.evaluator="inspect-ai"
LABEL org.invarlock.example.evaluator-version="0.3.254"
LABEL org.invarlock.example.evaluator-lock-sha256="sha256:${EVALUATOR_LOCK_SHA256}"

# Inspect AI is deliberately scoped to this integration image. The resulting
# image is inspected after this layer is built and that immutable ID is carried
# into both runtime receipts.
COPY requirements/workflows/inspect-ai-runtime-py312.txt \
    /opt/invarlock/evaluator-locks/inspect-ai-runtime-requirements.txt
RUN python -m pip install \
      --no-compile \
      --no-deps \
      --require-hashes \
      -r /opt/invarlock/evaluator-locks/inspect-ai-runtime-requirements.txt \
    && python -m pip check \
    && python -c 'import importlib.metadata; assert importlib.metadata.version("inspect-ai") == "0.3.254"'

# Ship the evaluator transaction as a package and keep its caller-owned trust
# and image-inspection helpers beside it for direct module execution.
COPY examples/integrations/evaluator_transaction \
    /opt/invarlock/examples/evaluator_transaction
COPY examples/integrations/trust_material.py \
    /opt/invarlock/examples/trust_material.py
COPY examples/integrations/local_registry.py \
    /opt/invarlock/examples/local_registry.py
COPY examples/integrations/launch.py \
    /opt/invarlock/examples/launch.py
COPY examples/integrations/bounded_command.py \
    /opt/invarlock/examples/bounded_command.py
ENV INVARLOCK_EVALUATOR=inspect-ai
ENV INVARLOCK_EVALUATOR_LOCK_SHA256=sha256:${EVALUATOR_LOCK_SHA256}
WORKDIR /opt/invarlock/examples
ENTRYPOINT ["python", "-m", "evaluator_transaction.cli", "worker"]
