ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID
ARG SOURCE_COMMIT
ARG SOURCE_BUNDLE_SHA256
ARG EVALUATOR_LOCK_SHA256
ARG EVALUATOR_RUNTIME=cpu

LABEL org.invarlock.example.base-image-id="${BASE_IMAGE_ID}"
LABEL org.invarlock.example.source-commit="${SOURCE_COMMIT}"
LABEL org.invarlock.example.source-bundle-sha256="${SOURCE_BUNDLE_SHA256}"
LABEL org.invarlock.example.evaluator="lm-evaluation-harness"
LABEL org.invarlock.example.evaluator-version="0.4.12+invarlock.nocache.1"
LABEL org.invarlock.example.evaluator-lock-sha256="sha256:${EVALUATOR_LOCK_SHA256}"
LABEL org.invarlock.example.evaluator-runtime="${EVALUATOR_RUNTIME}"

# LM Evaluation Harness is deliberately scoped to this integration image. The
# resulting image is inspected after this layer is built and that immutable
# image ID is carried into both runtime receipts.
COPY requirements/workflows/lm-evaluation-harness-py312.txt \
    /tmp/lm-evaluation-harness-cpu.txt
COPY requirements/workflows/lm-evaluation-harness-py312-cu129.txt \
    /tmp/lm-evaluation-harness-cu129.txt
COPY requirements/workflows/lm-evaluation-harness-upstream-wheel.txt \
    /tmp/lm-evaluation-harness-upstream-wheel.txt
COPY scripts/security/build_cache_free_lm_eval_wheel.py \
    /tmp/build-cache-free-lm-eval-wheel.py
RUN case "${EVALUATOR_RUNTIME}" in cpu|cu129) ;; *) exit 2 ;; esac \
    && install -m 0644 "/tmp/lm-evaluation-harness-${EVALUATOR_RUNTIME}.txt" \
      /tmp/lm-evaluation-harness-requirements.txt \
    && python -m pip download \
      --dest /tmp/lm-eval-upstream \
      --no-deps \
      --only-binary=:all: \
      --require-hashes \
      -r /tmp/lm-evaluation-harness-upstream-wheel.txt \
    && python /tmp/build-cache-free-lm-eval-wheel.py build-wheel \
      --input /tmp/lm-eval-upstream/lm_eval-0.4.12-py3-none-any.whl \
      --output-directory /tmp/lm-eval-derived \
    && python -m pip install \
      --no-compile \
      --no-deps \
      --require-hashes \
      -r /tmp/lm-evaluation-harness-requirements.txt \
    && python -m pip install \
      --no-compile \
      --no-deps \
      /tmp/lm-eval-derived/lm_eval-0.4.12+invarlock.nocache.1-py3-none-any.whl \
    && python -m pip check \
    && python -c 'import importlib.metadata, importlib.util; assert importlib.metadata.version("lm-eval") == "0.4.12+invarlock.nocache.1"; assert importlib.util.find_spec("sqlitedict") is None'

COPY examples/integrations/lm-evaluation-harness/example.py \
    /opt/invarlock/examples/lm-evaluation-harness-example.py
# The worker is copied as a flat script into the image. Keep the example-owned
# trust, image-inspection, attestation, and result-transfer helpers beside it.
COPY examples/integrations/trust_material.py \
    /opt/invarlock/examples/trust_material.py
COPY examples/integrations/local_registry.py \
    /opt/invarlock/examples/local_registry.py
COPY examples/integrations/launch.py \
    /opt/invarlock/examples/launch.py
COPY examples/integrations/bounded_command.py \
    /opt/invarlock/examples/bounded_command.py
COPY examples/integrations/evaluator_transaction \
    /opt/invarlock/examples/evaluator_transaction

ENTRYPOINT ["python", "/opt/invarlock/examples/lm-evaluation-harness-example.py", "worker"]
