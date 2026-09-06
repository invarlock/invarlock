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
LABEL org.invarlock.example.evaluator="openai-evals"
LABEL org.invarlock.example.evaluator-version="3.0.1.post1+invarlock.match.1"
LABEL org.invarlock.example.evaluator-lock-sha256="sha256:${EVALUATOR_LOCK_SHA256}"
LABEL org.invarlock.example.evaluator-runtime="${EVALUATOR_RUNTIME}"

# OpenAI Evals is deliberately scoped to this integration image.
COPY requirements/workflows/openai-evals-runtime-py312.txt \
    /tmp/openai-evals-runtime-cpu.txt
COPY requirements/workflows/openai-evals-runtime-py312-cu129.txt \
    /tmp/openai-evals-runtime-cu129.txt
COPY requirements/workflows/openai-evals-upstream-wheel.txt \
    /tmp/openai-evals-upstream-wheel.txt
COPY scripts/security/build_cache_free_lm_eval_wheel.py \
    scripts/security/build_restricted_openai_evals_wheel.py /tmp/
RUN case "${EVALUATOR_RUNTIME}" in cpu|cu129) ;; *) exit 2 ;; esac \
    && install -d /opt/invarlock/evaluator-locks \
    && install -m 0644 "/tmp/openai-evals-runtime-${EVALUATOR_RUNTIME}.txt" \
      /opt/invarlock/evaluator-locks/openai-evals-runtime-requirements.txt \
    && python -m pip download \
      --dest /tmp/openai-evals-upstream \
      --no-deps \
      --only-binary=:all: \
      --require-hashes \
      -r /tmp/openai-evals-upstream-wheel.txt \
    && python /tmp/build_restricted_openai_evals_wheel.py build-wheel \
      --input /tmp/openai-evals-upstream/evals-3.0.1.post1-py3-none-any.whl \
      --output-directory /tmp/openai-evals-derived \
    && python -m pip install \
      --no-compile \
      --no-deps \
      --require-hashes \
      -r /opt/invarlock/evaluator-locks/openai-evals-runtime-requirements.txt \
    && python -m pip install \
      --no-compile \
      --no-deps \
      /tmp/openai-evals-derived/evals-3.0.1.post1+invarlock.match.1-py3-none-any.whl \
    && python -m pip check \
    && python -c 'import importlib.metadata, importlib.util; assert importlib.metadata.version("evals") == "3.0.1.post1+invarlock.match.1"; assert all(importlib.util.find_spec(name) is None for name in ("nltk", "rouge_score"))'

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
ENV INVARLOCK_EVALUATOR=openai-evals
ENV INVARLOCK_EVALUATOR_LOCK_SHA256=sha256:${EVALUATOR_LOCK_SHA256}
WORKDIR /opt/invarlock/examples
ENTRYPOINT ["python", "-m", "evaluator_transaction.cli", "worker"]
