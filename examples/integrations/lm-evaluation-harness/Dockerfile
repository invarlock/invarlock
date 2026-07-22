ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID

LABEL org.invarlock.example.base-image-id="${BASE_IMAGE_ID}"

# LM Evaluation Harness is deliberately scoped to this integration image. The
# resulting image is inspected after this layer is built and that immutable
# image ID is carried into both runtime receipts.
COPY requirements/workflows/lm-evaluation-harness-py312.txt \
    /tmp/lm-evaluation-harness-requirements.txt
COPY requirements/workflows/lm-evaluation-harness-upstream-wheel.txt \
    /tmp/lm-evaluation-harness-upstream-wheel.txt
COPY scripts/security/build_cache_free_lm_eval_wheel.py \
    /tmp/build-cache-free-lm-eval-wheel.py
RUN python -m pip download \
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

ENTRYPOINT ["python", "/opt/invarlock/examples/lm-evaluation-harness-example.py", "worker"]
