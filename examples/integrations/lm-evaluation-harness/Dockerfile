ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID

LABEL org.invarlock.example.base-image-id="${BASE_IMAGE_ID}"

# LM Evaluation Harness is deliberately scoped to this integration image. The
# resulting image is inspected after this layer is built and that immutable
# image ID is carried into both runtime receipts.
COPY requirements/workflows/lm-evaluation-harness-py312.txt \
    /tmp/lm-evaluation-harness-requirements.txt
RUN python -m pip install \
      --no-compile \
      --require-hashes \
      -r /tmp/lm-evaluation-harness-requirements.txt

COPY examples/integrations/lm-evaluation-harness/example.py \
    /opt/invarlock/examples/lm-evaluation-harness-example.py

ENTRYPOINT ["python", "/opt/invarlock/examples/lm-evaluation-harness-example.py", "worker"]
