# TensorRT-LLM engine comparison

This one-command example downloads a revision-pinned Qwen3-0.6B checkpoint,
builds a source-authenticated TensorRT-LLM 1.2.1 runtime image, and converts it
into BF16 and ModelOpt-calibrated FP8 single-rank H100 engines. It then
compares them through InvarLock's public `evaluate`, `verify`, and `report`
commands. Both engine builds run concurrently on separate GPUs; evaluation
also runs the baseline and subject workers concurrently.

## Prerequisites

The maintained showcase requires Linux, Docker with two visible H100 GPUs,
network access for the pinned model downloads and runtime-image build, and
roughly 20 GB of temporary disk space. Both engines originate from the same
public Apache-2.0 checkpoint and therefore share one authenticated tokenizer
contract. The maintained 102-record schedule is also the FP8 calibration
input. Run it from a committed checkout:

```bash
make example-tensorrt-llm \
  EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/tensorrt-llm"
```

Use `EXAMPLE_ARGS="--workspace /new/path"` to choose a new output directory.
The command rejects dirty tracked source because the runtime image is built
from the exact committed Git archive. TensorRT-LLM engine bytes are not
assumed to be reproducible across builds. Instead, the transaction
authenticates each newly built engine identity together with the pinned model
revision, runtime image, tokenizer contract, schedule, and policy.

## Prepared-engine transaction

Advanced users who already have qualified engines can run the lower-level
transaction without rebuilding them. Install matching released wheels for
`invarlock` and `invarlock-runtime-tensorrt-llm`, then prepare a new input
directory with this fixed layout:

```text
tensorrt-inputs/
├── baseline-engine/
├── subject-engine/
├── tokenizer-contract.json
├── records.jsonl
└── policy.json
```

`records.jsonl` uses the `id`, `prompt`, and `expected` fields. `policy.json`
is the acceptance policy chosen before viewing the subject results. The input
directory and all engine files must be readable by numeric runtime user
`65532:65532`.

Use two GPU indices to execute the baseline and subject workers concurrently.
The maintained one-command showcase requires two distinct indices.

```bash
make example-tensorrt-llm-prepared EXAMPLE_ARGS="\
  --runtime-image 'registry.example/invarlock-tensorrt@sha256:PINNED_DIGEST' \
  --resource-root "$PWD/tensorrt-inputs" \
  --baseline-locator 'hf://owner/baseline@REVISION#tensorrt-llm-engine' \
  --subject-locator 'hf://owner/subject@REVISION#tensorrt-llm-engine' \
  --baseline-device cuda:0 \
  --subject-device cuda:1 \
  --evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/tensorrt-llm"
```

Both commands inspect the engines in the authenticated image with
networking disabled. They then create a request and separate trust profile,
run execution-free preflight, evaluate both engines, strictly verify the
signed evidence, and render an HTML report. Existing request, evidence, or
verifier-output paths are never overwritten. The prepared workflow requires
the caller-owned keys and trust root shown above; those materials are never
generated inside the transaction workspace.

The 102-record schedule covers factual, numeric, temporal, spatial, and common
language completions. Its policy, selected before execution, requires all 102
records, limits the paired 95% confidence-interval width to 20 percentage
points, and rejects a regression larger than 10 percentage points. The command
also requires each engine to solve at least 40% of the records, so an
uninformative zero-correct comparison cannot be presented as a successful
showcase. Its result supports only the bound one-token exact-match comparison;
broader model quality and TensorRT performance require separate measurements.
