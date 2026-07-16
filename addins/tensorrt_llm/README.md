# InvarLock TensorRT-LLM runtime add-in

This optional distribution connects authenticated TensorRT-LLM engine bundles
to InvarLock's public runtime-provider ABI. Vendor imports and CUDA execution
remain isolated behind the add-in's pinned runner protocol; InvarLock core keeps
portable engine identity and independent evidence verification.

## Install and check discovery

```bash
python -m pip install invarlock-runtime-tensorrt-llm
invarlock-tensorrt-llm-conformance
```

The conformance command returns JSON with `"ok":true`, provider
`tensorrt_llm`, and the ABI version accepted by the installed core package. The
wheel registers that provider through the `invarlock.runtime_providers`
entry-point group.

## Build the runtime image

Build from the repository root so the Dockerfile can include the core and
add-in distributions from the same checkout:

```bash
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"

docker build \
  --file addins/tensorrt_llm/runtime/Dockerfile \
  --build-arg SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
  --tag invarlock-tensorrt-llm:candidate \
  .
```

The maintained add-in targets keep image construction and GPU qualification
beside the optional package:

```bash
make -C addins/tensorrt_llm build
make -C addins/tensorrt_llm smoke
make -C addins/tensorrt_llm canary \
  INPUT_ROOT=/absolute/path/to/qualified-inputs \
  ENGINE_BUNDLE=engine \
  TOKENIZER_CONTRACT=tokenizer-contract.json \
  EXPECTED_ENGINE_TREE_SHA256=EXPECTED_BARE_SHA256 \
  EXPECTED_TOKENIZER_SHA256=EXPECTED_BARE_SHA256 \
  EXPECTED_OUTPUT_SHA256=EXPECTED_BARE_SHA256
```

`make -C addins/tensorrt_llm qualify-evidence` then runs the public
evaluate-to-verify transaction with explicit independent anchors.

The Dockerfile pins its NVIDIA base manifest. Resolve the finished candidate
to an immutable registry digest and qualify it on the intended GPU before use.

## Derive request settings

Build the request-side runtime spec from the exact engine bundle, tokenizer
contract, and pinned runner instead of transcribing engine metadata. The
inspection probes the vendor runner and CUDA runtime, so this Python must run
inside the digest-pinned, network-disabled image built from
[`runtime/Dockerfile`](runtime/Dockerfile), with the target GPU available:

```python
from pathlib import Path
import json

from invarlock_addins.tensorrt_llm.provider import TensorRTLLMProvider
from invarlock_addins.tensorrt_llm.session import TensorRTLLMRuntimeBindings

bindings = TensorRTLLMRuntimeBindings(
    engine_bundle_path=Path("/inputs/engine"),
    tokenizer_contract_path=Path("/inputs/tokenizer-contract.json"),
    runner_executable_path=Path(
        "/opt/invarlock/bin/tensorrt-llm-runner"
    ),
)
spec = TensorRTLLMProvider().inspect_runtime_spec(
    bindings,
    seed=0,
    context_length=2048,
    batch_size=1,
    max_output_tokens=64,
    timeout_seconds=300,
)
print(
    json.dumps(
        {"model_id": spec.model_id, "settings": dict(spec.settings)},
        indent=2,
        sort_keys=True,
    )
)
```

After resolving the built image to an immutable digest, the inspection pattern
is:

```bash
IMAGE=registry.example/invarlock-tensorrt-llm@sha256:PINNED_IMAGE_DIGEST
DIGEST=sha256:PINNED_IMAGE_DIGEST

docker run --rm --network none --gpus all \
  --env INVARLOCK_CONTAINER_EXECUTION=1 \
  --env INVARLOCK_RUNTIME_IMAGE="$IMAGE" \
  --env INVARLOCK_RUNTIME_IMAGE_DIGEST="$DIGEST" \
  --mount type=bind,src="$PWD/inputs",dst=/inputs,readonly \
  --entrypoint python "$IMAGE" /inputs/inspect.py
```

Use `spec.model_id` and `spec.settings` for a request side whose
`runtime.provider` is `tensorrt_llm`. Evaluation requires the same bundle and
support files as root-confined resources plus the authenticated outer container
on a matching CUDA compute capability. Vendor imports remain inside the runner
boundary; they are not dependencies of the core wheel.

## Qualify a candidate image and engine

The package includes a real provider-path canary. Run it inside the candidate
image, on the target GPU, after recording the engine-tree, tokenizer, and
expected-output digests through an independent trust path:

```bash
python -m invarlock_addins.tensorrt_llm.canary \
  --engine-bundle /inputs/engine \
  --tokenizer-contract /inputs/tokenizer-contract.json \
  --runner /opt/invarlock/bin/tensorrt-llm-runner \
  --expected-engine-tree-sha256 EXPECTED_BARE_SHA256 \
  --expected-tokenizer-sha256 EXPECTED_BARE_SHA256 \
  --expected-output-sha256 EXPECTED_BARE_SHA256
```

Set `INVARLOCK_RUNTIME_IMAGE`, `INVARLOCK_RUNTIME_IMAGE_DIGEST`, and
`INVARLOCK_CONTAINER_EXECUTION=1` on the container exactly as in the inspection
example. The canary authenticates the official runner, checks the target CUDA
facts, opens two fresh provider sessions, requires byte-identical output and
evidence records, and returns one compact JSON object with `"ok":true`. It is
candidate qualification, not evidence for a baseline-versus-subject decision;
run `invarlock evaluate` and `invarlock verify` for that decision.

## Runner protocol

`invarlock-tensorrt-llm-runner` is a bounded process protocol used by the
provider. It has exactly two modes:

| Invocation | Input | Output | Exit status |
| --- | --- | --- | --- |
| `--invarlock-runtime-info-v1` | No standard input | One `invarlock/tensorrt-llm-runner-info-v1` JSON line | `0`, or `70` on a closed runtime failure |
| `--invarlock-score-v1` | One canonical JSON request on standard input, at most 1 MiB | One `invarlock/tensorrt-llm-runner-response-v1` JSON line | `0`, or `70` on a closed runtime failure |
| Any other argument shape | None | None | `64` |

The score mode authenticates regular engine and tokenizer files, enforces
closed JSON and resource limits, checks deterministic scalar settings and the
container/network boundary, and rejects unbounded or malformed output. This is
an internal provider transport, not a replacement for the public
`evaluate -> verify -> report` workflow.

## Public Python surface

- `TensorRTLLMProvider` implements runtime-provider ABI `1` and derives a
  complete `ModelRuntimeSpec` through `inspect_runtime_spec`.
- `TensorRTLLMRuntimeBindings` names the engine, tokenizer contract, and runner
  inputs used for inspection and execution.
- `qualify_candidate` and `TensorRTLLMCanaryError` implement candidate-image
  qualification through the real provider path.
- `TensorRTLLMRunnerError` describes a closed runner-protocol failure.

Session and execution modules are implementation details. Applications should
enter through the provider ABI or the documented canary rather than construct a
session directly.
