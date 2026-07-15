# Native Runtime Provider Pair

**Status:** runnable with user-supplied, already qualified native artifacts.

This example performs the complete experimental runtime-provider transaction:

1. build one authenticated behavioral schedule;
2. prepare the exact baseline and subject bindings inside their reviewed images;
3. build one directed `policy-pack-v3`;
4. produce each side in a network-disabled, read-only container; and
5. independently replay the pair and write a positive digest-only receipt.

The baseline and subject can each use `llama_cpp` or `tensorrt_llm`. A mixed
GGUF/TensorRT-LLM pair is valid when both artifacts implement the same reviewed
records and deterministic output contract.

## Prerequisites

- an authenticated checkout of the same InvarLock source tag used to build the
  native runtime images;
- Docker, plus the NVIDIA Container Toolkit for a TensorRT-LLM side;
- immutable local image digests from successful platform qualification;
- one GGUF file or one closed TensorRT-LLM engine bundle per role;
- the exact TensorRT-LLM tokenizer contract for every TensorRT-LLM role; and
- reviewed behavioral records whose expected outputs are appropriate for both
  roles.

Build and qualify images before this example. The qualification targets and
hardware restrictions are documented in
[Runtime Providers](../../../docs/reference/runtime-providers.md#runtime-qualification).
Image qualification is a prerequisite; it is not the pair receipt produced by
this example.

## Prepare Inputs

Create an ignored working area and copy the closed templates:

```bash
mkdir -p artifacts/native-inputs
cp examples/integrations/runtime_providers/native-provider.env.example \
  artifacts/native-inputs/native-provider.env
cp examples/integrations/runtime_providers/llama-cpp-settings.example.json \
  artifacts/native-inputs/baseline-llama-cpp-settings.json
cp examples/integrations/runtime_providers/tensorrt-llm-settings.example.json \
  artifacts/native-inputs/subject-tensorrt-llm-settings.json
cp examples/integrations/runtime_providers/tensorrt-llm-tokenizer-contract.example.json \
  artifacts/native-inputs/subject-tokenizer-contract.json
```

The settings templates deliberately contain zero digests and cannot authorize
a run. Replace the tokenizer placeholder with the reviewed tokenizer JSON, then
derive settings from the exact mounted artifacts and backend. The helper refuses
to replace an existing output.

For a GGUF side, run the helper inside the qualified GGUF image:

```bash
GGUF_IMAGE_DIGEST=sha256:<reviewed-64-hex-digest>
GGUF_MODEL="$(pwd)/artifacts/native-inputs/baseline.gguf"
SETTINGS_DIR="$(pwd)/artifacts/native-inputs"

rm artifacts/native-inputs/baseline-llama-cpp-settings.json
docker run --rm --network none --read-only \
  --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g,mode=1777 \
  --env INVARLOCK_CONTAINER_EXECUTION=1 \
  --env "INVARLOCK_RUNTIME_IMAGE=$GGUF_IMAGE_DIGEST" \
  --env "INVARLOCK_RUNTIME_IMAGE_DIGEST=$GGUF_IMAGE_DIGEST" \
  --mount "type=bind,src=$GGUF_MODEL,dst=/models/model.gguf,readonly" \
  --mount "type=bind,src=$SETTINGS_DIR,dst=/outputs" \
  --entrypoint /usr/local/bin/invarlock "$GGUF_IMAGE_DIGEST" \
  advanced runtime-behavior inspect-inputs \
  --provider llama_cpp \
  --artifact /models/model.gguf \
  --backend-executable /opt/llama.cpp/llama-completion \
  --backend-source /opt/llama.cpp/source/llama.cpp-b10015.tar.gz \
  --context-length 8 \
  --max-output-tokens 1 \
  --timeout-seconds 300 \
  --out /outputs/baseline-llama-cpp-settings.json \
  --json
```

For a TensorRT-LLM side, run the helper on the selected GPU inside its qualified
image. Size the temporary filesystem above the engine bundle plus runtime
scratch requirements:

```bash
TENSORRT_IMAGE_DIGEST=sha256:<reviewed-64-hex-digest>
TENSORRT_ENGINE="$(pwd)/artifacts/native-inputs/subject-engine"
TOKENIZER_CONTRACT="$(pwd)/artifacts/native-inputs/subject-tokenizer-contract.json"
SETTINGS_DIR="$(pwd)/artifacts/native-inputs"

rm artifacts/native-inputs/subject-tensorrt-llm-settings.json
docker run --rm --gpus device=0 --network none --read-only \
  --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=16g,mode=1777 \
  --env FORCE_DETERMINISTIC=1 \
  --env INVARLOCK_CONTAINER_EXECUTION=1 \
  --env "INVARLOCK_RUNTIME_IMAGE=$TENSORRT_IMAGE_DIGEST" \
  --env "INVARLOCK_RUNTIME_IMAGE_DIGEST=$TENSORRT_IMAGE_DIGEST" \
  --env HOME=/tmp/invarlock-home \
  --env XDG_CACHE_HOME=/tmp/invarlock-cache \
  --env HF_HOME=/tmp/invarlock-hf \
  --env FLASHINFER_WORKSPACE_DIR=/tmp/invarlock-flashinfer \
  --mount "type=bind,src=$TENSORRT_ENGINE,dst=/engines/model,readonly" \
  --mount "type=bind,src=$TOKENIZER_CONTRACT,dst=/inputs/tokenizer-contract.json,readonly" \
  --mount "type=bind,src=$SETTINGS_DIR,dst=/outputs" \
  --entrypoint /bin/bash "$TENSORRT_IMAGE_DIGEST" \
  -c 'exec "$@"' -- /opt/invarlock/cli-venv/bin/invarlock \
  advanced runtime-behavior inspect-inputs \
  --provider tensorrt_llm \
  --artifact /engines/model \
  --backend-executable /opt/invarlock/bin/tensorrt-llm-runner \
  --tokenizer-contract /inputs/tokenizer-contract.json \
  --context-length 8 \
  --max-output-tokens 1 \
  --timeout-seconds 300 \
  --out /outputs/subject-tensorrt-llm-settings.json \
  --json
```

Repeat the matching derivation command for the other role when both sides use
the same provider. Review the resulting settings and keep them with the exact
artifacts they describe.

The included five-record material is a bounded one-token transaction fixture
for `TinyLlama/TinyLlama-1.1B-Chat-v1.0` revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Its checked-in expected strings are
example acceptance inputs, not retained proof that a GGUF/TensorRT-LLM pair
satisfies them. No five-record pair receipt is published with this fixture. A
live run must produce and verify both sides before it supports even the narrow
schedule-level claim. For a release decision, replace the fixture with a
reviewer-owned deterministic schedule selected before the subject result is
observed.

## Run The Pair

Edit `artifacts/native-inputs/native-provider.env` so every path names the exact
input for that role and every image value is an immutable digest. Then run:

```bash
set -a
source artifacts/native-inputs/native-provider.env
set +a
bash examples/integrations/runtime_providers/run_native_pair.sh
```

The wrapper rejects template digests, symlinked inputs, unsupported provider
names, unsafe mount strings, non-canonical TensorRT-LLM GPU selectors, and an
existing work directory before launching a container. Use
`device=<nonnegative-index>` or `device=<GPU-UUID>` for each TensorRT-LLM role.
Containers run as the invoking uid and gid so their `0600` outputs remain
readable by the host-side policy and verification commands. Both provider sides
run with networking disabled, a read-only root, all capabilities dropped, and
no-new-privileges. The selected output directory is the only persistent writable
bind mount; `/tmp` is a bounded, non-executable tmpfs for runtime scratch.

After every command, the wrapper requires the expected nonempty readable file
or complete six-file side directory before continuing. A command that exits
successfully without publishing its output cannot reach the final success
message.

Successful output is written below the new `INVARLOCK_NATIVE_WORK_DIR`:

```text
control/behavioral-schedule.json
control/acceptance-policy-pack.json
bindings/baseline-binding.json
bindings/subject-binding.json
sides/baseline/
sides/subject/
paired-receipt.json
```

The final command succeeds only when both immutable side bundles satisfy their
directed bindings and policy. A failed side or pair exits nonzero and does not
create a positive pair receipt.

## Claim Boundary

The receipt supports the policy-scoped `exact_match` claim for the authenticated
records. It does not establish weight, numerical, performance, export, or
general-quality equivalence, and it is not remote attestation. See the
[native runtime operator guide](../../../docs/user-guide/native-runtime-providers.md)
before using the receipt in a release decision.
