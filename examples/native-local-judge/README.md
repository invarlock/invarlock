# Judge frozen answers with a local model

This starter runs `hf_transformers` or `llama_cpp` as the judge through
`runtime-provider-judge`. It authenticates local model files and the runtime,
grades frozen answers, and publishes evidence for offline verification and
reporting. It uses no hosted judge SDK, API key or HTTP endpoint.

The bundled answer pair is invented fixture data with one independent unit.
It demonstrates setup and cannot satisfy the example's minimum of 20 units.
It is not a retained real-model qualification or evidence of judge accuracy.
Replace the answers, rubric and sampling assignments with reviewed data for an
actual decision. Repeated ratings do not create additional independent units.
For a retained, powered campaign whose two declared policies pass, see the
[K2 Luna held-out reference](../judge-measurements/references/k2-32b-luna-xhigh-heldout/README.md).

Choose the input format before preparing the request. The default sends the
complete canonical judge-request JSON. For a model that expects ChatML, set
`plan.prompt.runtime_format` to `chatml-v1` in your copied
`starter/recipe-template.json`. This renders the declared messages with explicit
ChatML role delimiters. The runtime does not silently choose a tokenizer template.
The format is bound into the plan and checked during offline replay.

Choose a model that follows the selected format and returns exactly
`{"rating":"correct"}` or `{"rating":"incorrect"}`, as instructed by this
starter's system prompt. Other output is retained as an invalid rating. A format
change requires a new workspace; it does not replace unsuccessful earlier runs.

## Prepare the runtime and local files

Use a matching core wheel, example checkout and operator-verified runtime image.
HF execution dependencies or the pinned llama.cpp backend belong in that image;
the host does not need `invarlock[judge]`. Follow the
[runtime provider guide](../../docs/user-guide/runtime-providers.md) to build and
pin the selected image and inspect its resources. Fetch model files during
setup, before entering the offline execution boundary. No command below
downloads models or dependencies.

Choose a new workspace and copy this starter into it:

```bash
LOCAL_JUDGE_WORKSPACE="$PWD/local-judge-workspace"
mkdir -m 700 "$LOCAL_JUDGE_WORKSPACE"
cp -R examples/native-local-judge "$LOCAL_JUDGE_WORKSPACE/starter"
mkdir "$LOCAL_JUDGE_WORKSPACE/artifacts"
```

Materialize either a complete safetensors checkpoint at `artifacts/judge`, or a
GGUF file at `artifacts/judge.gguf`. Use real files, not cache symlinks. For GGUF,
also place the authenticated backend executable and matching source archive at
`artifacts/backend/llama-completion` and `artifacts/backend/llama.cpp-source.tar`.
Keep caller-owned evidence and verifier signing keys outside this workspace.

The local collector runs only when InvarLock itself is inside the strict offline
container. Running the host CLI with `--runtime-profile` does not launch a third
judge container. The native baseline/subject image overrides do not select a
judge image. This starter uses frozen-answer v3 requests to make that boundary
explicit.

Native v1 `mode: run` requests with an explicit local judge model use the same
boundary without nested Docker. The CLI runs baseline, subject and judge in this
one container and requires the same declared `INVARLOCK_RUNTIME_IMAGE_DIGEST`
for all three. It rejects explicit OCI engine, worker CPU/memory/user and
entrypoint controls on that inline path. Native v1 still requires a signing key
inside the trusted process; use the frozen-answer `judge_import` route when the
signer must remain on a separate host.
The CLI checks container intent, a kernel-visible container marker and disabled
runtime opt-ins. It does not inspect the engine's network, filesystem or
capability settings or independently derive the current image digest. Apply the
`docker run` isolation below and verify the image digest before launch.

Start the selected, already-built CPU image. Replace the image and signing-key
paths with independently pinned local resources:

```bash
LOCAL_JUDGE_IMAGE='sha256:REPLACE_WITH_ENGINE_REPORTED_IMAGE_DIGEST'
LOCAL_JUDGE_DIGEST="$LOCAL_JUDGE_IMAGE"
LOCAL_JUDGE_SIGNING_KEY=/secure/keys/evidence-signer.pem

docker run --rm -it --pull=never --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --cpus 4 --memory 16g \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --mount "type=bind,src=$LOCAL_JUDGE_WORKSPACE,dst=/work" \
  --mount "type=bind,src=$LOCAL_JUDGE_WORKSPACE/artifacts,dst=/work/artifacts,readonly" \
  --mount "type=bind,src=$LOCAL_JUDGE_SIGNING_KEY,dst=/keys/evidence.pem,readonly" \
  --env HOME=/tmp --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
  --env INVARLOCK_CONTAINER_EXECUTION=1 \
  --env "INVARLOCK_RUNTIME_IMAGE=$LOCAL_JUDGE_IMAGE" \
  --env "INVARLOCK_RUNTIME_IMAGE_DIGEST=$LOCAL_JUDGE_DIGEST" \
  --env INVARLOCK_RUNTIME_DEVICE=cpu \
  --env INVARLOCK_JUDGE_RUNTIME_DEVICE=cpu \
  --workdir /work --entrypoint /bin/sh "$LOCAL_JUDGE_IMAGE"
```

Run as a non-root user. Select memory and CPU limits appropriate for your model.
For an approved HF CUDA image, expose the intended GPU through the container
engine and set both device variables to `cuda`. These direct-provider resource
variables accept `cpu` or `cuda`, not an indexed value such as `cuda:0`.
A CPU image cannot become a CUDA runtime through a device flag.
The current llama.cpp profile uses CPU execution. Keep all network,
remote-code and third-party-provider permission switches disabled.
`INVARLOCK_JUDGE_RUNTIME_DEVICE` selects the judge device and otherwise defaults
to `INVARLOCK_RUNTIME_DEVICE`; baseline and subject device overrides do not
select the judge device.

The signing key is available to this trusted runtime process through its
read-only mount. It is outside the published workspace, but this in-container
collector does not provide the native host orchestrator's worker/key separation.
For this frozen-answer v3 starter, if you require that separation, omit the key
mount, collect with `--unsigned`, then transfer the unchanged frozen runs, plan,
measurements and analysis policy to a trusted host and sign through the existing
[`judge_import` workflow](../judge-measurements/README.md). Verification still
checks the retained local runtime evidence; no second inference is required.

## Inspect one judge model

Run exactly one of the following snippets inside that container. Both write a
new `model.json`, the same artifact/runtime shape used by native requests.

For an HF checkpoint:

```bash
python - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer
from invarlock.engine import checkpoint_tree_sha256, hf_tokenizer_contract_sha256

path = Path("artifacts/judge")
tokenizer = AutoTokenizer.from_pretrained(
    path, local_files_only=True, trust_remote_code=False
)
model = {
    "artifact": {"path": str(path), "model_id": "local-judge", "locator": "local://judge"},
    "runtime": {"provider": "hf_transformers", "settings": {
        "checkpoint_tree_sha256": checkpoint_tree_sha256(path).removeprefix("sha256:"),
        "tokenizer_metadata_sha256": hf_tokenizer_contract_sha256(tokenizer),
        "offline": True, "seed": 0, "batch_size": 1,
        "context_length": 4096, "max_output_tokens": 128, "timeout_seconds": 300,
    }},
}
with open("model.json", "x", encoding="utf-8") as stream:
    json.dump(model, stream, indent=2, allow_nan=False)
PY
```

This identity binds the exact local checkpoint tree and tokenizer. If you also
declare an immutable upstream revision, obtain it from your reviewed model
inventory and add it before preparing the plan.

For a GGUF artifact, inspect the actual backend and model together:

```bash
export INVARLOCK_GGUF_RESOURCE_ROOT=/work/artifacts
export INVARLOCK_GGUF_BACKEND_EXECUTABLE=backend/llama-completion
export INVARLOCK_GGUF_BACKEND_SOURCE=backend/llama.cpp-source.tar
python - <<'PY'
import json
from pathlib import Path
from invarlock.runtime_providers.llama_cpp import LlamaCppProvider
from invarlock.runtime_providers.llama_cpp_session import LlamaCppRuntimeBindings

path = Path("artifacts/judge.gguf")
spec = LlamaCppProvider().inspect_runtime_spec(
    LlamaCppRuntimeBindings(
        gguf_path=path,
        executable_path=Path("artifacts/backend/llama-completion"),
        source_archive_path=Path("artifacts/backend/llama.cpp-source.tar"),
    ),
    seed=0, context_length=4096, batch_size=1, cpu_threads=4,
    prompt_batch_size=512, prompt_microbatch_size=512,
    max_output_tokens=128, timeout_seconds=300,
)
model = {
    "artifact": {"path": str(path), "model_id": spec.model_id, "locator": "local://judge.gguf"},
    "runtime": {"provider": spec.provider_name, "settings": dict(spec.settings)},
}
with open("model.json", "x", encoding="utf-8") as stream:
    json.dump(model, stream, indent=2, allow_nan=False)
PY
```

Inspection authenticates the backend and reads its version; it does not generate
ratings. Keep the GGUF resource variables set during preflight and evaluation.

## Freeze, preflight and evaluate

Still inside the same container:

```bash
python starter/prepare.py --workspace /work --model /work/model.json
invarlock evaluate request.json --signing-key /keys/evidence.pem --preflight --json
invarlock evaluate request.json --signing-key /keys/evidence.pem --json
```

The preparation helper authenticates the model bytes, computes
`artifact_identity_sha256` over the complete canonical artifact identity, and
places that bare SHA-256 value in the plan's `local_weights` identity. It freezes
the answer bindings and analysis policy. It makes no inference calls and
refuses existing generated destinations. Read the generated plan before
executing it; do not merely copy a weight-file hash into the judge identity.
If preparation is interrupted, preserve its partial output and prepare a new
workspace. The helper does not overwrite or silently complete a partial setup.

The generated collection configuration is:

```json
{
  "profile": "runtime-provider-text-frozen-answer-v1",
  "max_calls": 2,
  "max_output_tokens": 256
}
```

Those bounds cover two 128-token reservations for the one-case pair. When you
replace the starter's frozen runs and sampling assignments, preparation derives
the complete reservation from cases, sides, repetitions and the output bound.
The model
spec binds its context, seed and execution timeout. The hosted collector's
`invocation_timeout_seconds` option is not accepted for this profile. The
profile fixes temperature to `0`, top-p
to `1`, reasoning effort to `null`, and one uncached attempt with no retries or
tools. No local token use is presented as a hosted invoice or dollar estimate.

Retained measurements are bounded to 384 MiB, including duplicated observations
and outputs. Before each inference, collection counts the exact retained bytes
and reserves another 48 MiB for the next source plus a 1 MiB envelope. This
reserves only the next call, so short ratings can support larger schedules.
Schedules whose full worst-case size exceeds the limit require durable
checkpoints; this CLI example supplies one through its workspace. If the next
reservation cannot fit, collection stops before admitting another inference,
preserves completed shards, and does not publish an incomplete evidence pack.

The collector retains a separate admission and result for each trial. Completed
trials can be reused for the unchanged plan. An admitted trial without its result
is ambiguous and cannot silently run again. Preserve that workspace; investigate
the original execution before deliberately preparing a new transaction. Changes
to the artifact, runtime, frozen answers, rubric or collection configuration
require a new workspace. The hosted collector's explicit batch-stop control is
not available in this profile.

Preflight is execution-free. Evaluation runs the local judge and publishes its
retained provider observations and signed evidence. With the bundled fixture,
expect `insufficient_evidence`; successful publication is not policy acceptance.
Malformed returned ratings and per-record provider failures remain visible
without fabricated scores. An aborted batch retains its admission instead of
claiming completed evidence.

## Verify and report without the model

Exit the container. Use a separate core-only recipient environment with a
recipient-owned trust policy and verifier key:

```bash
invarlock verify "$LOCAL_JUDGE_WORKSPACE/evidence" \
  --trust-profile /secure/trust/local-judge-recipient.json \
  --receipt "$LOCAL_JUDGE_WORKSPACE/verification.receipt.json" \
  --verifier-signing-key /secure/keys/verifier.pem \
  --verifier-identity local-judge-recipient --json
invarlock report "$LOCAL_JUDGE_WORKSPACE/evidence" \
  --html "$LOCAL_JUDGE_WORKSPACE/report.html" --json
```

Prepare the [judge recipient policy](../../docs/reference/judge-measurements.md#replay-authentication-and-acceptance)
through an independent approval process. It binds the trusted evidence signer,
evaluated subject, exact frozen runs, local judge plan, retained measurements and
analysis. Do not derive authorization solely from the submitted evidence. A
verified but unaccepted result returns status 7 and can still be reported.

The retained `retained-runtime-provider-judge-v1` source carries the
local artifact identity, runtime observation, provider receipt, ordered schedule,
requests and responses. Verification checks those bindings and recomputes the
analysis offline, without model files, inference dependencies or API access. It
does not independently attest the original host or prove that repeated inference
on a different runtime would return identical bytes.

An OpenAI-compatible server on localhost follows a service contract. Its endpoint
or model label cannot substitute for these complete local artifact and runtime
bindings. Conversely, this example does not qualify arbitrary local HTTP servers,
chat templates, vision-text judges or TensorRT-LLM judge execution.
