# Runtime providers

Runtime-provider ABI `1` isolates backend-specific artifact identity and
execution from the evidence transaction. Hugging Face Transformers is built
into the core distribution. GGUF through `llama.cpp`, TensorRT-LLM, and Hugging
Face vision-text are first-party optional add-ins.

!!! info "Reference"

    - **Surface:** Runtime-provider ABI `1`, typed identities, settings, resources, discovery, and conformance
    - **Stability:** Providers must match versioned ABI `1` exactly; first-party settings and capability declarations are closed
    - **Use this page when:** Selecting, configuring, implementing, or validating a runtime provider

## Provider protocol

A provider implements `RuntimeProvider` with an exact `name` and
`abi_version`, then supplies:

```python
def validate_config(spec: ModelRuntimeSpec) -> None: ...
def capabilities() -> RuntimeProviderCapabilities: ...
def identify_artifact(spec: ModelRuntimeSpec) -> ModelArtifactIdentity: ...
def authenticate_artifact(
    spec: ModelRuntimeSpec,
    artifact_path: Path,
) -> ModelArtifactIdentity: ...
def prepare_execution(
    spec: ModelRuntimeSpec,
    resources: RuntimeArtifactResources,
) -> RuntimeExecutionContext: ...
def open(spec: ModelRuntimeSpec, context: RuntimeExecutionContext) -> RuntimeSession: ...
```

The opened session provides `score(batch)`, `runtime_receipt()`, and `close()`.
Provider state is opaque. Backend imports must be lazy so discovery and contract
inspection remain usable without loading a model runtime.

| Method | Input | Required result or effect |
| --- | --- | --- |
| `validate_config` | Closed `ModelRuntimeSpec` | Reject unknown, missing, malformed, or inconsistent settings |
| `capabilities` | None | Return closed artifact, task, metric, and execution-mode declarations |
| `identify_artifact` | Validated spec | Return a typed portable identity reproducible from authenticated inputs |
| `authenticate_artifact` | Validated spec and local artifact path | Read the local bytes without loading the backend, require their canonical identity to equal the spec, and return that identity |
| `prepare_execution` | Spec and caller-owned resources | Authenticate local resources and return an immutable execution context |
| `open` | Spec and context | Open one session without changing the authenticated identity |
| `session.score` | Ordered `EvaluationBatch` | Return one ordered `ScoringObservation` for that batch |
| `session.runtime_receipt` | Completed scoring state | Return a receipt bound to the latest complete observation |
| `session.close` | None | Release resources; repeated close calls must be safe |

## Capabilities and current scope

Capabilities are closed over:

- artifact formats: `hf_snapshot`, `gguf`, `tensorrt_llm_engine`;
- canonical tasks declared by each provider, currently `text_causal` and
  `vision_text_generation` across shipped integrations;
- metrics: `exact_match` and `normalized_nll_per_utf8_byte`; and
- execution modes: `in_process`, `local_process`, `container`.

The request loader rejects a built-in metric the selected provider does not
declare. A scorer extension is not a provider metric: the provider collects
authenticated output text through its declared exact-match path, and an
explicitly authorized verifier-replay scorer derives unit-interval values from
those text facts after collection.
Strict evidence execution further requires a real offline container and
`batch_size=1`, regardless of a provider's broader capability declaration.

| Shipped provider | Distribution | Artifact | Metric | Declared mode | Strict device |
| --- | --- | --- | --- | --- | --- |
| `hf_transformers` | `invarlock` | `hf_snapshot` | `exact_match`, `normalized_nll_per_utf8_byte` | `in_process` | `cpu` or `cuda`, inside its authenticated side worker |
| `llama_cpp` | `invarlock-runtime-gguf` | `gguf` | `exact_match` | `container`, `local_process` | `cpu`; strict evidence uses an authenticated side worker |
| `tensorrt_llm` | `invarlock-runtime-tensorrt-llm` | `tensorrt_llm_engine` | `exact_match` | `container` | `cuda` in an authenticated side worker |
| `hf_vision_text` | `invarlock-runtime-hf-vision-text` | `hf_snapshot` | `exact_match` | `container` | `cuda` in an authenticated side worker |

The built-in HF scorer computes log-probability facts by teacher-forcing a
target token sequence that must decode as the exact schedule `expected_output`
when appended to the prompt tokens. Boundary-unstable encodings fail closed.
It binds the target token count and verifier-checkable UTF-8 byte count in each
scoring record. Normalized NLL divides by expected-output bytes, so it remains
comparable across tokenizers. It measures teacher-forced expected-continuation
likelihood regression rather than general model quality. When authenticated
tokenizer digests and paired token counts match, the verifier may render a
token-weighted perplexity ratio as an interpretation without giving it policy
authority. The GGUF add-in
does not declare normalized NLL: its authenticated execution surface is
the raw `llama-completion` executable, which emits generated bytes but no
deterministic per-target-token log probabilities. TensorRT-LLM and Hugging Face
vision-text likewise remain exact-match only. A provider may not emit a metric it did not advertise, and
request loading rejects that mismatch before execution.

## Artifact identity

Providers return one typed identity:

- Hugging Face binds a model ID, immutable revision and/or checkpoint tree,
  plus tokenizer metadata;
- GGUF binds the regular file bytes, GGUF metadata, tensor inventory, and
  tokenizer metadata; and
- TensorRT-LLM binds the engine tree, file inventory, builder configuration,
  tokenizer metadata, engine metadata, and target compute capability.

The identity must be reproducible from the request-side `ModelRuntimeSpec` and
the exact artifact. The transaction compares imported identities with a fresh
provider derivation.

| Artifact format | Identity fields beyond `format` and `artifact_format` |
| --- | --- |
| Hugging Face snapshot | `model_id`, `immutable_revision` and/or `checkpoint_tree_sha256`, `tokenizer_metadata_sha256` |
| GGUF | `artifact_name`, `sha256`, `byte_length`, `gguf_metadata_sha256`, `tensor_inventory_sha256`, `tokenizer_metadata_sha256` |
| TensorRT-LLM engine | `bundle_name`, `engine_bundle_tree_sha256`, `file_inventory_sha256`, `builder_config_sha256`, `tokenizer_metadata_sha256`, `engine_metadata_sha256`, `target_compute_capability` |

Identity fields describe authenticated bytes, not merely a user-facing model
name. GGUF and TensorRT-LLM use privacy-safe digest-derived model IDs in their
runtime specs so local filenames and host paths cannot enter public evidence.

## First-party settings

Settings are JSON scalars in `request.yaml`. Unknown keys fail closed.

### `hf_transformers`

| Setting | Requirement | Meaning |
| --- | --- | --- |
| `immutable_revision` | Identity: revision or checkpoint tree required | Immutable upstream revision |
| `checkpoint_tree_sha256` | Required for local execution | Authenticated checkpoint-directory tree digest |
| `tokenizer_metadata_sha256` | Required | Authenticated tokenizer contract digest |
| `batch_size` | Required for strict receipt; must be `1` | Scoring batch size |
| `context_length` | Required, positive integer | Maximum input context |
| `max_output_tokens` | Required, positive integer | Maximum generated output |
| `offline` | Required and `true` | Backend offline intent |
| `seed` | Required, non-negative integer | Deterministic seed |
| `timeout_seconds` | Required, positive integer | Per-call timeout |

The provider loads only local files, disables remote code, requires
SafeTensors-compatible loading, and authenticates the checkpoint tree and
tokenizer contract against the declared digests. Hugging Face documents the
underlying library's [offline-mode
controls](https://huggingface.co/docs/transformers/installation#offline-mode);
InvarLock additionally binds its own strict runtime and receipt conditions.

### `llama_cpp`

Every key is required: `artifact_byte_length`, `artifact_sha256`,
`backend_binary_sha256`, `backend_source_sha256`, `backend_version`,
`batch_size`, `context_length`, `gguf_metadata_sha256`, `max_output_tokens`,
`seed`, `tensor_inventory_sha256`, `timeout_seconds`, and
`tokenizer_metadata_sha256`. Digest values are lowercase bare SHA-256 values.
The provider authenticates the GGUF bytes, typed metadata and tensor inventory,
tokenizer metadata, and the pinned `llama.cpp` executable and source tree.
Independent verification also requires the normalized-request digest from the
reviewed preflight result. It reconciles every GGUF identity, backend, and shared
execution setting in that request with the authenticated provider receipt.

### `tensorrt_llm`

Every key is required: `backend_build_sha256`, `backend_version`, `batch_size`,
`builder_config_sha256`, `context_length`, `engine_bundle_tree_sha256`,
`engine_metadata_sha256`, `file_inventory_sha256`, `max_output_tokens`,
`runner_binary_sha256`, `seed`, `target_compute_capability`, `timeout_seconds`,
and `tokenizer_metadata_sha256`. The target compute capability uses
`major.minor` notation and is checked against the observed GPU. The provider
binds the engine tree, build configuration, file inventory, runner, backend,
tokenizer, engine metadata, and GPU target. See NVIDIA's [TensorRT-LLM
documentation](https://nvidia.github.io/TensorRT-LLM/) for backend concepts;
the InvarLock add-in defines the evidence bindings above.

## Scoring obligations

For every schedule record, a provider emits one record with the same
`record_id` and `input_sha256`, in exactly the same order. Successful records
carry output facts, log-probability facts, or both. Failed records carry an
error code and no measured output facts; canonical evidence rejects them.

| Scoring-record field group | `ok` record | `error` record |
| --- | --- | --- |
| `record_id`, `input_sha256`, `status` | Required | Required |
| Output text and output digest | Required for `exact_match` and scorer-extension collection | Forbidden |
| `logprob_sum`, positive `token_count`, positive `utf8_byte_count` | Required for normalized NLL | Forbidden |
| `error_code` | Forbidden | Required |

The exact schema determines the allowed combinations. A batch must contain
exactly one record for every schedule entry, with no reordering, duplication,
omission, or replacement by positional IDs.

The observation binds the provider name, artifact-identity digest, schedule
digest, ordered records, and an aggregate-source digest computed from those
records. Providers report facts; InvarLock owns aggregate scores and policy
arithmetic.

Provider discovery does not authorize scorer extensions. An embedding that
selects `comparison.scorer_extension` must supply an explicit
`ScorerExtensionRegistry` to both evaluation and verification. That registry
may contain caller-injected scorers or may explicitly enable discovery from
the separate `invarlock.scorers` entry-point group.

The runtime receipt must agree with the provider's capabilities and bind:

- plugin distribution and version;
- backend identity with at least one source, binary, or build digest;
- artifact identity;
- deterministic execution settings;
- device facts;
- the outer image digest; and
- the scoring-observation digest.

## Caller-owned resources

Requests cannot grant host permissions or name executable support resources.
For each side, the caller supplies an absolute, non-symbolic resource root, a
primary artifact, closed support-resource names, device kind, and container
image digest through `RuntimeArtifactResources`. Each access repeats
component-by-component no-follow validation.

The public CLI environment resolver uses the variables listed in
[Environment variables](environment.md). A programmatic caller can supply its
own resource resolver to `evaluate_request_file`.

## Per-side OCI execution

The host owns the canonical schedule and launches one independently configured
worker for baseline and one for subject. A shared image, digest, device, and
entrypoint profile can provide defaults; every value also has a per-side
override. The selected image reference and digest must agree before launch.

Each worker receives only:

- its read-only job and canonical schedule;
- its read-only primary artifact and closed support resources;
- an isolated writable side-output directory; and
- the device mapping selected for that side.

The evidence-signing and verifier keys remain on the host. After both workers
exit successfully, the host loads exactly the six expected side files,
reproduces the artifact and runtime bindings, derives paired records, and signs
the closed evidence bundle.

CPU workers may run concurrently. Workers on two different explicit CUDA
indexes, such as `cuda:0` and `cuda:1`, may also run concurrently. Generic CUDA,
a shared explicit index, and CPU/CUDA pairs run sequentially. The selected CUDA
device only exposes hardware; the image must already contain a CUDA-capable
runtime. See [Environment variables](environment.md#canonical-hugging-face-runtime-images)
for the separate canonical CPU and x86_64 CUDA image targets.

The host applies the same caller-owned `OciWorkerLimits` to each side: a
positive CPU decimal, an integer MiB memory ceiling, and a numeric non-root
`UID:GID`. Defaults are four CPUs, 65536 MiB, and `65532:65532`. The submitted
request cannot alter them. A worker also receives a finite outer deadline
derived from its validated per-record `timeout_seconds` and the authenticated
schedule length, capped at 24 hours; expiry stops or kills the exact container
before the side fails closed.

## Discovery and authorization

Providers register in the `invarlock.runtime_providers` entry-point group. The
registry validates entry-point name, distribution identity, module/class path,
module-declared ABI, instance ABI, and provider name before use.

Python entry-point discovery follows the [PyPA entry-points
specification](https://packaging.python.org/en/latest/specifications/entry-points/).
For example:

```toml
[project.entry-points."invarlock.runtime_providers"]
example_runtime = "example_invarlock.provider:ExampleProvider"
```

Entry-point metadata belongs to an installed distribution. Making a source
tree importable with `PYTHONPATH` does not register a provider and cannot
satisfy this authorization boundary. The interpreter that performs provider
discovery must contain the matching-version core distribution and the exact
add-in wheel selected for the transaction.

The core provider and three named first-party add-ins are recognized without
enabling arbitrary plugins. Other installed entry points are ignored unless
`INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS` is explicitly enabled. Strict evidence
execution records that setting and requires it to be false, so enabling
arbitrary plugins is not compatible with the strict evidence path.

## First-party add-ins

Install the exact GGUF wheel into the qualification host interpreter and check
discovery through that interpreter:

```bash
"$PYTHON" -m pip install /wheelhouse/invarlock_runtime_gguf-EXACT.whl
"$PYTHON" -m invarlock_addins.gguf.conformance
```

Install and check the exact TensorRT-LLM wheel the same way:

```bash
"$PYTHON" -m pip install /wheelhouse/invarlock_runtime_tensorrt_llm-EXACT.whl
"$PYTHON" -m invarlock_addins.tensorrt_llm.conformance
```

Install and check the exact Hugging Face vision-text base wheel. Its declared
Pillow dependency supports host preflight; the inference stack remains in the
runtime image:

```bash
"$PYTHON" -m pip install \
  /wheelhouse/Pillow-EXACT.whl \
  /wheelhouse/invarlock_runtime_hf_vision_text-EXACT.whl
"$PYTHON" -m invarlock_addins.multimodal.conformance
```

Each command returns a JSON object whose `ok` value must be true and whose
provider and ABI must match the installed core. Conformance checks packaging,
entry-point identity, protocol structure, and lightweight contract behavior; it
does not execute a representative model or certify GPU compatibility.

Artifact inspection for either native add-in executes backend code. Run it in
the exact digest-pinned, network-disabled runtime image with read-only artifact
mounts. TensorRT-LLM inspection also requires the target GPU. The add-in READMEs
contain the binding constructors and inspection examples.

The conformance output is one JSON object:

```json
{
  "abi_version": "1",
  "errors": [],
  "ok": true,
  "provider": "llama_cpp"
}
```

`ok: true` proves lightweight packaging, discovery, and protocol compatibility.
It does not prove that a model artifact loads, that native binaries match an
image, or that a representative score succeeds. Follow it with a digest-pinned
runtime canary for the exact artifact/backend/device combination.

### Candidate wheel manifest

Maintained qualification executes caller-selected wheel artifacts, not an
arbitrary first-party installation already visible to the host interpreter.
Build the coordinated distributions, then use the maintained helper to create
one no-clobber manifest. For example, a GGUF transaction authenticates the core
and GGUF wheels together:

```bash
make dist-check
install -d -m 700 qualification
python scripts/qualification_candidate_wheels.py \
  --wheel dist/invarlock-*.whl \
  --wheel dist/addins/invarlock_runtime_gguf-*.whl \
  --output "$PWD/qualification/gguf-candidate-wheels.json"
export CANDIDATE_WHEEL_MANIFEST="$PWD/qualification/gguf-candidate-wheels.json"
```

The helper accepts one `--wheel` per artifact, resolves only real `.whl` files,
captures their SHA-256 digests, and creates a new
`invarlock/qualification-candidate-wheels-v1` document with restrictive file
permissions. The driver then requires the manifest to contain the core wheel
and every maintained add-in used by the request, with unique distributions, a
single version, and contents matching the authenticated source archive. A
built-in Hugging Face transaction needs only the core wheel; GGUF,
TensorRT-LLM, and vision-text transactions also need their selected add-in
wheel. Recreate the manifest whenever any wheel changes, and pass the same
`CANDIDATE_WHEEL_MANIFEST` to canary, readiness, and evidence qualification.

`PYTHON` owns the third-party dependency environment used by qualification.
It must provide the dependencies declared by the candidate wheels, such as
cryptography and JSON Schema support from the core or Pillow for vision-text
host preflight. It does not select the first-party candidate. The driver
extracts the authenticated wheels into a private site, places that site before
the interpreter's dependency paths, and probes the expected distribution and
provider entry-point metadata before running the public transaction. An older
installed InvarLock or maintained add-in therefore cannot substitute for a
manifest wheel. Model-runtime dependencies such as Torch, Transformers, or
TensorRT-LLM remain inside the digest-pinned runtime image.

The GGUF package documents a tiny `evaluate -> verify` canary using a real GGUF
artifact and `llama.cpp`. The Hugging Face vision-text package documents a
prompt-and-image `evaluate -> verify -> report` qualification using a local
checkpoint and authenticated content store. Its host preflight uses the same
frozen caller-owned side resources as execution and authenticates every
schedule-selected media object before any worker or GPU starts; workers repeat
the check before model preparation. The TensorRT-LLM package ships
`python -m invarlock_addins.tensorrt_llm.canary`, which checks the authenticated
engine bundle, exact runner protocol, selected GPU, image digest, and one
representative request. Run each qualification only inside its intended
digest-pinned runtime. None of these commands supplies or qualifies a
production model fixture.

Maintained qualification adds a signed image prerequisite around those
provider checks. Run `runtime-qualification-canary` at the repository root, or
the provider's `qualify-canary` target, once for a small representative request
through one exact image digest. Retain its evidence, signed verification
receipt, and verifier-owned trust profile. Subsequent readiness and evidence
targets require those paths as `CANARY_EVIDENCE`, `CANARY_RECEIPT`, and
`CANARY_TRUST_PROFILE`; they strictly reverify the canary and require its
runtime-image digest to equal the image being qualified. The retained canary
may be reused while that exact digest remains unchanged.

This signed canary is a real end-to-end transaction. By contrast, readiness is
execution-free and starts no model worker. The sequence prevents many jobs
from being launched through an image that cannot complete one representative
strict transaction. It does not prove that another model loads, fits device
memory, satisfies backend-specific compatibility, or completes its own run.

The maintained qualification targets delegate to one repository driver rather
than assembling separate shell transactions. The driver authenticates the Git
source archive against the execution files in the named local Git commit,
requires matching source labels on the runtime
image, executes every child from a private archive-backed source snapshot and
empty working directory, matches the caller-digested candidate wheels to that
source, and executes their core and maintained add-in code through an isolated
interpreter bootstrap. The qualification summary binds the candidate manifest,
candidate wheel digests, and selected interpreter digest in addition to the
normalized request digest returned by preflight.

Qualification assumes a controlled host: the selected Python dependency
closure, container daemon, kernel, driver, and device stack remain trusted
computing base. Release qualification therefore uses a fresh hash-locked Linux
environment, the real container journey, and a signed canary for each claimed
provider/task class. Stronger host claims require an independent rerun or
measured-execution attestation.

Completion also requires strict pack verification plus an independent check of
the signed receipt and the renderer-observed pack identity. Readiness mode
performs the same source, image, request, trust, output, provider, and
runtime-availability checks without starting a model worker, after reverifying
the required signed canary. Create the exact archive with
`scripts/qualification_source.py create`; maintained image builds reject absent
or malformed source bindings before invoking Docker or Podman.

For these targets, `QUALIFICATION_DRIVER_PYTHON` launches the standard-library
driver while `PYTHON` supplies the dependency paths used by provider discovery,
preflight, and the public CLI children. First-party distribution and
entry-point metadata come from `CANDIDATE_WHEEL_MANIFEST`, not from
`PYTHONPATH` or an installed core/add-in package. The vision-text host
environment additionally needs Pillow; its Torch and Transformers inference
dependencies remain in the runtime image.

The root and first-party add-in wrappers forward one shared resource-control
surface: `QUALIFICATION_DEVICE`, `QUALIFICATION_CPUS`,
`QUALIFICATION_MEMORY_MIB`, and `QUALIFICATION_USER`. The root wrapper requires
an explicit device; add-in wrappers use provider-appropriate defaults when it
is omitted. Keep the resolved image and digest plus all supplied resource
controls identical across signed canary, readiness, and evidence.

The TensorRT-LLM runner protocol accepts one canonical JSON request on standard
input and returns one canonical JSON response on standard output. Inspection
and evaluation are separate modes, nonzero exit is failure, unknown response
fields fail closed, and configured input/output byte and timeout bounds apply.
The add-in README is the normative operator example for those bounds.

## Authoring an ABI provider

Third-party authors implement the protocol against the stable ABI module, keep
backend imports behind `prepare_execution` or `open`, and publish one entry
point. This minimal shape illustrates the boundary; it omits the required typed
constructors and validation logic:

```python
from invarlock.core.runtime_provider import INVARLOCK_RUNTIME_PROVIDER_ABI


class ExampleProvider:
    name = "example_runtime"
    abi_version = INVARLOCK_RUNTIME_PROVIDER_ABI

    def validate_config(self, spec):
        ...

    def capabilities(self):
        ...

    def identify_artifact(self, spec):
        ...

    def authenticate_artifact(self, spec, artifact_path):
        ...

    def prepare_execution(self, spec, resources):
        ...

    def open(self, spec, context):
        ...
```

Use the exported typed value classes rather than free-form dictionaries.
Provider authors own their conformance harnesses and can use the first-party
add-in suites as examples.
Tests should cover unknown settings, byte-authentication and each identity mismatch,
no-follow resource access, record order and failure shapes, receipt-to-observation
binding, deterministic canonical bytes, session cleanup, and discovery without
importing the backend.

A non-first-party provider can be discovered only when
`INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS` is enabled. Strict evidence execution
requires that switch to be false. Consequently, the public ABI supports
experimentation and conformance, but a third-party distribution does not become
an authorized strict-evidence provider merely by installing an entry point.

The execution transaction observes this switch before opening any provider and
rejects when it is enabled. It never records a disabled value merely because a
provider returned otherwise valid records. A deployment that adds a provider
to its authorized first-party set must change and test its own
authorization boundary; environment opt-in is deliberately insufficient.

## Provider conformance checklist

A provider is suitable for the strict path only when it:

1. declares and matches ABI `1` at module and instance level;
2. keeps discovery free of eager backend imports;
3. returns only closed typed capabilities and identities;
4. authenticates local artifact and support resources without following links;
5. runs offline in an observed digest-pinned container;
6. emits one schedule-ordered scoring record per input;
7. binds its receipt to the exact artifact, observation, settings, device, and
   outer image; and
8. closes the session on both success and failure.

## Related documentation

- [Public contracts](contracts.md) defines the provider-emitted document
  schemas and canonical-byte rules.
- [Environment variables](environment.md) lists caller-owned GGUF and
  TensorRT-LLM resources.
- [Python API](api-guide.md) identifies the provider ABI types exported through
  the stable facade.
- [Release verification](release-verification.md) explains add-in version and
  conformance checks.
