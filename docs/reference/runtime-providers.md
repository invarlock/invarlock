# Runtime Providers

Runtime providers let InvarLock authenticate and compare model behavior without
requiring every model to be a Hugging Face/PyTorch checkpoint. The provider
boundary owns artifact identity, deterministic scoring, backend identity,
observed device facts, and a portable receipt. InvarLock owns policy replay and
the paired decision.

## Included Providers

| Provider | Artifact | Execution integration | Installation |
| --- | --- | --- | --- |
| `hf_transformers` | Immutable Hugging Face snapshot or local checkpoint tree | Built-in in-process adapter around an already loaded model, adapter, and scorer | `pip install "invarlock[hf]"` |
| `llama_cpp` | One authenticated GGUF file | First-party process-isolated `llama-completion` provider | Shipped connector plus the pinned GGUF runtime image |
| `tensorrt_llm` | Authenticated TensorRT-LLM engine bundle plus external tokenizer contract | First-party process-isolated runner protocol | Shipped connector plus the pinned TensorRT-LLM runtime image |

All three use the same `invarlock.runtime_providers` entry-point registry and
ship as first-party InvarLock connectors. Hugging Face is not a separate
add-in package: its connector is built in, while its Python backend is enabled
by the optional `[hf]` dependency extra. GGUF and TensorRT-LLM likewise ship
their lightweight connectors in the package, while their native backends stay
inside separately pinned OCI images.

The GGUF and TensorRT-LLM integrations do not use the Hugging Face adapter or
import PyTorch into the InvarLock provider process. Their backend dependencies
belong in separately built, reviewed runtime images:

```bash
make runtime-image-gguf \
  GGUF_BLACKBOX_MODEL=/path/to/stories15M-q4_0.gguf
make runtime-image-tensorrt-llm \
  TENSORRT_LLM_CANARY_ENGINE_BUNDLE=/path/to/authenticated-engine-bundle \
  TENSORRT_LLM_CANARY_TOKENIZER_CONTRACT=/path/to/tokenizer-contract.json \
  TENSORRT_LLM_CANARY_ENGINE_TREE_SHA256=<reviewed-64-hex-tree-digest> \
  TENSORRT_LLM_CANARY_TOKENIZER_SHA256=<reviewed-64-hex-file-digest> \
  TENSORRT_LLM_CANARY_EXPECTED_OUTPUT_SHA256=<reviewed-64-hex-output-digest>
```

The build targets use pinned backend inputs. Record the resulting image's
immutable `sha256:...` digest through an independent build or review channel;
do not treat a mutable local image tag as a trust anchor. The plugin inventory's
`ready` status means the lightweight Python connector is importable; it does not
probe the backend executable, image, GPU, or a particular model artifact.
Machine-readable inventory makes that distinction explicit:
`connector_status` reports connector availability, `backend_delivery` reports
whether the backend comes from a Python extra or OCI image, and the metadata-only
inventory always reports `runtime_qualification: not_probed`. Running a
platform-specific qualification flow does not mutate that inventory. Its result
must be reviewed separately. Likewise, plugin maturity and
`strict_assurance_allowed` describe connector ownership and strict-contract
eligibility; neither field proves that a backend, image, platform, or model
artifact was qualified.

The Docker image definitions, Make targets, and black-box release script are
source-tree qualification tooling rather than installed-package commands. Use the matching
InvarLock source tag (or an authenticated checkout of that exact tag) when
building these images; a PyPI wheel or sdist alone does not contain that
tooling.

The TensorRT-LLM image is based on NVIDIA's pinned NGC 1.2.1 container. Building
it requires access to `nvcr.io` and compliance with the applicable NVIDIA NGC
license terms. Running its smoke or provider path additionally requires Docker
with the NVIDIA Container Toolkit, a compatible NVIDIA driver, and a supported
NVIDIA GPU. Building the image alone is not hardware qualification.

The TensorRT-LLM build first creates a clearly named candidate image. It runs
the CUDA/runtime smoke and two single-rank scores of the fixed `InvarLock`
prompt through fresh provider sessions before tagging the configured stable
local image. The engine and tokenizer paths, their independently reviewed
digests, and the reviewed output-text digest are required before the build
starts. Missing or malformed bindings fail before building; any fixture,
output, evidence-replay, or determinism mismatch fails before promotion. If a
runtime qualification fails after the candidate is built, the target retains
that candidate for diagnosis. No failure path creates or replaces the stable
tag.

Both native providers use the `first_party_experimental` plugin-maturity tier.
GGUF currently targets Linux CPU execution with the pinned llama.cpp build and
records the CPU identity observed inside the execution environment. TensorRT-LLM
targets Linux with an NVIDIA GPU; its runner observes the CUDA device, compute
capability, driver, and runtime, then requires the observed compute capability
to match the engine target. Each isolated score requires InvarLock's
`FORCE_DETERMINISTIC=1` execution marker and fixed greedy decoding settings. The
canary establishes repeatability for its reviewed fixture by requiring
byte-identical evidence across two fresh sessions. The closed environment also
sets backend telemetry and usage-reporting opt-out variables.

## End-to-End Behavioral Journey

Start from two closed JSON inputs. `dataset-identity.json` contains exactly
`provider`, `dataset_name`, `config_name`, `revision`, and `split`.
`behavioral-records.json` is an array whose records contain exactly
`record_id`, `input_text`, and `expected_output`. Build the authenticated
schedule without supplying record hashes yourself:

```bash
invarlock advanced runtime-behavior build-schedule \
  --records behavioral-records.json \
  --dataset-identity dataset-identity.json \
  --out behavioral-schedule.json \
  --json
```

The builder derives every input digest, validates the closed material through
the canonical schedule builder, and writes canonical JSON without replacing an
existing output.

Next prepare one binding for each role inside its selected pinned runtime
image. `prepare-binding` opens the provider, so it must run inside the same
strict, network-disabled container boundary required by the corresponding side
run. Launch the image with all three runtime-boundary variables set:

```text
INVARLOCK_CONTAINER_EXECUTION=1
INVARLOCK_RUNTIME_IMAGE_DIGEST=sha256:<reviewed-64-hex-digest>
INVARLOCK_RUNTIME_IMAGE=<repository>@sha256:<reviewed-64-hex-digest>
```

Pass that same reviewed digest to `--container-image-digest`. The provider
requires the launch variables, image reference, CLI value, and later runtime
manifest to agree; the CLI option by itself is insufficient. These values are
consistency bindings supplied by the execution environment, not remote
attestation that the named image or host executed.

Use the same provider settings, artifact, native executable, auxiliary backend
material, and reviewed image digest that the corresponding side run will use:

```bash
invarlock advanced runtime-behavior prepare-binding \
  --provider llama_cpp \
  --model-id "gguf-sha256-${BASELINE_GGUF_SHA256}.gguf" \
  --settings baseline-gguf-settings.json \
  --artifact /models/baseline.gguf \
  --backend-executable /opt/llama.cpp/llama-completion \
  --backend-source /opt/llama.cpp/source/llama.cpp-b10015.tar.gz \
  --container-image-digest "$REVIEWED_GGUF_IMAGE_DIGEST" \
  --out baseline-binding.json \
  --json

invarlock advanced runtime-behavior prepare-binding \
  --provider tensorrt_llm \
  --model-id "tensorrt-llm-sha256-${SUBJECT_ENGINE_TREE_SHA256}" \
  --settings subject-tensorrt-settings.json \
  --artifact /engines/subject \
  --backend-executable /opt/invarlock/bin/tensorrt-llm-runner \
  --tokenizer-contract /models/subject-tokenizer.json \
  --container-image-digest "$REVIEWED_TENSORRT_IMAGE_DIGEST" \
  --out subject-binding.json \
  --json
```

`prepare-binding` validates and identifies the actual native inputs, opens and
closes the provider without scoring, and derives the execution-settings digest
through the same canonical code used by side production. It also refuses to
replace an existing binding. Each role-binding JSON file contains exactly these
five fields:

| Field | Value |
| --- | --- |
| `provider_name` | Canonical provider name, such as `llama_cpp` or `tensorrt_llm` |
| `artifact_format` | `hf_snapshot`, `gguf`, or `tensorrt_llm_engine` |
| `artifact_identity_sha256` | SHA-256 of the provider's canonical path-free artifact identity |
| `outer_image_digest` | Independently reviewed `sha256:...` runtime-image digest |
| `execution_settings_sha256` | SHA-256 of the canonical provider execution settings |

Build one directed `policy-pack-v3` from the authenticated schedule and exact
per-role bindings:

```bash
invarlock advanced runtime-behavior build-policy \
  --schedule behavioral-schedule.json \
  --baseline-binding baseline-binding.json \
  --subject-binding subject-binding.json \
  --tier balanced \
  --minimum-subject-score 0.95 \
  --maximum-regression 0.01 \
  --evidence-surface behavior \
  --evidence-surface tokenizer \
  --out acceptance-policy-pack.json \
  --json
```

The builder derives the schedule digest and dataset identity from the schedule,
validates both directed bindings, and writes canonical JSON without replacing
an existing output.

Run each side separately in its appropriate pinned runtime image. A side run is
strict, offline, and role-directed: the same authenticated schedule and policy
pack must be supplied to the baseline and subject runs. Runtime behavior
requires `policy-pack-v3`, whose directed baseline and subject entries authorize
the exact provider, artifact identity and format, outer image, execution
settings, and schedule for each role. The command writes a strictly reloaded
side bundle only after provider evidence and its runtime manifest verify.

The examples below show the InvarLock command executed inside the selected
runtime image. Launch that image with networking disabled, mount the artifact,
schedule, policy pack, and settings read-only, and mount only the selected
output directory writable.

The side-run container launch uses the same three runtime-boundary variables and
the same reviewed digest passed to `--container-image-digest` during binding
preparation.

```bash
invarlock advanced runtime-behavior run-side \
  --role baseline \
  --provider llama_cpp \
  --model-id "gguf-sha256-${GGUF_SHA256}.gguf" \
  --settings baseline-gguf-settings.json \
  --artifact /models/baseline.gguf \
  --backend-executable /opt/llama.cpp/llama-completion \
  --backend-source /opt/llama.cpp/source/llama.cpp-b10015.tar.gz \
  --container-image-digest "$REVIEWED_GGUF_IMAGE_DIGEST" \
  --schedule behavioral-schedule.json \
  --policy-pack acceptance-policy-pack.json \
  --out runtime-sides/baseline

invarlock advanced runtime-behavior run-side \
  --role subject \
  --provider tensorrt_llm \
  --model-id "tensorrt-llm-sha256-${ENGINE_TREE_SHA256}" \
  --settings subject-tensorrt-settings.json \
  --artifact /engines/subject \
  --backend-executable /opt/invarlock/bin/tensorrt-llm-runner \
  --tokenizer-contract /models/subject-tokenizer.json \
  --container-image-digest "$REVIEWED_TENSORRT_IMAGE_DIGEST" \
  --schedule behavioral-schedule.json \
  --policy-pack acceptance-policy-pack.json \
  --out runtime-sides/subject
```

The settings files contain public scalar settings and expected identities, not
host paths or secrets. Digest and byte-length values must come from the exact
artifacts and binaries mounted for the run. `llama_cpp` requires:

- `artifact_byte_length`, `artifact_sha256`, `gguf_metadata_sha256`,
  `tensor_inventory_sha256`, and `tokenizer_metadata_sha256`;
- `backend_binary_sha256`, `backend_source_sha256`, and `backend_version`; and
- `seed`, `context_length`, `batch_size`, `max_output_tokens`, and
  `timeout_seconds`.

`tensorrt_llm` requires:

- `engine_bundle_tree_sha256`, `file_inventory_sha256`,
  `builder_config_sha256`, `tokenizer_metadata_sha256`, and
  `engine_metadata_sha256`;
- `runner_binary_sha256`, `backend_build_sha256`, `backend_version`, and
  `target_compute_capability`; and
- `seed`, `context_length`, `batch_size`, `max_output_tokens`, and
  `timeout_seconds`.

Strict cross-runtime sides require `batch_size=1`, meaning one sequence per
scheduled record. Backend-specific token-evaluation or engine build batch
limits are not treated as a portable comparison setting.

For TensorRT-LLM, `--tokenizer-contract` is a closed
`invarlock/tensorrt-llm-tokenizer-contract-v1` wrapper. It contains exactly the
nested `tokenizer_json`, `eos_token_id`, `pad_token_id`,
`add_special_tokens=false`, `skip_special_tokens=true`, and
`clean_up_tokenization_spaces=false`; the SHA-256 of the full wrapper bytes is
`tokenizer_metadata_sha256`. The canonical runner records the device name,
compute capability, driver version, and CUDA runtime version that it observes,
and it requires the observed compute capability to match
`target_compute_capability`. Its backend-build digest is derived from pinned
backend content and checked against the expected `backend_build_sha256` in the
settings. `runner_binary_sha256` names the installed launcher wrapper; the
connector module and its dependency closure remain bound by the reviewed outer
image digest. The outer image digest remains an independently supplied binding;
the receipt is not remote attestation of the execution host or image.

After both sides exist, verify them in a separate process or trust domain. This
command reloads both bundles, replays their observations against the same
schedule and policy, and writes a positive, digest-only receipt. It does not
modify either side bundle.

```bash
invarlock advanced runtime-behavior verify-pair \
  --baseline runtime-sides/baseline \
  --subject runtime-sides/subject \
  --schedule behavioral-schedule.json \
  --policy-pack acceptance-policy-pack.json \
  --receipt runtime-sides/paired-receipt.json \
  --json
```

Commands fail closed with exit status `2`. `--json` emits one versioned object
and never emits a positive receipt for a failed pair.

The verifier proves that two distinct, closed side directories satisfy their
directed policy bindings; it does not prove that they were produced by
independent hosts or operators. A policy may intentionally authorize the same
artifact on both sides for a no-change compatibility check. Use isolated
producers or attestation when execution independence itself is required.

## Runtime Qualification

Build and smoke the backend image on the platform where it will run:

```bash
make runtime-smoke-gguf
make runtime-smoke-tensorrt-llm
make runtime-canary-tensorrt-llm \
  TENSORRT_LLM_CANARY_ENGINE_BUNDLE=/path/to/authenticated-engine-bundle \
  TENSORRT_LLM_CANARY_TOKENIZER_CONTRACT=/path/to/tokenizer-contract.json \
  TENSORRT_LLM_CANARY_ENGINE_TREE_SHA256=<reviewed-64-hex-tree-digest> \
  TENSORRT_LLM_CANARY_TOKENIZER_SHA256=<reviewed-64-hex-file-digest> \
  TENSORRT_LLM_CANARY_EXPECTED_OUTPUT_SHA256=<reviewed-64-hex-output-digest>
```

The GGUF stable-image build requires its 19,077,344-byte black-box fixture,
which is not vendored. Supply the local `stories15M-q4_0.gguf` file from
`ggml-org/tiny-llamas` revision
`99dd1a73db5a37100bd4ae633f4cfce6560e1567`; the target requires SHA-256
`6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04`
and never downloads it:

```bash
make runtime-blackbox-gguf \
  GGUF_BLACKBOX_MODEL=/path/to/stories15M-q4_0.gguf
```

The stable-image target runs this black-box on the candidate before assigning
the stable local tag. The black-box resolves the image tag to its actual image
digest, launches two
separate read-only and network-disabled containers, and invokes the installed
wheel without a source-tree mount or `PYTHONPATH`. It requires the pinned prompt
output and byte-identical canonical observations and provider receipts across
both runs.

The TensorRT-LLM smoke requires Docker with the NVIDIA Container Toolkit and a
visible CUDA device. It checks the pinned package/image environment and the
runner information protocol. The required canary then authenticates the
reviewed engine tree and tokenizer file, executes two real scores in fresh
provider sessions, requires the reviewed output digest, and requires
byte-identical canonical observations and receipts. Those checks qualify the
local image tag, but do not establish a behavioral claim for a particular
schedule. That claim still requires a real `run-side` using the intended engine
and tokenizer, followed by `verify-pair` against the intended schedule and
policy.

## Hugging Face Use

Hugging Face remains the built-in model-loading path for `invarlock evaluate`
when the `[hf]` extra is installed. The runtime-provider Python API also accepts
`hf_transformers`, but strict runtime-behavior evidence does not accept an
arbitrary scorer. It requires the provider-owned
`HFTransformersCausalScorer`, bound to the exact native model and artifact, a
materialized local checkpoint tree in canonical safetensors form, the live
tokenizer contract digest, eval-mode model modules, deterministic greedy causal
scoring, and `batch_size=1`.

The installed `runtime-behavior run-side` command deliberately does not create
those in-process objects from paths. Use `invarlock evaluate` for the normal HF
journey, or call `invarlock.runtime_behavior.run_side` from a controlled Python
integration with pre-bound objects. This avoids an ambiguous command that would
silently load remote code or invent an unauthenticated adapter.
These stricter runtime-behavior requirements do not change the normal
`invarlock evaluate` Hugging Face path.

## Claim Scope

The cross-provider claim is narrow: deterministic record IDs and expected
outputs are replayed under the authenticated schedule, and the provider reports
the `exact_match` metric. Policy decides whether the paired result is
acceptable.

It does **not** establish:

- weight, tensor, spectral, RMT, activation, or quantization equivalence;
- token-level, log-probability, latency, throughput, or numerical equivalence;
- general model quality outside the authenticated records;
- correctness of TensorRT, TensorRT-LLM, llama.cpp, CUDA, or a model export; or
- remote attestation that the declared runtime image actually executed.

Artifact, backend, tokenizer, schedule, policy, observed device, and declared
image bindings make mismatches visible to replay within the execution trust
boundary. The receipt records what the local provider observes; it does not
attest a potentially compromised host. None of these bindings expands the
behavioral measurement beyond exact match and the policy pack's declared claim.
