# Getting started

InvarLock evaluates one pinned baseline and one pinned subject on the same
deterministically prepared schedule, applies a release-regression policy to the
selected paired interval, publishes signed evidence, and lets a separate
verifier replay the result.

For a first useful result without a model or container engine, run the
[five-minute signed-evidence check](https://github.com/invarlock/invarlock/tree/main/examples/quickstart).
It installs the core wheel, verifies retained evidence against independent
anchors, issues a new verifier-signed receipt, and renders HTML on a regular
CPU. Continue here when you are ready to produce evidence from your own model
artifacts.

```text
invarlock evaluate request.yaml
invarlock verify evidence/ ... --receipt verification.receipt.json
invarlock report evidence/
```

!!! tip "User guide"

    **Outcome:** Prepare and run a real paired Hugging Face comparison through
    the OCI-backed `evaluate -> verify -> report` path.

    **Audience:** First-time operators, release decision owners, and engineers
    integrating a model build with an evidence gate.

    **Prerequisites:** Python 3.12 or newer; Docker or Podman; a digest-addressed
    local InvarLock runtime image; two local SafeTensors snapshots; a pinned
    JSONL evaluation source; and separate Ed25519 evidence-signer and verifier
    keys.

To check results from an existing evaluator without preparing an OCI runtime,
start with [Existing evaluation pipelines](pipeline-integration.md). That
unreleased companion workflow supports multiple metrics and slices with its own
captured-result evidence contract. This page covers the core execution and
independent receipt path.

## What each command owns

| Transaction | Reads | Writes | Result |
| --- | --- | --- | --- |
| `evaluate` | Closed request, JSONL source, local artifacts, caller-owned OCI/runtime inputs, evidence-signing key | One no-clobber signed evidence directory | Paired comparison, interval, optional sample and side-accuracy qualification, and evaluation-time policy verdict |
| `verify` | Untrusted evidence plus independent artifact, schedule, policy, runtime, and signer anchors; verifier identity; and verifier key | One no-clobber signed receipt outside the bundle | Independent replay and acceptance/rejection record |
| `report` | Signature-authenticated evidence | Console text and optional no-clobber HTML | Human view of the canonical comparison |

Run mode is the primary path. Import mode is available when another controlled
execution already produced the complete provider sidecars; the final section
uses the repository fixture to exercise that secondary path without model
inference.

## 1. Install the HF provider

```bash
python -m pip install "invarlock[hf]"
invarlock --version
invarlock evaluate --help
```

Build or obtain the runtime image from the same release you intend to run. It
must be available locally under a digest-bearing reference or exact image ID.
The host launcher uses `--pull=never`, so evaluation never substitutes an image
download for caller-controlled preparation.

For a source checkout, choose the image that matches the worker device:

First create and authenticate the exact Git source archive using the
[runtime-image procedure](runtime-providers.md#build-or-obtain-the-runtime-image).
Pass its commit, path, and digest to the build target shown below.

| Device | Dockerfile | Build and smoke |
| --- | --- | --- |
| CPU, including the matching Apple Silicon closure | `runtime/Dockerfile` | `make runtime-image && make runtime-smoke` |
| x86_64 NVIDIA CUDA 12.6 | `runtime/Dockerfile.cuda` | `make runtime-image-cuda && make runtime-smoke-cuda` |

The CUDA device flag only exposes a GPU to a CUDA-capable image. It does not
add CUDA libraries to the CPU image.

For a complete journey that compares a pinned Qwen3.5-0.8B checkpoint with an
explicit behavioral derivative and derives all verifier inputs independently,
run the checked-in
[Hugging Face example](https://github.com/invarlock/invarlock/tree/main/examples/integrations/hf-transformers).
The remaining sections
explain the same transaction field by field for a real release workspace.

Record the resulting `sha256:...` identity. A local image ID authenticates
local bytes but is not a fetchable reference for another operator; use a pinned
registry-manifest reference when the evidence must be reproduced elsewhere.

## 2. Prepare the request workspace

Use a new directory for one comparison:

```text
release-check/
├── request.yaml
├── artifacts/
│   ├── baseline/                 # local HF SafeTensors snapshot
│   └── subject/                  # candidate snapshot
├── inputs/
│   └── release-regression.jsonl
└── policy/
    └── acceptance.json
```

The evidence destination declared in `request.yaml` is created beneath this
directory by the host transaction. Keep the evidence-signing private key
outside the tree. It remains host-only and is never mounted into a model worker.

### Pin the evaluation source

Each selected JSONL line must be a JSON object. For example:

```json
{"case_id":"release/001","prompt":"Return the capital of France.","expected":"Paris"}
{"case_id":"release/002","prompt":"Complete: 2 + 2 =","expected":"4"}
```

Hash the exact file bytes:

```bash
python -c 'import hashlib, pathlib; print(hashlib.sha256(pathlib.Path("release-check/inputs/release-regression.jsonl").read_bytes()).hexdigest())'
```

Do not sort, serialize again, or edit the file after recording the digest. Run mode
preserves source order, maps the declared top-level fields, and constructs
canonical `invarlock/runtime-behavioral-schedule-v1` bytes. If `id_field` is
omitted, IDs are deterministically generated as `record/00000000`,
`record/00000001`, and so on. Stable source IDs are preferable for review.

### Pin both HF snapshots

Derive `checkpoint_tree_sha256` and `tokenizer_metadata_sha256` from each exact
local snapshot with the public helpers shown in
[Runtime providers](runtime-providers.md#hugging-face-transformers). The tree
digest rejects links, special files, and changes during traversal. The
tokenizer digest binds the tokenizer contract used to interpret target text.

Normalized NLL remains comparable across authenticated tokenizer contracts
because the target loss is normalized by expected-output UTF-8 bytes. When the
two tokenizer digests match and each pair also has the same positive target
token count, the report can add a verifier-derived token-weighted perplexity
ratio as a likelihood interpretation.

### Write a one-metric policy

This example permits a normalized expected-continuation NLL ratio up to `1.05`:

```json
{"resolved_policy":{"metrics":{"normalized_nll_per_utf8_byte":{"ratio_max":1.05}}}}
```

The verdict uses the interval upper bound, not only the point ratio. See
[Schedule and policy](schedule-and-policy.md) before selecting a production
threshold.

## 3. Write the run request

Replace every illustrative digest with the value derived from the exact input.
Digest fields inside HF provider settings use the exact form returned by their
public helper; runtime-image identities always use the `sha256:` prefix.

```yaml
format_version: invarlock/evaluation-request-v1
comparison:
  baseline:
    artifact:
      path: artifacts/baseline
      model_id: acme/baseline
      locator: hf://acme/baseline@0123456789abcdef0123456789abcdef01234567
    runtime:
      provider: hf_transformers
      settings:
        batch_size: 1
        checkpoint_tree_sha256: "1111111111111111111111111111111111111111111111111111111111111111"
        context_length: 2048
        immutable_revision: 0123456789abcdef0123456789abcdef01234567
        max_output_tokens: 64
        offline: true
        seed: 7
        timeout_seconds: 300
        tokenizer_metadata_sha256: "3333333333333333333333333333333333333333333333333333333333333333"
  subject:
    artifact:
      path: artifacts/subject
      model_id: acme/subject
      locator: hf://acme/subject@fedcba9876543210fedcba9876543210fedcba98
    runtime:
      provider: hf_transformers
      settings:
        batch_size: 1
        checkpoint_tree_sha256: "2222222222222222222222222222222222222222222222222222222222222222"
        context_length: 2048
        immutable_revision: fedcba9876543210fedcba9876543210fedcba98
        max_output_tokens: 64
        offline: true
        seed: 7
        timeout_seconds: 300
        tokenizer_metadata_sha256: "3333333333333333333333333333333333333333333333333333333333333333"
  dataset:
    path: inputs/release-regression.jsonl
    sha256: "4444444444444444444444444444444444444444444444444444444444444444"
    format: jsonl
    name: release-regression
    split: validation
    input_field: prompt
    expected_output_field: expected
    id_field: case_id
  policy: policy/acceptance.json
  task: text_causal
  metric: normalized_nll_per_utf8_byte
execution:
  mode: run
output:
  evidence: evidence/release-001
```

The request contains comparison intent, not host authorization. It cannot
select an OCI engine, grant network access, name arbitrary executables, or
embed private-key bytes.

## 4. Evaluate through OCI

From the directory containing `release-check/`, run:

```bash
invarlock evaluate release-check/request.yaml \
  --signing-key evidence-signer.pem \
  --baseline-runtime-image registry.example/invarlock-runtime-cuda@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --baseline-runtime-image-digest sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --subject-runtime-image registry.example/invarlock-runtime-cuda@sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd \
  --subject-runtime-image-digest sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd \
  --container-engine docker \
  --baseline-runtime-device cuda:0 \
  --subject-runtime-device cuda:1
```

`evaluate` performs preflight before it starts either worker. Append
`--preflight --json` when you want to stop at that boundary and inspect the
result. It validates the request, referenced inputs, schedule, policy, runtime
capabilities, signing key, output destination, OCI engine, and both local
pinned images without starting a container or creating output. Rerun the same
command without `--preflight` only when you deliberately want to continue into
the evaluation.

Use `--container-engine podman` when appropriate. Device values are `cpu`,
`cuda`, or `cuda:<index>`. Shared `--runtime-image`,
`--runtime-image-digest`, `--runtime-device`, and `--runtime-entrypoint`
options provide defaults for both sides. Per-side options override those
defaults. Workers that share a generic or identical CUDA device run
sequentially; explicitly different CUDA indexes can run in parallel.

When invoked from the host, `evaluate` authenticates the source and prepares
the canonical schedule, then launches a constrained worker for each side. Each
worker receives only its artifact and support resources read-only, an isolated
writable output directory, a read-only container root, and the selected
CPU/CUDA resources. Networking is disabled and capabilities are dropped. The
host validates both outputs, atomically publishes
`release-check/evidence/release-001`, and signs the bundle with its host-only
key.

A successful command exits `0` and reports the evidence directory. The bundle
contains the prepared canonical schedule, both provider observations, paired
records, the comparison report, runtime bindings, checksums, manifest, and
evidence signature.

## 5. Verify with independent anchors

The verifier must obtain these values through its own trusted channel:

- the exact policy file;
- expected baseline and subject artifact-identity digests;
- the expected canonical schedule digest;
- expected baseline and subject runtime-image digests;
- expected evidence-signer fingerprint;
- expected normalized-request digest when either side uses `llama_cpp`;
- its verifier identity and private key.

```bash
invarlock verify release-check/evidence/release-001/ \
  --policy release-check/policy/acceptance.json \
  --expected-baseline-artifact sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc \
  --expected-subject-artifact sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee \
  --expected-schedule sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --expected-baseline-runtime sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --expected-subject-runtime sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd \
  --expected-signer sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --receipt release-check/verification.receipt.json \
  --verifier-signing-key verifier.pem \
  --verifier-identity release-verifier \
  --json
```

Acceptance requires exit status `0`, `ok: true`, `integrity_ok: true`, a
passing policy verdict, the expected signer authenticity, and a valid signed
receipt. An integrity-valid policy rejection may still produce a signed
rejection receipt and then exit nonzero.

## 6. Render the report

```bash
invarlock report release-check/evidence/release-001/
invarlock report release-check/evidence/release-001/ \
  --html release-check/evidence.html \
  --explain
```

The report shows both side means, the point comparison, the selected paired
interval, threshold, and canonical verdict. `report` checks the embedded
evidence signature and bundle integrity before rendering, but it does not use
the independent verifier anchors or replace the signed receipt.

## Interpret the result

Every current `invarlock/comparison-report-v3` report records:

- a point estimate over all authenticated records;
- a paired Newcombe 95% interval plus regression/improvement counts and an
  exact McNemar probability for exact match, or the fixed 2,048-replicate
  `paired_percentile_bootstrap_sha256_v1` interval for normalized NLL;
- the selected policy limit; and
- optional record-count and interval-width qualification when the policy binds
  those coupled requirements;
- optional baseline and subject accuracy qualification for exact match when the
  policy binds `minimum_side_accuracy`; and
- a verdict controlled by the conservative interval bound and any configured
  sample and side-accuracy qualification.

For exact match, the lower bound must clear the percentage-point floor. For
byte-normalized NLL, the upper bound must remain below the ratio ceiling. Its
interval resamples paired schedule positions, so each baseline observation
stays coupled to its subject observation.

Preflight can establish that the authenticated schedule meets a configured
minimum count, but interval width remains pending until execution. Strict
verification also preserves v2 reports without side-accuracy qualification and
v1 evidence with its original exact-match method.

Normalized NLL is teacher-forced expected-continuation likelihood regression,
not a general model-quality measure. A displayed token-weighted perplexity
ratio is derived interpretation only; it has no policy, interval, or verdict
authority.

This describes sensitivity to paired resampling of the fixed schedule. It is
not a population confidence interval, statistical-power calculation, safety
claim, or proof that the schedule represents production traffic.

## Secondary offline import smoke

The repository's
[`examples/`](https://github.com/invarlock/invarlock/tree/main/examples)
directory contains complete synthetic provider sidecars. It makes no network
calls and performs no model inference, but exercises the same evidence,
signature, interval, verifier, receipt, and rendering contracts:

```bash
cd examples
python generate_keys.py --output-dir .keys
invarlock evaluate request.yaml --signing-key .keys/evidence-signer.pem

EVIDENCE_SIGNER_FINGERPRINT="$(tr -d '\n' < .keys/evidence-signer.fingerprint)"
BASELINE_ARTIFACT_DIGEST="$(python -c 'import json; print(json.load(open("trusted-inputs/input-digests.json"))["baseline_artifact"])')"
SUBJECT_ARTIFACT_DIGEST="$(python -c 'import json; print(json.load(open("trusted-inputs/input-digests.json"))["subject_artifact"])')"
SCHEDULE_DIGEST="$(python -c 'import json; print(json.load(open("trusted-inputs/input-digests.json"))["canonical_schedule"])')"
invarlock verify artifacts/evidence/ \
  --policy policy/acceptance.json \
  --expected-baseline-artifact "$BASELINE_ARTIFACT_DIGEST" \
  --expected-subject-artifact "$SUBJECT_ARTIFACT_DIGEST" \
  --expected-schedule "$SCHEDULE_DIGEST" \
  --expected-baseline-runtime sha256:1111111111111111111111111111111111111111111111111111111111111111 \
  --expected-subject-runtime sha256:2222222222222222222222222222222222222222222222222222222222222222 \
  --expected-signer "$EVIDENCE_SIGNER_FINGERPRINT" \
  --receipt verification.receipt.json \
  --verifier-signing-key .keys/verifier.pem \
  --verifier-identity local-example-verifier

invarlock report artifacts/evidence/ --html evidence.html --explain
```

Use this fixture to test key handling and artifact transport, not as evidence
that a runtime-backed model comparison works.

## Apply the transaction to a release change

The [model-change workflow guide](change-scenarios.md) maps fine-tuned, pruned,
quantized, GGUF, TensorRT-LLM, multimodal, harness, endpoint, and evidence
handoff inputs to the same transaction. The
[runnable examples](https://github.com/invarlock/invarlock/tree/main/examples)
execute the maintained Hugging Face, PEFT, TorchAO, GGUF/llama.cpp, LM
Evaluation Harness, and TensorRT-LLM journeys. A separate evidence-handoff
fixture exercises acceptance, policy rejection, and tamper rejection without a
model runtime. Artifact creation remains in the system that owns it.

## Recovery and next steps

Evidence, receipt, and HTML destinations are no-clobber. Do not repair a
published bundle in place. Retain a failed attempt when review policy requires
it, fix the source boundary, and use a new output name.

Continue with:

- [Evaluation request](evaluation-request.md) for every request field;
- [Schedule and policy](schedule-and-policy.md) for source preparation,
  thresholds, and interval semantics;
- [Runtime providers](runtime-providers.md) for artifact identity and provider
  qualification;
- [Evidence and verification](evidence-and-verification.md) for transfer and
  independent review; and
- [Troubleshooting](troubleshooting.md) for fail-closed recovery.
