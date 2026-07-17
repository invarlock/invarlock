# Evaluation request

An evaluation request closes one baseline-versus-subject release-regression
transaction over exact artifacts, a pinned evaluation source, one task,
runtime settings, one built-in metric or bound scorer extension, one policy,
optional authenticated observations, and a fresh evidence destination. The schema is
[`contracts/evaluation_request.schema.json`](https://github.com/invarlock/invarlock/blob/main/contracts/evaluation_request.schema.json).

The request is comparison intent, not host authorization. It does not grant
network access, choose an OCI engine, select arbitrary executables, inject
secrets, or authorize its own runtime digest.

!!! tip "User guide"

    **Outcome:** Author a closed run request for a real paired comparison, or a
    closed import request for complete provider material produced elsewhere.

    **Audience:** Evaluation operators and integration engineers preparing a
    release-regression decision.

    **Prerequisites:** Pinned model inputs, a local JSONL source for run mode or
    a canonical schedule for import mode, a reviewed single-scorer policy,
    provider settings, and a new request-relative evidence destination.

## Choose the execution mode

| Mode | Use it when | Dataset form | Runtime behavior |
| --- | --- | --- | --- |
| `run` | InvarLock should execute both selected providers now | Digest-pinned local JSONL object | Host CLI delegates to a caller-selected Docker/Podman image, prepares the canonical schedule, and scores both sides |
| `import` | Another controlled execution already produced complete InvarLock provider sidecars | Safe relative path to canonical schedule JSON | Local transaction authenticates and replays all imported materials; no OCI delegation is needed |

Run mode is the primary model-comparison path. Import mode is not an aggregate
score upload: it requires ordered record-level observations, identities,
runtime bindings, and canonical paired records that InvarLock can re-derive.

## Request anatomy

The root object has four required fields and one optional field:

| Field | Purpose |
| --- | --- |
| `format_version` | Selects `invarlock/evaluation-request-v1` |
| `comparison` | Names baseline, subject, dataset, task, policy, and exactly one built-in metric or scorer-extension binding |
| `execution` | Selects the closed `run` or `import` shape |
| `observations` | Optionally names canonical JSON context to authenticate with the comparison |
| `output` | Names one safe request-relative, no-clobber evidence directory |

Unknown fields are rejected. A misspelling or unsupported setting is therefore
a request error rather than an ignored option.

## Complete run request

Run mode requires request-relative artifact paths and a pinned local JSONL
dataset object:

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
    limit: 400
  policy: policy/acceptance.json
  task: text_causal
  metric: normalized_nll_per_utf8_byte
execution:
  mode: run
output:
  evidence: evidence/release-001
```

The digest values are illustrative. HF provider-setting digests are bare
lowercase 64-character values. Replace them with values derived from the exact
artifacts and tokenizer contracts before evaluation.

### Run-mode dataset object

| Field | Required | Meaning |
| --- | --- | --- |
| `path` | Yes | Existing request-relative regular JSONL file |
| `sha256` | Yes | Bare lowercase SHA-256 of the exact source bytes |
| `format` | Yes | Exactly `jsonl` |
| `name` | Yes | Logical dataset name recorded in schedule identity |
| `split` | Yes | Logical split recorded in schedule identity |
| `input_field` | Yes | Top-level text field mapped to the `prompt` input part |
| `expected_output_field` | Yes | Distinct top-level text field mapped to `expected_output` |
| `id_field` | No | Distinct top-level text field mapped to stable `record_id` |
| `content_role` | With any content field | Canonical role assigned to the content part; vision-text uses `image` |
| `content_id_field` | No | Top-level content identifier mapped to a content-addressed input part |
| `content_sha256_field` | With any content field | Bare lowercase SHA-256 of the content bytes |
| `content_byte_length_field` | With any content field | Positive byte length of the content object |
| `content_media_type_field` | With any content field | Canonical media type for the content object |
| `limit` | No | Exact prefix length, from 1 through 10,000 |

The transaction verifies `sha256` before parsing. It reads one JSON object per
line, preserves source order, selects the exact prefix when `limit` is present,
and derives each `input_sha256`. Without `id_field`, record IDs are
`record/00000000`, `record/00000001`, and so on. The resulting canonical
schedule identity binds the source digest, name, fixed config
`evaluation-records-jsonl-v1`, split, order, and selected records. The normalized
request separately preserves the exact source path, digest, field mapping, and
limit; both objects are bound into the evidence.

Blank lines, non-object rows, missing or non-text mapped fields, duplicate IDs,
digest mismatch, and a requested limit larger than the file fail closed. Fields
not named by the mapping may remain in the source, but they are not copied into
the schedule.

`content_role` and all four content-field mappings are supplied together or
omitted together. A text-only record becomes one `prompt` text part. A record
with content mappings becomes one content part followed by one `prompt` text
part. The content part carries the declared role, content ID, media type, byte
length, and SHA-256. The schedule contains no host path or URI. The selected
provider resolves content IDs inside its separately authorized content store
and authenticates the bytes before use. `vision_text_generation` requires role
`image`; another canonical future task may declare its own role through its
provider/add-in contract.

## Task and structured-input contract

`comparison.task` is a canonical provider capability identifier. The request,
canonical schedule, both provider capability declarations, both evaluation
batches, and both receipts must agree on it. `text_causal` is the built-in
Hugging Face task. `vision_text_generation` is implemented by the optional
Hugging Face vision-text add-in. The contract also reserves canonical
`masked_language` and `text_seq2seq` identifiers so future providers can share
the same schedule and evidence boundaries; no built-in execution path is
assigned to those identifiers.

Every scheduled record carries ordered `input_parts`. Text parts include their
canonical role, text, and text digest. Content parts include a role,
content-safe identifier, media type, byte length, and digest. `input_sha256` is
derived from the entire ordered part list, so changing a prompt, reordering
parts, or changing a content binding changes the schedule identity.

The schedule authenticates the declaration, not the host object by itself. In
run mode, preflight resolves the caller-authorized provider store and proves
that each selected content ID currently names matching bytes and valid media.
The worker repeats that validation from the exact parsed schedule before model
preparation, and the scorer reopens the object before use. No host path or store
authority enters the portable schedule.

### Comparison sides

Each side has the same shape:

- `artifact.path` identifies local request-owned material in run mode;
- `artifact.model_id` and `artifact.locator` provide portable logical identity;
- `runtime.provider` selects one installed ABI implementation; and
- `runtime.settings` carries only provider-owned JSON scalar settings.

For local HF snapshots, derive `checkpoint_tree_sha256` and
`tokenizer_metadata_sha256` with the public helpers in
[Runtime providers](runtime-providers.md#hugging-face-transformers). The
provider verifies the declared tree and tokenizer contract against the loaded
artifact before and after scoring where required.

The baseline and subject may use different providers only when both declare the
selected task and built-in or collection metric. A cross-runtime comparison records behavior under two
authenticated runtime configurations; it does not assert that their generation
semantics are identical beyond the bound inputs and settings.

## Supported metrics

### `exact_match`

Each side receives score `1` when its authenticated `output_text` is exactly the
scheduled `expected_output`, otherwise `0`. Equality does not trim whitespace,
case-fold, normalize Unicode, or extract an answer. The point comparison is:

```text
delta_pp = 100 * (subject_mean - baseline_mean)
```

The report also records baseline-pass to subject-fail regressions,
baseline-fail to subject-pass improvements, both-pass and both-fail counts, and
the exact two-sided McNemar probability. The policy passes only when the paired
Newcombe 95% effect-size interval's lower bound is at least
`policy.metrics.exact_match.delta_min_pp`.

### `normalized_nll_per_utf8_byte`

For each successful record, the provider teacher-forces the authenticated
expected continuation and supplies finite `logprob_sum`, positive `token_count`,
and positive `utf8_byte_count`. InvarLock verifies the byte count against the
scheduled target and derives:

```text
record_nll = -logprob_sum / utf8_byte_count
ratio = arithmetic_mean(subject_record_nll) / arithmetic_mean(baseline_record_nll)
```

The baseline mean must be positive. The report passes only when the interval's
upper bound is at most
`policy.metrics.normalized_nll_per_utf8_byte.ratio_max`.

This ratio is insensitive to tokenizer token-count differences because target
UTF-8 bytes are the normalization unit. It is not perplexity and is not a pooled
byte-weighted loss. It measures expected-continuation likelihood regression
under the authenticated prompt, target, provider, and runtime; it is not a
general model-quality measure.

### Derived perplexity interpretation

For a normalized-NLL report, InvarLock derives token-normalized NLL when both
artifacts bind the same authenticated tokenizer and each pair has the same
positive target-token count:

```text
record_nll = -logprob_sum / token_count
```

Each side perplexity is the exponential of token-weighted mean NLL. The report
records both side perplexities, their ratio, the tokenizer digest, and total
target-token count. If the facts are not comparable, it records a deterministic
unavailable reason.

The perplexity ratio is verifier-derived interpretation only. It is not a
selectable metric and has no policy threshold, confidence interval, or verdict
authority.

The built-in HF provider supports both built-in metrics for `text_causal`. The
first-party GGUF, TensorRT-LLM, and Hugging Face vision-text add-ins currently
support exact match for their declared tasks. Import execution support remains
provider-specific: both imported provider receipts must declare the selected
task and collection metric. A scorer extension collects authenticated text
outputs through the exact-match provider surface before verifier replay.

## Deterministic scorer extension

`comparison` selects exactly one built-in `metric` or one complete
`scorer_extension` binding. It cannot contain both. A binding has this closed
shape:

```yaml
scorer_extension:
  format_version: invarlock/scorer-extension-binding-v1
  scorer_abi: "1"
  scorer_id: example.token_f1
  scorer_version: 1.0.0
  descriptor_sha256: "5555555555555555555555555555555555555555555555555555555555555555"
  configuration:
    lowercase: true
  configuration_sha256: "6666666666666666666666666666666666666666666666666666666666666666"
```

The digests are illustrative. The descriptor binds supported tasks, text
output, configuration schema, replay mode, unit-interval value semantics,
arithmetic-mean aggregation, higher-is-better direction, and disabled network,
external-model, and human judgment. The configuration digest covers canonical
JSON configuration bytes.

Providers collect ordinary authenticated text outputs. The scorer receives
exactly `expected_output`, `output_text`, and `output_sha256` for each ordered
record, even when the underlying schedule also contains authenticated content.
It returns one deterministic value from `0` through `1` per record. InvarLock
computes both arithmetic means, the subject-minus-baseline percentage-point
delta, the deterministic paired schedule-resampling interval, and the verdict.

Strict evaluation and verification require an explicitly authorized scorer.
The CLI discovers the exact installed entry point only when
`--allow-installed-scorers` is supplied to each transaction. The Python API
accepts an explicit `ScorerExtensionRegistry` containing an installed or
caller-injected scorer.
The scorer ID, version, descriptor digest, configuration digest, task, input
kinds, output kind, pairing, source facts, and replay result must all agree.
Replay runs twice and must produce byte-identical canonical results.

This boundary can support separately implemented deterministic token F1,
structured extraction, or VQA answer normalization. InvarLock does not ship
those scorer packages. SQL or code execution, model-based semantic similarity,
network and human scoring, external models, and LLM judges are excluded until
separate authenticated contracts exist. Judge outputs remain optional
authenticated observations.

## Authenticated optional observations

The optional root `observations` array attaches canonical JSON context to the
same signed evidence transaction:

```yaml
observations:
  - id: subject-spectral
    kind: spectral
    scope: subject
    path: observations/subject-spectral.json
```

Each entry has a unique canonical ID, a canonical kind, a scope of
`comparison`, `baseline`, or `subject`, and a request-relative path to a
canonical JSON object. Evaluation reads the bytes without following links,
requires canonical encoding and a maximum size of 1 MiB, and places the payload
under `observations/<id>.json`. The signed manifest binds its digest, kind,
scope, comparison, schedule, policy, and both artifact identities. Strict
verification replays those bindings, and the human report renders the payload
in a separate context section.

Observations are authenticated context. The selected paired comparison, its
paired interval, and policy are the complete acceptance calculation. Adding,
removing, or changing an observation cannot alter the verdict; any byte change
after publication invalidates bundle integrity. Spectral, random-matrix, and
variance summaries from `invarlock-diagnostics` use this path.

## Interval-controlled verdict

Exact-match reports use `newcombe_hybrid_score_paired_v1`, scope
`paired_binary_outcomes`, and interval mass `0.95`. The lower bound controls the
exact-match policy. Paired regression and improvement counts plus the exact
McNemar probability provide supporting context but do not override that bound.

Normalized-NLL reports use `paired_percentile_bootstrap_sha256_v1` with:

- scope `authenticated_schedule`;
- interval mass `0.95`;
- 2,048 replicates; and
- index draws derived deterministically from the authenticated schedule digest,
  replicate number, and draw number.

Each replicate resamples paired positions, so baseline and subject observations
for one scheduled record remain coupled. The upper bound controls the
normalized-NLL policy. A favorable point estimate cannot override an interval
that crosses the policy limit.

This is a deterministic resampling interval over the authenticated finite
schedule. It is not a population confidence interval, proof of dataset
representativeness, or general model-quality claim.

## OCI delegation for run mode

When the host CLI sees `execution.mode: run` and is not already inside a strict
container boundary, it launches the request in Docker or Podman. Caller-owned
CLI options or environment variables provide:

| Concern | CLI option |
| --- | --- |
| Evidence-signing private key | `--signing-key` |
| Shared runtime image default | `--runtime-image`, `--runtime-image-digest` |
| Baseline image override | `--baseline-runtime-image`, `--baseline-runtime-image-digest` |
| Subject image override | `--subject-runtime-image`, `--subject-runtime-image-digest` |
| OCI engine | `--container-engine` (`docker` or `podman`) |
| Per-worker CPU ceiling | `--runtime-cpus` (default `4`) |
| Per-worker memory ceiling | `--runtime-memory-mib` (default `65536`) |
| Non-root worker identity | `--runtime-user` (default `65532:65532`) |
| Shared device | `--runtime-device` (`cpu`, `cuda`, or `cuda:<index>`) |
| Per-side override | `--baseline-runtime-device`, `--subject-runtime-device` |
| Shared or per-side worker profile | `--runtime-entrypoint`, `--baseline-runtime-entrypoint`, `--subject-runtime-entrypoint` |

The host authenticates the source and prepares the canonical schedule before
launching one constrained worker per side. The launcher uses `--pull=never`,
disables container networking, drops capabilities, and sets a read-only
container root. It also applies caller-owned CPU and memory ceilings, a numeric
non-root identity, and a finite deadline derived from the provider timeout and
authenticated schedule length. Each worker receives only its own artifact and support
resources read-only plus an isolated writable output directory. The host
validates both outputs, publishes `output.evidence`, and signs it; the signing
key is never mounted into either worker.

Common image, device, and entrypoint options are defaults. Per-side options
allow different digest-pinned images or devices. Workers that share a generic
or identical CUDA selector run sequentially; workers assigned explicitly
different CUDA indexes can run in parallel.

The request cannot enable network or remote code. Runtime-image identity,
device selection, resource ceilings, and worker identity remain
caller-controlled so submitted YAML cannot grant itself host capabilities.

## Import mode

Import mode uses a canonical schedule path under `comparison.dataset` and
repeats the same path under `execution.schedule`:

```yaml
comparison:
  # baseline and subject declarations omitted here
  dataset: inputs/schedule.json
  policy: policy/acceptance.json
  task: text_causal
  metric: exact_match
execution:
  mode: import
  records: import/paired-records.json
  schedule: inputs/schedule.json
  baseline:
    identity: import/baseline/model-artifact.identity.json
    receipt: import/baseline/runtime-provider.receipt.json
    observation: import/baseline/runtime-scoring.observation.json
    run_report: import/baseline/report.json
    runtime_manifest: import/baseline/runtime.manifest.json
    runtime_config: import/baseline/run.yaml
  subject:
    identity: import/subject/model-artifact.identity.json
    receipt: import/subject/runtime-provider.receipt.json
    observation: import/subject/runtime-scoring.observation.json
    run_report: import/subject/report.json
    runtime_manifest: import/subject/runtime.manifest.json
    runtime_config: import/subject/run.yaml
output:
  evidence: evidence/imported-001
```

Each side provides a typed artifact identity, provider receipt, scoring
observation, runtime-side report, runtime manifest, and runtime configuration.
InvarLock re-derives paired records from the schedule and observations and
requires canonical equality with `execution.records`. Do not mix sidecars from
different executions; their digests and identities cross-bind.

## Paths, validation, and recovery

Request references are relative to the request file. Absolute paths, traversal,
URLs in path fields, drive prefixes, backslashes, empty components, and symbolic
link escapes are rejected. Public `model_id` and `locator` values are identities,
not filesystem references.

Before execution, review:

- exact baseline, subject, tree, tokenizer, and runtime identities;
- JSONL digest, mapping, order, selection limit, and expected outputs;
- provider support for the chosen metric;
- policy block, threshold direction, and interval-bound rule;
- offline settings, deterministic seed, context length, output limit, and
  timeout;
- a new durable output path; and
- separation between evidence-signer and verifier keys.

`output.evidence` is no-clobber and publication is atomic. If evaluation fails,
fix the request, input, provider, OCI, or publication boundary and retry with a
new destination. Never repair a distributed bundle in place.

Continue with [Schedule and policy](schedule-and-policy.md) for decision design,
[Evidence and verification](evidence-and-verification.md) for the resulting
artifact lifecycle, and [Troubleshooting](troubleshooting.md) for fail-closed
recovery.
