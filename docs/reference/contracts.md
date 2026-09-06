# Public contracts

The workflow uses closed, versioned JSON contracts. Source mirrors live in
[`contracts/`](https://github.com/invarlock/invarlock/tree/main/contracts), and
byte-identical package-owned copies ship in the core wheel. Verification always
loads the package-owned copies: a working directory or environment variable
cannot substitute a different schema.

!!! info "Reference"

    - **Surface:** Versioned request, evidence, provider, runtime, report, and receipt contracts
    - **Stability:** Closed public interchange formats; incompatible shape or meaning changes require a new format identifier
    - **Use this page when:** Authoring contract objects, validating canonical bytes, or reviewing cross-file digest and signature bindings

## Schema-backed contracts

| File | Format | Purpose |
| --- | --- | --- |
| `evaluation_request.schema.json` | `invarlock/evaluation-request-v1` | One closed run-or-import request |
| `evidence_pack.schema.json` | `invarlock/evidence-pack-v1` | Canonical bundle manifest and fixed payload paths |
| `evidence_observation.schema.json` | `invarlock/evidence-observation-v1` | Typed observation-only envelope and comparison bindings |
| `trust_inputs.schema.json` | `invarlock/trust-inputs-v1` | Independent policy, anchors, verifier identity/key path, and scorer authorization |
| `acceptance_predicate.schema.json` | `invarlock/acceptance-predicate-v2` | Portable projection of one technical decision in an in-toto Statement |
| `recipient_acceptance_policy.schema.json` | `invarlock/recipient-acceptance-policy-v2` | Current recipient trust, freshness, version, signer, and verdict rules |
| `evaluator_qualification_profile.schema.json` | `invarlock/evaluator-qualification-profile-v1` | Evaluator identity, execution provenance, and authority classification |
| `evaluator_qualification_schedule.schema.json` | `invarlock/evaluator-qualification-schedule-v1` | Independent ordered record and reference identities |
| `evaluator_qualification_export.schema.json` | `invarlock/evaluator-qualification-export-v1` | Normalized per-record facts or an observation-only summary |
| `evaluator_qualification_result.schema.json` | `invarlock/evaluator-qualification-result-v1` | Digest-bound qualification outcome and import authority |

The acceptance predicate and recipient policy are described in
[Acceptance attestations](acceptance-attestations.md). The detailed InvarLock
receipt remains the authoritative replayable result.

Evaluator qualification has two stable wire classifications:

| Qualification result | Meaning |
| --- | --- |
| `outcome: qualified_for_import`, `authority: verdict_authority` | Complete ordered facts passed deterministic recomputation and are independently replayable |
| `outcome: observation_only`, `authority: observation_only` | Authenticated context was retained but cannot contribute imported verdict facts |

These fields do not express adapter maintenance or signed-journey maturity.
Those independent axes belong to the examples-layer qualification catalog, so
an export cannot promote itself by claiming support or demonstration status.

## Identity and evaluation-context boundaries

A model name is a display label. It does not select the evaluated artifact,
package, runtime, scorer or complete execution context. Independent verification
requires recipient-selected expectations and checks the relationships between
the authenticated objects, in addition to validating their schemas.

| Requirement | Existing binding and recipient check | Authority and limit |
| --- | --- | --- |
| Exact artifact | Typed HF snapshot, GGUF or TensorRT-LLM identity; independent artifact-identity digest and actual content checks where supplied | Content identity is separate from a model name or declared ancestry |
| Tokenizer and template | Artifact tokenizer-metadata digest; supported providers measure relevant files, including HF chat-template metadata | Additional processor or execution settings must be checked through the applicable provider/request binding |
| Evaluated task and schedule | Normalized request, canonical schedule, and both authenticated provider capability declarations must agree | A valid signature cannot make conflicting task declarations consistent |
| Generation, runtime and security settings | Provider-specific request/receipt checks and independent runtime digests; a full normalized-request digest can additionally pin the exact declared context | Authenticated settings and local enforcement tests do not establish independent hardware attestation |
| Scorer and evaluation data | Scorer identifier/version/descriptor/configuration bindings, policy bytes, and independently selected schedule identity | A qualified scoring domain does not imply support for every behavior of the external evaluator |
| Complete transaction request | Optional independent `request_digest` in the trust-input profile, or `--expected-request-digest`; required for the llama.cpp path | Receipt v2 records this expectation; receipt v1 does not acquire it retroactively |
| Package-to-model mapping | The [ModelKit example](../user-guide/modelkit-handoff.md) verifies recipient-selected package blobs, both model directories and their relation to replayed evidence | This is an example-owned point-of-use check; the generic acceptance envelope alone does not verify a ModelKit |
| Transformation or contextual observations | Canonical payload digest in the normalized request, comparison-bound observation envelope, and signed manifest inventory | Authenticates the payload and its association; arbitrary payload claims are not independently validated |
| Current recipient acceptance | Trusted envelope and receipt signers, exact transported identity consistency, actual subject binding, contract versions, freshness and current policy | Current acceptance is separate from the original technical result |

The independently selected complete-request digest can require exact declared
settings and observation contents without introducing another model hash. It
does not prove that a hosted service truthfully reported its revision or that an
observation's scientific conclusion is correct. Do not obtain an expected digest
from incoming evidence and describe the resulting equality as independent trust.

Observation payloads may use a versioned profile for method, configuration,
probe-set identity, assumptions, result and uncertainty. A consumer must implement
that profile's semantic checks before claiming to understand or validate it.
The current recipient policy does not implement a generic required lineage or
context-profile result. Its optional receipt trust-profile digest identifies the
selected verifier configuration; it does not create missing validation logic.
Unknown policy fields reject, while opaque observation payloads retain only
observation authority.

Declared transformation history, empirical similarity and exact artifact
identity remain separate. No supported behavioral fingerprint proves ancestry
merely because its payload is signed. A future profile must avoid including the
complete normalized-request digest inside a payload already hashed by that
request; bind component identities first and let the envelope add comparison
bindings after normalization.

The separate [pipeline contracts](pipeline-contracts.md) bind complete captured
runs and policy bytes, including source versions and per-record context. Their
independent run digests include outputs, so they are not pre-execution context
identities. Replaying signed pipeline evidence verifies the captured comparison;
it does not prove that an untrusted capture worker executed the declared model.
The native rehearsal's recipient independently pins the protocol and capture
before reconstructing those runs.

## Provider contracts

The provider ABI uses these schema-backed documents:

| File | Format | Purpose |
| --- | --- | --- |
| `model_artifact_identity.schema.json` | `invarlock/model-artifact-identity-v1` | Portable HF, GGUF, or TensorRT-LLM artifact identity |
| `runtime_provider_receipt.schema.json` | `invarlock/runtime-provider-receipt-v1` | Provider, backend, artifact, settings, device, image, and observation binding |
| `runtime_scoring_observation.schema.json` | `invarlock/runtime-scoring-observation-v1` | Ordered backend-measured record facts |
| `runtime_behavioral_schedule.schema.json` | `invarlock/runtime-behavioral-schedule-v1` | Dataset identity and ordered input records |
| `runtime_manifest.schema.json` | `runtime-manifest-v1` | Strict execution envelope and sibling sidecar digests |
| `runtime_provider_capabilities.json` | `runtime-provider-capabilities-v1` | Provider ABI, artifacts, tasks, metrics, execution modes, and requirements |

Python callers can load these exact packaged objects with functions from
`invarlock.public_contracts`. Those loaders return schema dictionaries; they do
not validate semantic cross-bindings by themselves.

```python
from invarlock.public_contracts import (
    load_evaluation_request_schema,
    load_evidence_observation_schema,
    load_evidence_pack_schema,
    load_trust_inputs_schema,
    load_acceptance_predicate_schema,
    load_recipient_acceptance_policy_schema,
    load_model_artifact_identity_schema,
    load_runtime_behavioral_schedule_schema,
    load_runtime_manifest_schema,
    load_runtime_provider_capabilities_schema,
    load_runtime_provider_receipt_schema,
    load_runtime_scoring_observation_schema,
)
```

Each loader returns a new dictionary decoded from the package-owned contract.
`ContractLoadError` identifies a missing, malformed, or non-object packaged
contract. Format constants such as `EVALUATION_REQUEST_FORMAT_VERSION`,
`EVIDENCE_PACK_FORMAT_VERSION`, `TRUST_INPUTS_FORMAT_VERSION`, and
`RUNTIME_PROVIDER_ABI_VERSION` are exported from the same module for exact
comparisons.

The schemas use [JSON Schema Draft
2020-12](https://json-schema.org/draft/2020-12). Every contract object is
closed: fields not named by its schema or exact code-enforced shape are
rejected rather than ignored.

## Independent trust-input profile

`invarlock/trust-inputs-v1` is the portable caller-owned input to independent
verification. It contains the policy path, baseline and subject artifact
digests, schedule digest, both runtime digests, evidence-signer fingerprint,
verifier identity, verifier signing-key path, and installed-scorer
authorization. The object and all nested objects are closed.

Policy and key paths are safe relative paths resolved from the profile's
directory. Absolute paths, traversal, symlinks, duplicate JSON members,
unknown fields, and missing files are rejected. Formatting does not affect the
profile digest: the loader hashes canonical JSON and the verifier records that
digest in its signed receipt. The profile never contains private-key bytes.

## Evaluation request fields

The request root contains four required fields and one optional field:

| Field | Type | Required value or role |
| --- | --- | --- |
| `format_version` | String | `invarlock/evaluation-request-v1` |
| `comparison` | Object | Baseline, subject, dataset, task, policy, and exactly one metric or scorer-extension binding |
| `execution` | Object | Exactly one `run` or `import` transaction |
| `observations` | Array | Zero to 64 authenticated context attachments |
| `output` | Object | Exactly `evidence`, a safe relative destination |

Each comparison side has the same closed shape:

| Path | Type | Requirement | Meaning |
| --- | --- | --- | --- |
| `artifact.model_id` | String | Yes | Human-stable artifact name; URL syntax is rejected |
| `artifact.locator` | String | Yes | Portable source locator bound into request intent |
| `artifact.path` | Safe relative path | Required in run mode | Artifact path below the request root |
| `runtime.provider` | Provider name | Yes | Selected runtime-provider ABI implementation |
| `runtime.settings` | Object of JSON scalars | Yes | Provider-owned settings validated against capabilities |

`comparison.policy` is always a safe relative path. `comparison.dataset` is
mode-specific:

- run mode requires a closed local-dataset object with `path`, bare lowercase
  `sha256`, `format: jsonl`, `name`, `split`, `input_field`, and
  `expected_output_field`, plus optional `id_field`, an all-or-none content
  role and field mapping, and `limit`;
- import mode requires a safe relative path to canonical
  `invarlock/runtime-behavioral-schedule-v1` bytes.

The canonical `comparison.task` binds the request, schedule, provider
capabilities, evaluation batches, and provider receipts. Built-in identifiers
are `text_causal`, `masked_language`, `text_seq2seq`, and
`vision_text_generation`; a provider must explicitly declare execution
support. The request selects exactly one built-in `metric` or one complete
`scorer_extension` binding. Built-in metrics are `exact_match` and
`normalized_nll_per_utf8_byte`; request loading requires both selected
providers to declare the chosen built-in metric. A scorer extension instead
uses `exact_match` as its provider collection metric so that expected and
observed text are authenticated for verifier replay. The built-in
`hf_transformers` provider declares both built-in metrics. The
first-party `llama_cpp`, `tensorrt_llm`, and `hf_vision_text` add-ins currently
declare exact match for their tasks.

### Deterministic scorer extension

`comparison.scorer_extension` is a closed
`invarlock/scorer-extension-binding-v1` object. It binds scorer ABI `1`, a
dotted scorer ID, semantic version, descriptor digest, configuration object,
and canonical configuration digest. The descriptor fixes supported tasks and
input/output kinds, the configuration-schema digest, and these v1 semantics:

- replay reads exactly authenticated `expected_output`, `output_text`, and
  `output_sha256` facts for every ordered record;
- each result is a finite higher-is-better value in `[0, 1]`;
- the core computes the arithmetic mean, subject-minus-baseline percentage-
  point delta, and fixed 2,048-replicate paired interval; and
- network access, an external model, and human judgment are forbidden in an
  acceptance scorer.

The independently supplied policy must contain
`resolved_policy.metrics.scorer_extension` with the same `scorer_id`,
`scorer_version`, `descriptor_sha256`, and `configuration_sha256`, plus
`delta_min_pp`. Evaluation and verification require an explicitly authorized
`ScorerExtensionRegistry`; the request and evidence cannot authorize scorer
code. The verifier runs the scorer twice, requires identical canonical
results, then independently reconstructs the core-owned aggregate, paired
interval, threshold comparison, and verdict.

This boundary can support deterministic text scorers such as token F1,
structured-field extraction, or VQA answer normalization when separately
implemented and authorized. Those scorer packages are separately installed and
require explicit authorization. SQL or code execution, model-based semantic
similarity, network services,
human review, and LLM judges require different trust contracts; judge outputs
can be attached as authenticated observations without acceptance authority.

### Evaluator input boundary

Import mode is the general extension boundary for measurements produced by an
external evaluator. An evaluator's output is admissible for an acceptance
decision only when InvarLock can authenticate the ordered per-record inputs
and outputs, bind them to the exact schedule, artifacts, runtime, and source,
and deterministically recompute the decision-contract metric or authorized
scorer.

An adapter alone does not establish evaluator neutrality. The generic
qualification boundary binds the profile, independent schedule, normalized
export, retained upstream output, runner bundle, and dependency declaration.
For a deterministic exact-match profile, every ordered input and output must be
present and InvarLock independently recomputes every score. Aggregate-only
results, missing or reordered record facts, and external-judge outputs whose
scores cannot be deterministically replayed remain observation-only and expose
no runtime-import records.

The maintained [evaluator qualification
matrix](evaluator-qualification.md) executes representative upstream tools
through example-owned runners. Every deterministic profile also scores and
replays the complete 102-record output of a pinned real model evaluation
through the runtime-import boundary. The matrix separately records profiles
that demonstrate the deeper model-running, signed transaction journey. These
levels record evidence maturity rather than a permanent support hierarchy and
can advance without changing the generic boundary. Evaluator names and native
parsers remain outside the engine; a private evaluator crosses the same JSON,
CLI, or Python SDK boundary.

### Run request

```yaml
format_version: invarlock/evaluation-request-v1
comparison:
  baseline:
    artifact:
      model_id: acme/baseline
      locator: registry://acme/baseline@immutable-revision
      path: models/baseline
    runtime:
      provider: hf_transformers
      settings:
        immutable_revision: 0123456789abcdef0123456789abcdef01234567
        checkpoint_tree_sha256: "1111111111111111111111111111111111111111111111111111111111111111"
        tokenizer_metadata_sha256: "3333333333333333333333333333333333333333333333333333333333333333"
        batch_size: 1
        context_length: 2048
        max_output_tokens: 64
        offline: true
        seed: 7
        timeout_seconds: 120
  subject:
    artifact:
      model_id: acme/subject
      locator: registry://acme/subject@immutable-revision
      path: models/subject
    runtime:
      provider: hf_transformers
      settings:
        immutable_revision: fedcba9876543210fedcba9876543210fedcba98
        checkpoint_tree_sha256: "2222222222222222222222222222222222222222222222222222222222222222"
        tokenizer_metadata_sha256: "3333333333333333333333333333333333333333333333333333333333333333"
        batch_size: 1
        context_length: 2048
        max_output_tokens: 64
        offline: true
        seed: 7
        timeout_seconds: 120
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
  evidence: artifacts/evidence
```

The digest values are illustrative. A real request must bind the actual source,
artifact, and tokenizer bytes. Provider identity digests in `runtime.settings`
and `comparison.dataset.sha256` use bare lowercase 64-character values; general
bundle and runtime identities use the `sha256:` prefix where their contracts
require it.

For run mode, the transaction verifies the JSONL digest, preserves source
order, maps the declared top-level text fields, selects the exact prefix named
by `limit`, derives stable IDs or deterministic position IDs, and builds the
canonical schedule. Blank lines, invalid JSON objects, missing mapped text,
duplicate IDs, digest mismatch, or a limit larger than the source fail closed.

### Import request

Import mode replaces provider execution with authenticated sidecars. In
addition to `mode`, it requires `records`, `schedule`, and these six references
for both `baseline` and `subject`:

| Import-side field | Expected document |
| --- | --- |
| `identity` | `invarlock/model-artifact-identity-v1` |
| `receipt` | `invarlock/runtime-provider-receipt-v1` |
| `observation` | `invarlock/runtime-scoring-observation-v1` |
| `run_report` | `invarlock/runtime-side-report-v1` |
| `runtime_manifest` | `runtime-manifest-v1` |
| `runtime_config` | Canonical provider run configuration |

```yaml
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
```

This fragment is inserted under the same request root as the `comparison` and
`output` objects. In import mode, `comparison.dataset` and
`execution.schedule` both name the canonical schedule. Evaluation re-derives
artifact identities and paired records; supplying a valid-looking aggregate is
not sufficient.

## Parser and path limits

| Boundary | Limit or rule |
| --- | --- |
| Request YAML | At most 1 MiB, 64 nested levels, and 10,000 syntax nodes |
| Policy input | At most 4 MiB |
| Other request input | At most 64 MiB per file |
| Local JSONL source or schedule bytes | At most 16 MiB and 1 to 10,000 selected records |
| References | Relative, root-confined, forward-slash paths; no links, traversal, URLs, drive prefixes, or empty components |

The YAML loader accepts only a JSON-compatible mapping. It rejects aliases,
anchors, tags, directives, merge keys, duplicate keys, include-like keys,
unsafe scalar types, and non-canonical scalar spellings. File reads repeat
component-by-component no-follow checks at use time, so a path that passed the
first parse cannot be replaced with a symbolic link unnoticed.

The built-in comparison metrics are `exact_match` and
`normalized_nll_per_utf8_byte`. Exact-match reports include paired outcome
counts, an exact two-sided McNemar probability, and a versioned paired Newcombe
95% interval whose lower bound controls policy. Current v3 reports and
historical v2 reports use the continuity-corrected method; strict verification
preserves the original method for legacy v1 reports. Normalized-NLL reports include the
fixed 2,048-replicate `paired_percentile_bootstrap_sha256_v1` interval over the
authenticated schedule and apply the policy ceiling to its upper bound. A
scorer-extension comparison uses the same fixed paired-resampling method over
unit-interval record values and applies `delta_min_pp` to the lower bound.
[Decision semantics](../assurance/decision-semantics.md) defines the exact
arithmetic.

Every metric policy contains its threshold and may also contain the coupled
`minimum_record_count` and maximum-width fields. Exact match and scorer
extensions use `maximum_interval_width_pp`; normalized NLL uses
`maximum_interval_width_ratio`. Exact match may independently add
`minimum_side_accuracy`, a finite value from `0` through `1` that both side
means must meet. A v3 report passes only when the metric-bound check and every
configured sample-qualification and side-accuracy check pass. Historical v2
reports have no side-accuracy section and retain their original semantics.

## Provider document field map

The schemas remain authoritative for types, patterns, nullability, and nested
conditional rules. This map makes every top-level contract field discoverable:

| Contract | Required top-level fields |
| --- | --- |
| Runtime capabilities | `format_version`, `provider_abi`, `provider_name`, `artifact_formats`, `tasks`, `metrics`, `execution_modes`, `required_extra`, `required_image` |
| Artifact identity | `format_version`, `artifact_format`, plus the format-specific identity fields listed in the runtime-provider reference |
| Scoring observation | `format_version`, `provider_name`, `artifact_identity_sha256`, `schedule_sha256`, `records`, `aggregate_source_sha256` |
| Provider receipt | `format_version`, `plugin`, `backend`, `capabilities`, `artifact_identity`, `execution_settings`, `device`, `outer_image_digest`, `scoring_observation_sha256` |
| Runtime manifest | `manifest_version`, `generated_at_utc`, `verifier_contract_version`, `report`, `config`, `execution_mode`, `outer_container`, `runtime_provider` |
| Behavioral schedule | `format_version`, `task`, `dataset_identity`, `records` |

Nested provider-receipt groups are closed:

| Group | Fields |
| --- | --- |
| `plugin` | provider name, distribution, distribution version, provider ABI |
| `backend` | name, version, and at least one source, binary, or build SHA-256 |
| `execution_settings` | seed, context length, batch size, output limit, timeout, network permission |
| `device` | kind, name, optional compute capability, driver, and CUDA runtime |
| `capabilities` | the complete capability object above |
| `artifact_identity` | one exact HF, GGUF, or TensorRT-LLM identity variant |

Each scoring record requires `record_id`, `input_sha256`, and `status`. An `ok`
record contains output text plus its digest, log-likelihood facts, or both. An
`error` record contains only a canonical `error_code` and no measured facts.
Log-likelihood facts are finite `logprob_sum`, positive `token_count`, and
positive `utf8_byte_count`. Byte-normalized NLL verifies the byte count against
the scheduled target. Matching authenticated tokenizer metadata and equal
positive paired token counts allow a verifier-derived perplexity interpretation
without adding a metric or policy surface.

The runtime manifest fixes `execution_mode` to `container`. Its outer-container
object binds the image reference/digest and observed execution switches; its
runtime-provider object binds the provider name, ABI, and sibling identity,
observation, and receipt files. `report` and `config` bind their sibling files
by digest. Unknown fields at any of these levels are rejected.

## Closed formats without standalone schemas

Several exact shapes are enforced by code and verifier replay rather than a
separate JSON Schema file:

| Format | Role |
| --- | --- |
| `invarlock/evidence-input-identity-v1` | One input role, material digest, and optional locator/media type |
| `invarlock/paired-records-v1` | Verifier-derived baseline/subject scores in schedule order |
| `invarlock/runtime-side-report-v1` | Minimal link from one side to its provider observation |
| `invarlock/scorer-extension-descriptor-v1` | One scorer's capabilities, input facts, result semantics, and trust constraints |
| `invarlock/scorer-extension-binding-v1` | Exact scorer identity and canonical configuration selected by the request |
| `invarlock/scorer-extension-result-v1` | Ordered unit-interval record results and core-owned arithmetic mean from replay |
| `invarlock/comparison-report-v3` | Current canonical means, point comparison, metric-specific paired interval, optional sample and exact-match side-accuracy qualification, threshold, and verdict |
| `invarlock/comparison-report-v2` | Historical canonical report without side-accuracy qualification; accepted for backward verification, not emitted for new evaluations |
| `invarlock/comparison-report-v1` | Legacy canonical report replayed with its original exact-match interval method; accepted for backward verification, not emitted for new evaluations |
| `invarlock/evidence-pack-signature-v1` | Ed25519 signature over canonical `manifest.json` bytes |
| `invarlock/evidence-pack-verify-v1` | Machine-readable independent verification result |
| `invarlock/evidence-verification-receipt-v1` | Signed statement binding the pack, artifact/schedule/policy/runtime/signer anchors, verifier, optional trust-profile digest, and verdict |
| `invarlock/evidence-verification-receipt-v2` | The v1 statement plus an independently supplied normalized-request digest; emitted when that anchor is present and required for GGUF evidence |
| `invarlock/evidence-verification-receipt-signature-v1` | Ed25519 envelope for the receipt statement |

These are documented for inspection and interchange, not as permission to
construct partial objects. Use `evaluate` to produce bundles, `verify` to
produce receipts, and the stable Python
`verify_signed_verification_receipt` facade to validate a received receipt.

## Canonical JSON and digests

Canonical JSON uses UTF-8, sorted object keys, compact separators, and finite
numbers. Newline behavior is contract-specific: most bundle JSON documents and
signed receipt statements use one trailing line feed, while canonical schedule,
artifact-identity, scoring-observation, and embedded substructure hash inputs
use the same representation without a final line feed. Writers and readers
must use the contract serializer rather than normalize whitespace themselves.

The base JSON syntax follows [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259).
InvarLock deliberately narrows it with unique keys, finite numbers, closed
schemas, exact encodings, and semantic cross-replay.

Bundle-level digests use lowercase `sha256:<64 hex>`. Some provider ABI fields
and the checksum-file digest use a bare lowercase 64-character SHA-256 value;
their schemas identify that distinction. Signatures use Ed25519. A key
fingerprint is `sha256:` followed by the SHA-256 of the raw Ed25519 public key.
Ed25519 is specified by [RFC 8032](https://www.rfc-editor.org/rfc/rfc8032).

## Digest and signature dependency chain

```text
input/runtime bytes -> typed identity or material digest
provider observation + receipt + runtime manifest -> side report
schedule + both observations -> paired records -> comparison report
all payload bytes -> checksums.sha256
payload references + checksum-file digest -> manifest.json
canonical manifest bytes -> Ed25519 evidence signature
manifest digest + independent anchors (+ GGUF request digest) + verdict -> verifier-signed receipt
```

A later layer authenticates references to earlier layers; it does not replace
their semantic checks. In particular, a valid evidence signature proves that
one key signed the manifest, not that artifact identities, scores, policy, or
runtime declarations are true. Independent verification supplies and checks
those acceptance anchors.

## Semantic validation

Schema validity is only the first layer. The verifier also recomputes:

- fixed path and closed-inventory rules;
- canonical bytes and file digests;
- artifact, schedule, runtime, observation, and request cross-bindings;
- record order, input digests, and observation-record digests;
- exact-match or normalized-NLL scores, or explicitly authorized scorer-
  extension results replayed twice from authenticated text facts;
- paired exact-match counts, exact McNemar probability, and the report-version-specific Newcombe interval,
  or deterministic paired replicates and interval endpoints for normalized NLL
  and scorer-extension deltas;
- derived perplexity facts when tokenizer and paired token counts are
  comparable;
- comparison means, threshold arithmetic, optional count/interval-width and
  exact-match side-accuracy qualification, and policy verdict; and
- evidence signer and verifier signature bindings.

A custom reader that performs schema validation alone is not equivalent to
`invarlock verify`.

## Version and change discipline

A format identifier names one exact shape and interpretation. Additive fields
are not silently accepted by closed objects. A breaking artifact change requires
a new format identifier and explicit reader behavior. Runtime providers must
also match ABI `1` exactly.

The v3 comparison report is the current writer format. Strict verification
continues to reconstruct v2 and v1 reports under their original shapes and
arithmetic when signed historical packs identify those formats; it never
relabels a reconstructed report as another version.

Receipt v1 remains valid for existing evidence whose runtime does not require a
request-level executable binding. A supplied request digest selects receipt v2.
GGUF verification requires that external request anchor because the normalized
request authorizes the exact llama.cpp binary, source, version, execution
settings, and GGUF identity reconciled with provider evidence.

Core and first-party add-ins are released at the same package version. Provider
add-ins declare the exact coordinated core release, while the provider ABI remains
the runtime compatibility gate. See [Release verification](release-verification.md).

## Related documentation

- [Evidence artifacts](artifacts.md) maps contract roles to fixed bundle paths.
- [Runtime providers](runtime-providers.md) defines the typed ABI that produces
  provider contract objects.
- [Reports and receipts](reports.md) explains the decision and independent
  verification objects.
- [Release verification](release-verification.md) separates package versioning
  from evidence-format and provider-ABI compatibility.
