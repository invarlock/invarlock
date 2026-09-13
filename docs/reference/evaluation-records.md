# Evaluation Records

!!! info "Reference"
    **Surface:** Captured evaluation records and evidence bindings.
    **Stability:** Public core contract for captured evaluation.
    **Use this page when:** Implementing or reviewing captured evidence.

Captured comparisons use neutral evaluation record contracts. A record keeps
its stable ID, input, expected value, output, score provenance, metadata, and
error state. Baseline and subject records must use the same complete schedule.

The public request contract is `invarlock/evaluation-request-v2`. Deterministic
and recorded-score captured comparisons use directory pack
`invarlock/evidence-pack-v2` and an external
`invarlock/evidence-verification-receipt-v3`. The pack has one manifest signature,
not a second signed record envelope. `verify` checks recipient-owned policy,
complete-run and normalized-request digests, and signer identity. `report` is a
deterministic presentation step and does not confer verification authority.

The captured comparison binds `bindings.baseline`, `bindings.subject`, and
`bindings.policy`. Metric results use `baseline_mean` and `subject_mean`; absolute
policy bounds use `subject_minimum` and `subject_maximum`. The captured SDK helper
is `compare_runs(baseline=..., subject=..., policy=...)`, with no alternate role
keywords or wire-field aliases. These captured fields do not rename
native historical reports or third-party exports. Native pack v1 and receipt
v1/v2 contracts are unchanged. Selecting `comparison.metric: judge` instead
publishes judge evidence with its own measurement and recipient trust contracts;
see [judge measurements](judge-measurements.md).

For the exact JSON schemas, see the files under `contracts/` and their synced
copies under `src/invarlock/_data/contracts/`.

## Record contracts

| Schema | Wire identifier | Meaning |
| --- | --- | --- |
| `evaluation_run.schema.json` | `invarlock/evaluation-run-v1` | Complete canonical run with records and score provenance |
| `evaluation_case_set.schema.json` | `invarlock/evaluation-case-set-v1` | Independently reviewed IDs, inputs, expected values and metadata |
| `comparison_policy.schema.json` | `invarlock/comparison-policy-v1` | Metric, slice, sample, interval and subject-bound requirements |
| `multi_metric_comparison.schema.json` | `invarlock/multi-metric-comparison-v1` | Deterministically replayed paired results and three-way decision |
| `normalized_captured_request.schema.json` | `invarlock/evaluation-request-v2` | Path-free portable request identity stored inside the captured pack |

Runs require `format`, `source` (`name`, `version`), `run_id`, `artifact_digest`,
`source_digest`, `score_provenance`, and `records`. Every record has `id`, `input`,
`expected`, `output`, `scores`, `metadata`, `error`, and `context`. Null output or
an explicit error is retained, not silently dropped. Likelihood scoring can use
a null output when valid reference-continuation facts are present. Matching IDs
must bind matching input, reference and metadata. Missing required scorer facts
produce `insufficient_evidence`, rather than a pass computed over survivors. Duplicate
IDs or changed paired facts are integration errors.

`artifact_digest` attributes the run to a supplied artifact identity; it does not
prove inference. `source_digest` binds raw imported bytes when applicable and is
part of the complete run. Canonical `run_digest` covers the whole normalized run,
including provenance and all records, not just its model or schedule. Do not
substitute a raw export checksum for a complete-run pin.

`load_run` supports `invarlock`, `jsonl`, `inspect-json`, `lm-eval-samples`, and
`promptfoo-jsonl`. These adapters normalize recorded outputs, never register a
runtime provider or import execution authority. Original upstream bytes and
their `source_digest` remain unchanged. A canonical `invarlock` run supplies its
own metadata; external adapters require explicit source, run ID and artifact
identity, with approved score provenance when recorded metrics are used.

## Capture and explicit input projections

`invarlock.engine.capture_evaluator_run` accepts explicitly mapped per-case rows
from any evaluator, with its actual source name/version, run ID and supplied
artifact digest. `evaluator_input_capabilities` reports usable counts and
unavailable IDs for `exact_match`, `normalized_nll_per_utf8_byte` and `judge`.
Availability describes input facts; it grants neither qualification nor proof
of an upstream scorer execution.

For a structured input, `input_projection` may select an existing string with
`{"kind": "json-pointer", "pointer": "/input/question"}`. Pointers must be
rooted at `/input` or `/context` and select a string. The capture retains the original input and
context plus the exact projection binding. Paired runs must agree on the
original input and projection configuration as well as the selected text.
Objects and arrays are never implicitly converted to text.
External adapter sources may declare the same projection in a captured request.
Canonical `adapter: invarlock` runs already contain their bindings and reject
source overrides, including a second projection.

Set `comparison.metric` to `exact_match`, `normalized_nll_per_utf8_byte` or
`judge` to select a scorer explicitly. Exact match and NLL must agree with the
single metric in the comparison policy. Omit this selector for an ordinary
multi-metric comparison. Judge uses a separate recipe and retained-call path.

## Captured reference likelihoods

A record may additionally retain `likelihood` with these closed fields:

| Field | Meaning |
| --- | --- |
| `basis` | Exactly `reference_continuation` |
| `logprob_sum` | Finite, zero or negative natural-log probability sum of the reference continuation |
| `token_count` | Positive integer count of measured reference tokens |
| `utf8_byte_count` | Positive integer equal to the reference string's UTF-8 byte length |
| `input_digest`, `reference_digest` | Canonical digests of the original input and reference |
| `artifact_digest`, `source` | Exact evaluated artifact and evaluator identity from the run |
| `configuration_digest`, `tokenizer_digest` | Pins for the likelihood measurement configuration and tokenizer |

These facts must originate from an actual likelihood measurement. Generated
answer probabilities and summary losses have different semantics. With an input
projection, `input_digest` still identifies the preserved original input.

The NLL policy metric uses `direction: lower`, `unit: nats_per_utf8_byte`,
`aggregation: mean` and `ratio_max` in place of `maximum_regression`. Its
configuration requires `configuration_digest`, `baseline_tokenizer_digest` and
`subject_tokenizer_digest`. The score per record is `-logprob_sum / utf8_byte_count`;
the comparison is the ratio of subject and baseline means, with the same paired
resampling method as native NLL. Interval width is measured in ratio units.
Reference and configuration bindings and the declared tokenizer identities are
checked before scoring. Replay does not execute the model or tokenizer, so their
measurement claims remain source assertions. The separate
[real Harness likelihood reference](https://github.com/invarlock/invarlock/blob/main/examples/captured-results/references/harness-likelihood/README.md)
retains six same-model CPU pairs; synthetic contract tests and historical
exact-match qualifications do not establish other native likelihood profiles.

Judge selection uses the separate judge recipe and retained measurement contract
through the same captured request. It does not reinterpret a `recorded` scalar
score as a native judge trial. `prompt.reference_mode: per_case` explicitly adds
the frozen string reference to each judge request as a separate field. Omitted
or `none` preserves the reference-free behavior. The reference never becomes part
of the evaluated model input. See [judge measurements](judge-measurements.md).

## Policy and identity

Policies contain 1..16 metrics and up to 16 metadata slices in addition to the
overall scope. Each metric selects a unique name, kind, closed configuration,
direction (`higher`/`lower`), unit, mean aggregation, minimum count, maximum
regression and maximum interval width; NLL replaces maximum regression with
`ratio_max`. Optional absolute limits on the subject mean are
`subject_minimum` and `subject_maximum`.

Built-in captured kinds are `exact_match`, `normalized_match`,
`numeric_tolerance`, `json_exact`, `json_fields`, `token_f1`, and
`normalized_nll_per_utf8_byte`. Exact match and normalized NLL share native
scorer semantics while retaining captured-input assurance. `normalized_match` and `token_f1` pin
`unicode_version`; a recipient with an incompatible Unicode environment refuses
replay without issuing a receipt. `recorded` instead selects a `score_key` and
exact `accepted_provenance` (`kind`, `source`, `version`, `unit`, `rubric_digest`).
The verifier authenticates attributed values and paired arithmetic, not the
judge, human, measurement apparatus or external metric implementation.

An independent planned case set can be pinned in policy as
`expected_case_set_digest`. Use `freeze_case_set`, `case_set_digest`, and
`validate_run_case_set` to require the reviewed schedule, not merely agreement
between two possibly incomplete captures.

`normalize_captured_request` strips physical source/output paths and binds each
adapter, complete run digest, explicit `expected_run_digest` when authored,
source overrides, and policy digest. `captured_request_digest` hashes those
canonical portable bytes. An explicit run pin is semantic and is not dropped
even when equal to the derived run digest. Relocating a request tree changes no
portable identity; changing a source, provenance, policy or pin can change it.
Use these SDK helpers, not a handwritten YAML/JSON hash recipe.

## Pack boundary

Captured pack v2 has exactly `manifest.json`, `checksums.sha256`, `request.json`,
`inputs/policy.json`, `records/baseline.json`, `records/subject.json`, and
`reports/evaluation.report.json`. Signed packs also have
`manifest.signature.json`; `authentication: unsigned_local` packs have no
signature and a null signing-key fingerprint. The signature uses the existing
`invarlock/evidence-pack-signature-v1` envelope over the canonical manifest.
There is no second signed comparison envelope, embedded verification receipt,
or implicit legacy migration.

The normalized request, policy, records and report are canonical JSON with a
final LF. The checksum ledger uses fixed payload paths. `comparison_id` hashes
the canonical object containing `kind: captured`, the normalized request digest,
both run digests and the policy digest; it identifies the comparison inputs.
Extra files/directories, path traversal, symlinks, duplicate keys, non-finite
numbers and byte-limit violations are rejected. Payloads must be regular files;
verification checks a stable snapshot and validates their digests.
Publication is atomic and no-clobber. Keep receipts and rendered outputs outside
the immutable pack. See [capacity](evaluation-capacity.md),
[reports and receipt scopes](reports.md), and the
[signed trust-profile flow](../user-guide/captured-results.md#signed-handoff).
