# Pipeline contracts

!!! info "Reference"

    **Surface:** `invarlock.pipeline`, `invarlock-pipeline`, and five packaged
    `pipeline_*.schema.json` contracts.

    **Stability:** Unreleased v1 formats. Existing core evidence contracts and
    retained verification transactions keep their existing semantics.

    **Use this page when:** Mapping evaluator exports, defining metrics or
    reviewing what a pipeline signature and decision establish.

## Records and identities

`make_run` creates `invarlock/pipeline-run-v1`. Required run identity fields are
`source` (name and version), `run_id`, `artifact_digest`, `source_digest`,
`score_provenance` and `records`. The SDK defaults `source_digest` to null; native
imports hash the original export bytes. This records provenance without
claiming that a self-reported source identity is authenticated.

Every record has an `id`, `input`, `expected`, `output`, `context`, `scores`,
`metadata` and `error`. The SDK fills empty context, scores and metadata and a
null error. `input`, `expected`, `output` and `context` accept finite JSON values;
scores are finite numbers and slice tags are strings. Top-level and record
structures are closed. Unknown fields must be mapped explicitly.

Both runs must contain exactly the same unique IDs. Pairing sorts IDs, so export
row order may differ. Inputs, expected values and metadata must be canonically
identical per pair. Context may differ because it captures release-dependent
prompts, retrieved material or decoding settings. It is still covered by the
complete run digest and signature. These checks prevent inconsistent pairing;
they do not prove that either evaluation operator selected a representative schedule.

A file or normalized run is at most 64 MiB and 10,000 records. Complete embedded
evidence is at most 192 MiB. Policies contain at most 16 metrics and 16 named
slices, plus the automatically checked `overall` slice. Comparisons use ordinary
CPU memory and require no evaluator dependency. Native exports larger than these
limits need an explicit supported projection before import.

## Pinning reusable inputs

Each `pipeline-project-v1` side may specify `expected_run_digest`, a lowercase
`sha256:` digest of the complete normalized run. Compute it from a reviewed
normalized export with `invarlock-pipeline digest baseline.json --run`, then
retain it in the reviewed project:

```json
{
  "path": "baseline.json",
  "adapter": "invarlock",
  "expected_run_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
}
```

Replace the placeholder with the actual reviewed digest. The check happens after
loading and normalization, before signing or publishing results. `--baseline`
and `--candidate` override the path but retain its expected digest. A mismatch
returns integration-error exit code 2 and publishes no result directory.
Omitting the pin preserves ordinary comparisons of changing CI exports.

For native adapters, normalize the export using the same source, artifact and
score-provenance settings before obtaining the run digest. The pin includes the
original export's `source_digest`, all run identities, records and their order.
Reformatting an already normalized JSON file preserves its digest; changing
native export bytes changes its provenance and therefore its normalized digest.

Pins let a later workflow reuse an approved run without silently substituting
its contents. Approve and retain the expected digest independently of the input
being checked. A pin computed from a replacement that has not been reviewed is no assurance
of its identity. Pins do not establish representative sampling, truthful execution
or semantic correctness. They also do not define an intended case set before
capture; two matching exports can both omit intended cases unless the capture
protocol independently checks coverage. Keep separately captured workflows and
phases separately identified; do not splice incompatible captures into an older
signed run.

## Native adapters

| Adapter | Accepted native shape | Pairing and scoring limits |
| --- | --- | --- |
| `invarlock` | Normalized pipeline run JSON | Validates closed schema; identity overrides are rejected |
| `jsonl` | One SDK record per line | Caller supplies stable inputs, references, context and explicit score semantics |
| `inspect-json` | JSON EvalLog version 1 or 2 | Successful run, one completion, one target and first epoch; numeric scores or explicit `C`/`I`; non-text completions rejected |
| `lm-eval-samples` | Generation `--log_samples` JSONL | Requires `doc`, `doc_id`, `target`, `arguments`, `filtered_resps`; document is stable input, arguments are context; exactly one textual filtered response |
| `promptfoo-jsonl` | Full result rows as JSONL | Requires `testCase.vars`, rendered `prompt`, `testIdx`, `promptIdx`; variables are input and prompt is context; target comes only from `testCase.metadata.invarlock_expected` |

Promptfoo compact exports without original variables and prompts cannot support
these pairing checks. Supply full rows or map through the SDK. Keep the prompt
index mapping stable when comparing variants. LM Evaluation Harness multiple
choice/log-likelihood outputs, multiple generations, agent traces and
multimodal response content need explicit adapters; the generation parser does
not infer their meaning. Inspect metadata exposes only string values as slices.

Promptfoo's typed `failureReason` distinguishes ordinary failed assertions (`1`)
from provider or grading execution failures (`2`). A consistent failed assertion
retains its response for quality scoring even though the native row also uses
`error` for the assertion's explanation. Conflicting success, score, reason or
response fields fail import. Execution failures remain upstream errors; exports
without a typed reason conservatively treat a nonempty error as an upstream
error. These rules are exercised against a native Promptfoo 0.121.19 capture.

Native recorded scores never inherit deterministic replay authority from the
adapter name. The policy must approve their source, version, unit and, for
judgments, rubric digest. Source metadata is a declaration until your own
capture and signing controls establish it.

## Deterministic scorer semantics

All shipped deterministic metrics produce higher-is-better scores in `[0,1]`.

- `exact_match`: literal equality of two strings; no normalization.
- `normalized_match`: Unicode NFKC, collapsed whitespace and case folding by
  default. Set `configuration.casefold` to false for case-sensitive matching.
- `numeric_tolerance`: finite numeric values or numeric strings. Correct when
  absolute difference is at most the larger of `absolute` and
  `relative * abs(reference)`. Defaults are zero. Invalid candidate answers
  score zero; invalid references or tolerances fail the integration. Arithmetic
  preserves decimal distinctions up to 256 significant digits and adjusted
  exponents from -1000 through 1000. Export high-precision values as strings to
  avoid rounding in your evaluator's JSON number serialization.
- `json_fields`: average literal canonical JSON equality at the unique RFC 6901
  pointers in `configuration.fields`. Missing candidate fields score zero;
  missing reference fields fail. Duplicate JSON keys are rejected. Numeric
  representation and JSON types are preserved; `true` does not equal `1`.
- `token_f1`: multiset precision/recall F1 over normalized whitespace tokens.
  Punctuation remains part of each token. Two empty strings score one.

`normalized_match` and `token_f1` require an explicit
`configuration.unicode_version`, such as `"15.0.0"` for Python 3.12. Normalization,
case folding and whitespace classification use the executing Python runtime's
Unicode tables. The declared version must equal `unicodedata.unidata_version`;
a missing or different version is an integration error before scoring. This
requirement also applies when `casefold` is false. Recipient verification needs
a runtime with the policy's Unicode version. Changing the version changes the
policy and scorer binding; it must not silently rescore historical evidence.
`invarlock-pipeline init` records the local version explicitly in its generated
policy. The other deterministic and recorded metrics do not take this setting.

The four additional scorers are also available in the core transaction registry
as `invarlock.normalized_match`, `invarlock.numeric_tolerance`,
`invarlock.json_fields` and `invarlock.token_f1`, version `1.0.0`. They require an
explicit scorer binding and do not require enabling installed third-party
scorers. See [Scorer extensions](api-guide.md) for the binding contract.

Deterministic evaluator qualification accepts the same metric kinds. Non-exact
qualification requires each schedule row's independent `reference_output` and
its matching `reference_output_sha256`. Every reported score must equal local
recomputation. The qualification result's `scorer_binding()` provides the shared
binding for runtime import. Retained upstream matrix rows still demonstrate
the exact metric recorded in each existing profile.

## Policy and statistics

Each metric declares `name`, `kind`, `configuration`, `direction`, `unit`,
`aggregation: mean`, `minimum_count`, `maximum_regression` and
`maximum_interval_width`. Optional `candidate_minimum` and `candidate_maximum`
apply to the observed candidate mean. Units apply to means, deltas, intervals
and thresholds; a score delta of `0.02` is two percentage points. Recorded
metrics additionally declare `score_key` and `accepted_provenance`.

Binary shipped scorers use the existing paired Newcombe interval in score units.
Continuous scores use 2,048 deterministic paired bootstrap replicates, percentile
interpolation at 2.5% and 97.5%, and a SHAKE256 stream seeded by the paired
schedule. Constant observed differences give a zero-width bootstrap interval.
That does not establish zero population uncertainty. Both methods assume an
appropriate paired sampling design; resampling individual cases is unsuitable
for dependent conversation turns unless those turns are aggregated into
independent cases before capture.

For higher-is-better metrics, the lower delta bound must be at least the negative
allowed regression. For lower-is-better metrics, the upper bound must be no
larger than the allowed regression. Missing results or too few cases produce
`insufficient_evidence`. Failing relative or absolute bounds produce
`regression`; otherwise excessive interval width produces
`insufficient_evidence`. Every metric is checked in every named slice using an
exact conjunction of metadata tags. An empty slice never passes.

Overall precedence is `regression`, then `insufficient_evidence`, then `pass`.
Intervals are marginal, not a simultaneous confidence guarantee across all
metrics and slices. There is no correction for repeated candidate selection,
no automatic sampling design and no promise that a policy pass establishes
safety, compliance, semantic correctness or faithful model execution.

## Evidence, verification and reports

`invarlock/pipeline-evidence-v1` embeds both normalized runs, the full policy and
the exact comparison. Optional Ed25519 signatures bind canonical JSON under a
distinct domain. The format carries no embedded trusted public key.

Independent verification requires a recipient-owned key, full run digests and
policy. It rejects unsigned evidence, changed expected inputs, changed policy,
invalid signatures and even a signed comparison whose arithmetic does not
replay. Authentication of a recorded judge score establishes its attribution
under the caller's trust setup; it cannot turn that judgment into a recomputed
fact. This format is separate from the core runtime evidence and signed receipt.

The CLI atomically publishes completed result and key directories without
replacing existing destinations. Files are owner-readable/writable and new
directories are private. HTML escapes data and runs no scripts; Markdown escapes
untrusted labels; JUnit reports policy failures and insufficient evidence as
failures and errors respectively. Human reports present the comparison;
signature authentication requires the separate verify operation.
