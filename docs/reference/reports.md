# Reports and receipts

InvarLock separates provider facts, a canonical comparison report, independent
verification output, a signed verification receipt, and console or HTML summaries.
They have different trust meanings.

> **Reference**
>
> **Surface:** Runtime-side reports, canonical comparison reports,
> verification results, signed receipts, and console or HTML summaries
>
> **Stability:** Versioned machine contracts are stable by format; console
> and HTML output are presentation surfaces
>
> **Use this page when:** Parsing a decision, validating a receipt, or
> distinguishing signed evidence from independently accepted evidence

Report formats depend on the selected scorer and evidence family. The native
runtime sections below describe exact match, normalized NLL and deterministic
extensions. Captured comparisons use the multi-metric report described in
[evaluation records](evaluation-records.md). Native and captured `judge` requests
produce judge analysis and verification results described in
[judge measurements](judge-measurements.md); they do not use the native
`invarlock/comparison-report-v3` shape. All are rendered by `invarlock report`.

## Report appearance

HTML reports open in light mode by default. The **Dark mode** button records an
explicit choice for reports on the same browser origin. Reports do not switch
based on the operating system theme. Documentation preferences remain separate.
Without JavaScript, reports stay light and all evidence remains readable. If
browser storage is unavailable, the button still works for the current page.
Printing always uses the light palette and hides the theme control.

## Runtime-side report

Each side has an `invarlock/runtime-side-report-v1` object:

```json
{
  "format": "invarlock/runtime-side-report-v1",
  "provider": "hf_transformers",
  "artifact_identity_sha256": "...",
  "scoring_observation_sha256": "...",
  "schedule_sha256": "...",
  "record_count": 20
}
```

It binds one side to its artifact identity, scoring observation, provider, and
schedule. Detailed settings and runtime identity remain in the provider
receipt, observation, artifact identity, and runtime manifest. The verifier
reconstructs the runtime-side object exactly.

## Canonical comparison report

Native exact-match/NLL and deterministic-extension evaluations write
`reports/evaluation.report.json` as an
`invarlock/comparison-report-v3` object. Strict verification also accepts
signed `invarlock/comparison-report-v2` and `invarlock/comparison-report-v1`
objects. Version 2 omits side-accuracy qualification; version 1 additionally
uses the legacy exact-match interval method. A verifier never silently upgrades
the arithmetic or policy meaning of an existing pack.

The current report contains:

- `comparison_id`, `metric`, and `record_count`;
- baseline and subject `mean_score`;
- one metric-specific `comparison` object;
- one metric-specific `uncertainty` object;
- `paired_binary` for exact match, `derived_measurements` for normalized NLL,
  or `scorer_extension` and `scorer_replay` for an authorized extension;
- `policy_digest`;
- optional `sample_qualification` when the policy binds count and precision
  requirements;
- optional `side_accuracy` when exact-match policy binds a minimum accuracy for
  both sides; and
- `verdict`, either `pass` or `fail`.

A complete exact-match report has this shape:

```json
{
  "baseline": {"mean_score": 0.75},
  "comparison": {
    "kind": "exact_match_delta_pp",
    "minimum": -15.0,
    "value": 5.0
  },
  "comparison_id": "cmp-example",
  "format": "invarlock/comparison-report-v3",
  "metric": "exact_match",
  "policy_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
  "record_count": 20,
  "subject": {"mean_score": 0.8},
  "paired_binary": {
    "baseline_pass_subject_fail": 1,
    "baseline_fail_subject_pass": 2,
    "both_pass": 14,
    "both_fail": 3,
    "discordant_pairs": 3,
    "mcnemar_exact_two_sided_p_value": 1.0,
    "effect_size_pp": 5.0,
    "effect_size_confidence_interval": {
      "method": "newcombe_hybrid_score_paired_v2",
      "confidence_level": 0.95,
      "lower_pp": -14.975805254302834,
      "upper_pp": 24.866848542801065
    }
  },
  "sample_qualification": {
    "record_count": {"minimum": 20, "observed": 20, "passed": true},
    "interval_width": {
      "maximum": 50.0,
      "observed": 39.8426537971039,
      "unit": "percentage_points",
      "passed": true
    },
    "passed": true
  },
  "uncertainty": {
    "interval_mass": 0.95,
    "lower": -14.975805254302834,
    "method": "newcombe_hybrid_score_paired_v2",
    "scope": "paired_binary_outcomes",
    "upper": 24.866848542801065
  },
  "verdict": "pass"
}
```

Values are illustrative. Verification reconstructs the whole object and
requires canonical equality. It does not independently trust stored means,
point comparison, interval bounds, threshold, or verdict.

### Exact match

```json
{
  "kind": "exact_match_delta_pp",
  "value": 1.5,
  "minimum": -2.0
}
```

`value = 100 * (subject_mean - baseline_mean)`. The metric-bound check passes when
`uncertainty.lower >= comparison.minimum`, where `minimum` comes from
`resolved_policy.metrics.exact_match.delta_min_pp`. If sample qualification is
present, its count and width checks must also pass. If side-accuracy
qualification is present, both side means must meet its minimum. The final
verdict is the conjunction of all applicable checks.

For this metric, baseline and subject `mean_score` are literal exact-match
accuracies between `0` and `1`. Comparison and interval bounds are percentage
points between `-100` and `100`.

`paired_binary` reports baseline-pass to subject-fail regressions,
baseline-fail to subject-pass improvements, both-pass and both-fail counts, the
number of discordant pairs, and the exact two-sided McNemar probability. Its
effect size is the same subject-minus-baseline percentage-point delta. The
continuity-corrected paired Newcombe hybrid-score 95% interval is repeated as
the canonical `uncertainty` object in a v2 or v3 report. The policy uses its
lower bound; the McNemar probability does not control the verdict. A v1 report keeps
method `newcombe_hybrid_score_paired_v1` and is replayed only with that original
method.

### Sample and precision qualification

A metric policy may omit sample qualification. If it enables qualification,
the two fields must be supplied together:

- `minimum_record_count` plus `maximum_interval_width_pp` for exact match or an
  authorized scorer extension; or
- `minimum_record_count` plus `maximum_interval_width_ratio` for normalized
  NLL.

`minimum_record_count` is an integer from 1 through 10,000. Percentage-point
width must be positive and at most 200; ratio width must be positive. The
report records the observed width as `uncertainty.upper - uncertainty.lower`.
Its final verdict is:

```text
metric bound passes
and observed record count >= minimum record count
and observed interval width <= maximum interval width
```

The report therefore distinguishes a metric rejection from insufficient count
or precision without allowing a favorable point value to bypass either gate.

### Side-accuracy qualification

An exact-match policy may independently set `minimum_side_accuracy` from `0`
through `1`. A v3 report then includes `side_accuracy` with the configured
minimum, each side's observed exact-match mean and pass result, and a combined
`passed` result. Both sides must meet the inclusive floor. This check is
independent of the coupled sample controls, and the final verdict also requires
it to pass. Historical v2 reports do not contain this section.

### Normalized NLL per UTF-8 byte

```json
{
  "kind": "normalized_nll_ratio",
  "value": 0.98,
  "maximum": 1.05
}
```

Each side's `mean_score` is the arithmetic mean of verifier-derived per-record
teacher-forced expected-continuation
`-logprob_sum / utf8_byte_count`. The baseline mean must be positive and
`value = subject_mean / baseline_mean`. The metric-bound check passes when
`uncertainty.upper <= comparison.maximum`, where `maximum` comes from
`resolved_policy.metrics.normalized_nll_per_utf8_byte.ratio_max`.

This is a ratio of equally weighted record means, not pooled NLL, byte-weighted
loss, or perplexity. It measures expected-continuation likelihood regression
under authenticated inputs and runtime, not general model quality.

### Derived perplexity interpretation

```json
{
  "perplexity_ratio": {
    "status": "available",
    "basis": "authenticated_target_likelihood",
    "method": "target_token_weighted_perplexity_ratio_v1",
    "tokenizer_metadata_sha256": "...",
    "target_token_count": 4096,
    "baseline_perplexity": 12.3,
    "subject_perplexity": 12.6,
    "ratio": 1.0243902439
  }
}
```

For normalized NLL, `derived_measurements.perplexity_ratio` is available when
both artifacts bind matching authenticated tokenizer metadata and every pair
has the same positive target-token count. The verifier derives both side
perplexities and their ratio from authenticated target log probabilities. When
the facts are not comparable, the object carries `status: unavailable` and a
closed reason instead.

This object is interpretation only. It is not a metric or policy input and has
no threshold, confidence interval, or verdict authority.

### Normalized-NLL paired resampling interval

Every normalized-NLL comparison report contains:

```json
{
  "method": "paired_percentile_bootstrap_sha256_v1",
  "scope": "authenticated_schedule",
  "interval_mass": 0.95,
  "replicates": 2048,
  "lower": 0.98,
  "upper": 1.04
}
```

Index draws are derived from SHA-256 of the authenticated schedule digest,
replicate number, and draw number. Baseline and subject scores at each selected
schedule position are resampled together. The complete comparison is recomputed
2,048 times; `lower` and `upper` are linearly interpolated 2.5th and 97.5th
percentiles. A one-record schedule yields a point interval.

The interval and verdict are deterministic verifier-replay outputs. The scope
`authenticated_schedule` is literal: this is a finite-schedule stability
interval, not a population confidence interval or representativeness claim.

### Authorized deterministic scorer extension

An extension report uses:

```json
{
  "kind": "scorer_extension_delta_pp",
  "value": 2.5,
  "minimum": -1.0
}
```

For each side, an explicitly authorized deterministic scorer replays exactly
the authenticated expected output, output text, and output digest for every
record and returns one higher-is-better value in `[0, 1]`. The engine owns the
arithmetic mean and computes
`value = 100 * (subject_mean - baseline_mean)`. It then applies the same fixed
2,048-replicate paired schedule-resampling method to the paired record values.
The metric-bound check passes only when
`uncertainty.lower >= comparison.minimum`, where
`minimum` comes from `resolved_policy.metrics.scorer_extension.delta_min_pp`.

`scorer_extension` records the bound scorer ID, version, descriptor digest,
and configuration digest. `scorer_replay` binds the per-side deterministic
replay results. Verification requires the caller to authorize the same scorer
in a `ScorerExtensionRegistry`, runs it twice, and reconstructs the complete
report. A stored scorer result is never accepted as an aggregate assertion.

The scorer-extension v1 contract supports deterministic text replay. Core ships
`invarlock.normalized_match`, `invarlock.numeric_tolerance`,
`invarlock.json_fields`, `invarlock.json_exact` and `invarlock.token_f1`; the CLI
can use these without installed-scorer authorization. SDK callers supply a
`ScorerExtensionRegistry(allow_installed=False)`. Other implementations, such as
a separately supplied VQA normalization scorer, require explicit authorization.
SQL or code execution, model-based semantic similarity,
network services or externally assigned ratings, and LLM judges are outside scorer-extension
acceptance replay. The built-in `judge` scorer replays its bounded measurement
contract and, for native run/import evidence, its retained runtime capture. Its
report shows both evaluated model identities, runtime settings and digests,
judge model/configuration, rubric, counts and uncertainty. Other recorded judge
results may be authenticated as observations, where they have no verdict authority.

## Verification result

Native deterministic `invarlock verify --json` emits an `invarlock/evidence-pack-verify-v1` result. Important
fields include:

| Field | Meaning |
| --- | --- |
| `ok` | Integrity passed and the replayed policy verdict is not `fail` |
| `integrity_ok` | Structural, signature, digest, cross-binding, and semantic replay passed |
| `reports_verified` | The canonical report, including interval and verdict, matched verifier replay |
| `verification_scope` | `paired_comparison` only after successful complete replay |
| `assurance_status` | `verified` or `not_verified` |
| `policy_verdict` | `pass`, `fail`, or unavailable when replay could not complete |
| `observations` | Authenticated observation IDs, kinds, scopes, and content digests; never verdict inputs |
| `authenticity` | Whether the evidence signer matched the independent signer anchor |
| `anchors` | Policy path/digest, artifact and schedule digests, runtime digests, and signer fingerprint supplied by the caller |
| `warnings`, `errors` | Closed diagnostic arrays |

The high-level transaction adds `signed_receipt`, `verifier_identity`, and
`verifier_fingerprint`. Profile-based verification also adds
`trust_profile_digest`. Stdout is useful process output; the separately signed
receipt is the portable verifier assertion.

Captured verification emits `invarlock/evidence-pack-verify-v2` with
`kind: captured`, `ok`, `integrity_ok`, `policy_verdict`, `decision`,
`replay_status`, `signed_receipt`, `pack_manifest_digest`, and `metric_summaries`.
A completed `regression` or `insufficient_evidence` decision has
`policy_verdict: fail`, `integrity_ok: true`, and `ok: false` (exit `7`).
Local work-budget or unsupported-scoring-environment refusal is incomplete
verification (exit `2`) without a receipt, not evidence corruption.

Do not infer acceptance from `integrity_ok` alone. Native deterministic acceptance
requires `assurance_status: verified`; captured acceptance requires
`replay_status: completed`. Both require status `0`, `ok: true`,
`integrity_ok: true`, exact anchors, and a valid receipt when the result crosses
a process boundary.

## Signed verification receipt

The receipt contains a `statement` and `signature`. The statement format is
`invarlock/evidence-verification-receipt-v1` for native evidence without a request
anchor; v2 additionally binds an independently supplied request digest. Example v1:

```json
{
  "format": "invarlock/evidence-verification-receipt-v1",
  "pack_manifest_digest": "sha256:...",
  "anchors": {
    "policy_digest": "sha256:...",
    "artifact_digests": {
      "baseline": "sha256:...",
      "subject": "sha256:..."
    },
    "schedule_digest": "sha256:...",
    "runtime_digests": {
      "baseline": "sha256:...",
      "subject": "sha256:..."
    },
    "pack_signer_fingerprint": "sha256:..."
  },
  "verifier": {
    "identity": "release-verifier",
    "signing_key_fingerprint": "sha256:...",
    "trust_profile_digest": "sha256:..."
  },
  "verdict": {
    "ok": true,
    "integrity_ok": true,
    "policy_verdict": "pass",
    "verification_status": 0
  }
}
```

The signature envelope is
`invarlock/evidence-verification-receipt-signature-v1`. It uses Ed25519, embeds
the verifier public key in PEM form, and signs canonical statement bytes. The
embedded key is verification material, not a trust source. A receipt reader
must still supply the expected verifier identity and fingerprint, pack, policy,
artifact digests, schedule digest, runtime digests, and expected evidence
signer. For profile-based verification, it must also supply the expected
canonical trust-profile digest. Explicit-option receipts record `null` for that
field.

A receipt is written for a completed policy or integrity rejection when the
transaction can form a safe statement, then the command exits nonzero. A
successful receipt is internally consistent only when `ok` and `integrity_ok`
are true, `policy_verdict` is not `fail`, and `verification_status` is zero.

Downstream readers use
`invarlock.engine.verify_signed_verification_receipt`. The stable API returns a
`ReceiptVerification`; acceptance requires its `ok` field to be true.

Captured packs use only `invarlock/evidence-verification-receipt-v3`. Its
`verification_scope: captured_comparison` cannot authorize native acceptance or
deployment. Anchors bind the expected complete baseline/subject runs, policy,
normalized request, and signer; `pack_manifest_digest` identifies the manifest
actually examined, even if it differs from those expectations. `replay_status`
is `not_started`, `failed`, or `completed`. `scoring_assurance` is an ordered
array of `{name, slice, kind, scoring_assurance}` entries with each assurance
`recomputed` or `recorded`, populated only for complete matching replay under
intact bindings; integrity rejection carries `null`, even after replay completed.
A signed rejection may record an unsigned input pack without authenticating it.
A valid signature on a rejection receipt may authenticate successfully while
`statement.verdict.ok` remains false. Receipt authentication does not replay
payloads or promote a rejection to acceptance. Native receipt v1/v2 semantics
and historical bytes remain unchanged.

| Caller-supplied receipt anchor | Compared with |
| --- | --- |
| Evidence pack directory | Receipt manifest digest and pack's canonical manifest |
| Policy file | Receipt policy digest and pack-bound policy identity |
| Baseline and subject artifact digests | Receipt anchors and both pack-bound artifact identities |
| Canonical schedule digest | Receipt anchor and pack-bound dataset identity |
| Baseline and subject runtime digests | Receipt anchors and both runtime manifests |
| Expected evidence-signer fingerprint | Receipt anchor and authenticated evidence signature |
| Expected verifier identity | Receipt statement identity |
| Expected verifier fingerprint | Embedded receipt public key and receipt signature |
| Expected trust-profile digest or `null` | Receipt verifier profile binding |

## Console and HTML reports

Render a controlled evaluation bundle for inspection:

```bash
invarlock report evidence/ --html comparison.html
invarlock report evidence/ --explain
invarlock report evidence/ --html comparison.html --json
```

`invarlock report` checks the closed inventory, checksums, canonical JSON and
embedded evidence signature before rendering. The console displays a formatted
summary; the self-contained HTML works offline and can be printed. The HTML
file must be outside the evidence pack. Rendering preserves every bundle byte.
`--json` emits a rendering result object, including the HTML path when requested;
it does not turn the report into an independent acceptance receipt.

Native default/HTML-only CLI calls retain the exact
`invarlock/evidence-report-v1` result (`ok`, `pack_manifest_digest`, and `html`,
which is `null` when not requested). All captured report calls, and native calls
requesting Markdown or JUnit, return `invarlock/evidence-report-v2`:

```json
{
  "format_version": "invarlock/evidence-report-v2",
  "kind": "captured",
  "ok": true,
  "pack_manifest_digest": "sha256:...",
  "requested_outputs": {"html": "report.html", "junit": "results.xml"},
  "written_outputs": {"html": "report.html", "junit": "results.xml"},
  "failed_output": null,
  "errors": []
}
```

`kind` is `runtime` for native v2 output. Destination maps can contain `html`,
`markdown`, and `junit`. All paths must be distinct, new, and outside the evidence
directory. Upfront collision refusal writes nothing; a later write failure
returns `ok: false`, lists completed files in `written_outputs`, and identifies
the failed format. Rendering does not discover or authenticate adjacent receipts.
Unsigned captured packs remain explicitly unsigned local reports. JUnit records
regression as failure and insufficient evidence as error.

### Read the result and its requirements

The report leads with **Policy satisfied** or **Policy not met**, the recorded
verdict and the checks responsible for it. HTML places the assurance checks
beside the verdict, with signature validation and independent recipient
acceptance kept separate. Each metric shows baseline and
candidate values, change, observed pair count, an interval and the configured
requirements. The interval diagram labels its scale, the policy boundary and the no-change
reference: zero for a difference and one for a ratio. Shading marks only the
region meeting the change requirement. Labels identify the interval endpoints,
estimate, policy limit and neutral reference; axis ticks use rounded steps in
the displayed units. Separate count, precision and absolute
score requirements still apply. The table shows which numerical checks passed
or were not met. Exact recorded values remain
available in the evidence and technical details.

For captured comparisons, the technical details identify both recorded runs,
their complete-run digests, attributed artifact digests and evaluator
source name and version. These fields make the report traceable to the exact
supplied runs. Optional model and prompt context describes the evaluator-recorded
change; it does not establish checkpoint identity. Only `verify` can add
independent authentication and replay assurance.

Hosted captured evidence attributes a service observation, with an explicit
configuration, harness and observation window, instead of asserting an immutable
weight artifact. An observed model name or exposed revision remains a source
assertion. Replaying or rendering the evidence does not make new service calls
or establish current service behavior.

Keep policy satisfaction, regression or other policy failure, insufficient
evidence, and integrity or verification failure separate. An authentic policy
rejection differs from a submission that cannot be verified. Neither report
rendering nor technical policy satisfaction makes the organizational decision;
record that decision separately with its remaining conditions and scope.

For exact match, means appear as percentages and changes as percentage points.
The decision tests the paired interval's lower bound, not just the observed
change. An improving candidate can still fail an absolute accuracy floor. When
configured, the report shows observed and required record count, interval width,
and the separate baseline and candidate accuracy checks. An absent requirement
is identified as absent; rendering does not supply a new threshold.

For normalized negative log-likelihood, the change is a candidate-to-baseline
ratio and the upper ratio bound is compared with the policy maximum. Its
finite-schedule resampling interval describes stability on the authenticated
schedule. It does not establish population uncertainty. Available derived
perplexity values appear separately and do not affect acceptance; an unavailable
interpretation includes its recorded reason.

Complete binary-mean results can also show exact match counts when the retained
mean and record count determine an integer count consistently. Missing results
and fractional scores do not acquire match counts. HTML places match counts
beneath the corresponding score, and missing, included and required pair counts
beneath the usable-pair count. Required counts appear only when policy is bound.

HTML includes expandable identities, exact comparison data, paired outcome
analysis where available, and supplementary authenticated observations.
`--explain` adds technical details to the text view. Paired outcome analysis can
include improvement and regression counts, discordant pairs and the exact
McNemar probability; those diagnostics do not replace the configured decision.

### Understand the comparison

HTML, Markdown and terminal reports share the same comparison grouping and
put **What was compared** before the results. HTML aligns sides in a table;
Markdown and terminal output use paired fields so long identifiers remain
readable at narrow widths. The
section aligns baseline and subject fields side by side, highlights differences,
and groups matching displayed fields with the recorded workflow, task, dataset
and coverage information under **Additional recorded context**. Matching
previews do not establish equality of the complete retained fields. They appear
under **Matching displayed fields**, separately from additional context and
ambiguous entries; acronyms and original ambiguous labels are preserved. On small
screens, each comparison field stacks its explicitly labelled baseline and
subject values while retaining the table headers for assistive technology. The recorded changes explain
what the evidence establishes about the difference between the two sides.

Native run and import reports use the authenticated request and input identities.
They show recorded model IDs and provider names, dataset preparation and schedule
identity when available. Different artifact digests establish different artifacts;
they do not establish which model setting, prompt or weight changed.

Captured reports inspect context across every record. Consistent recorded model
keys, revisions, capture roles, dataset labels and message structure appear as
comparison context. Missing or mixed values are explicit. A matching model key is
a recorded label, not proof that model weights or runtime settings were identical.
The renderer does not infer a model revision from filenames or external catalogs.

For paired prompt changes, the report compares retained message sequences by
case ID. Effective messages and captured HTTP request messages are distinct
projections; the latter do not reveal hidden service-side instructions.
When every pair differs only by one uniform added system instruction, the report
identifies that change and includes a bounded instruction preview in the details.
It does not display full case prompts by default. Missing or inconsistent context
cannot establish a uniform prompt change. Comparison-only SDK reports cannot
recover record context without the associated evidence.

### Compare multiple metrics and scopes

Reports with multiple results include an expandable overview of every metric
and scope. It shows the baseline, subject, change, observed pair count, decision
and any failed or unavailable checks. The overall verdict and result counts
remain visible when choosing a metric; selecting one result does not change
the overall decision. The overview opens automatically if any result fails or
needs more evidence, so adverse checks are visible before choosing a tab.

Metric tabs group related scopes together. For example, quality and latency
measured overall and on an exceptions slice produce four results under two
metrics. Select a tab or an overview result to inspect its details. Use the arrow
keys, Home and End to move between tabs, or **Show all** to read every metric in
one page. Reports with one metric do not need tabs. Overlapping scope counts
must not be added together as though they were separate cases.

The report remains one offline HTML file. A fixed, hash-authorized script handles
navigation only; no evidence labels or values are inserted into executable code.
Without JavaScript, all metric details remain visible and section links still
work. Printing includes every metric, even when only one tab is selected.
Navigation does not score results, change policy decisions or verify evidence.

### Keep rendering separate from recipient acceptance

The assurance panel distinguishes embedded bundle authentication from
independent recipient acceptance. A bundle's embedded signer is not a
recipient-owned trust anchor. Use `invarlock verify` with independent policy,
identity and signer inputs to produce the signed acceptance or rejection
receipt. A report can show an authenticated policy failure.

A successful `report` invocation indicates that rendering completed, including
when the recorded policy verdict is `fail`. Automation should parse verified
JSON or validate a receipt, never scrape the console or HTML. Renderer failures
use the selected text or JSON mode; argument-parser errors remain usage errors.

### Reports from captured evaluation exports

```bash
invarlock evaluate request.yaml --signing-key signing-key.pem
invarlock verify evidence/ --trust-profile recipient/trust-inputs.json \
  --receipt verification.receipt.json
invarlock report evidence/
```

The trust profile must independently supply the captured policy, complete-run
and request pins, evidence signer, and verifier identity and key. The core
commands retain JSON output for automation. `report` regenerates presentations
from bounded evidence after checking the closed inventory, payload digests,
canonical JSON, cross-file bindings and embedded signature for signed packs. It
does not recompute scores or replay comparison arithmetic, and the embedded
signer does not authorize itself as a recipient trust anchor.

Captured reports distinguish unsigned local evidence, an embedded signature
validated without recipient authorization, and unavailable signing information
in a comparison-only view. They preserve `pass`, `regression` and
`insufficient_evidence` as the recorded decisions. A `regression` decision means
a policy bound failed, which can be an absolute floor even when the observed
candidate improved. Missing paired results remain missing; overlapping scope
counts are not independent samples. Advanced configuration and missing-ID lists
use labeled previews while the original values remain in bound evidence.
Configuration previews use indented text with explicit limits and truncation
markers; the renderer does not reinterpret them as executable content or
complete configuration. Small screens stack each requirement with labelled Observed, Required and
Result values. All checks remain visible without horizontal scrolling. HTML
follows the system light or dark appearance; printed reports use the light
color scheme.

Use `invarlock verify` with recipient-owned policy and complete-run digests for
independent authentication and replay. Successful captured report
regeneration exits `0` regardless of the stored policy decision. Evaluation exits
`0` on publication unless `--fail-on-policy` requests the local gate; independent
verification rejects adverse decisions. See the
[captured-results guide](../user-guide/captured-results.md) for the full
workflow and its assurance limits.

## Failure-state interpretation

| State | Integrity | Policy | Portable receipt | Interpretation |
| --- | --- | --- | --- | --- |
| Accepted | True | `pass` | Signed success receipt | Conservative interval bound and any configured sample/precision requirements met the policy under all supplied anchors |
| Policy rejection | True | `fail` | Signed rejection receipt when completion reached | Authentic evidence did not meet the policy |
| Integrity rejection | False | Unavailable or untrusted | Signed rejection receipt when safe completion reached | Pack must not be used |
| Precondition failure | Not completed | Unavailable | May be absent | Caller input or structure prevented completed verification |
| Render success only | Embedded signature checked for signed packs; unsigned evidence remains local | Recorded verdict only | None | Inspection of evidence, not independent acceptance |

## Related documentation

- [Decision semantics](../assurance/decision-semantics.md) defines the exact
  score, interval, and threshold arithmetic.
- [Evidence artifacts](artifacts.md) maps reports to fixed bundle paths.
- [Public contracts](contracts.md) specifies canonical bytes and signature
  dependency order.
- [Command-line interface](cli.md) documents JSON modes and exit status.
- [Python API](api-guide.md) defines result and signed-receipt types.

For bounded frozen-answer ratings, see [judge measurements](judge-measurements.md).
Judge reports keep offline replay, signer authentication and recipient acceptance
distinct and use additive judge-specific JSON formats. Rendering replays the
retained measurements; `verify` performs envelope signature authentication
against the recipient policy. They show the first 50
case IDs by default; repeat `report --case-id ID` to inspect any retained cases
without loading every answer and judge response into one report. Case selection
does not change the complete replay or the recorded policy decision.
