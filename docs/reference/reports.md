# Reports and receipts

InvarLock separates provider facts, a canonical comparison report, independent
verification output, a signed verification receipt, and a human rendering.
They have different trust meanings.

!!! info "Reference"

    - **Surface:** Runtime-side reports, canonical comparison reports,
      verification results, signed receipts, and human renderings
    - **Stability:** Versioned machine contracts are stable by format; console
      and HTML output are presentation surfaces
    - **Use this page when:** Parsing a decision, validating a receipt, or
      distinguishing signed evidence from independently accepted evidence

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

New evaluations write `reports/evaluation.report.json` as an
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
present, the final report passes only when its count and width checks also pass.

For this metric, baseline and subject `mean_score` are literal exact-match
accuracies between `0` and `1`. Comparison and interval bounds are percentage
points between `-100` and `100`.

`paired_binary` reports baseline-pass to subject-fail regressions,
baseline-fail to subject-pass improvements, both-pass and both-fail counts, the
number of discordant pairs, and the exact two-sided McNemar probability. Its
effect size is the same subject-minus-baseline percentage-point delta. The
continuity-corrected paired Newcombe hybrid-score 95% interval is repeated as
the canonical `uncertainty` object in a v2 report. The policy uses its lower
bound; the McNemar probability does not control the verdict. A v1 report keeps
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

The scorer-extension v1 contract is suitable for separately supplied
deterministic text scorers such as token F1, structured-field extraction, or VQA
answer normalization. Those scorers are separately installed and require
explicit authorization. SQL or code execution, model-based semantic similarity,
network or human services, and LLM judges are outside acceptance replay. Judge
results may be authenticated as observations, where they have no verdict
authority.

## Verification result

`invarlock verify --json` emits an `invarlock/evidence-pack-verify-v1` result. Important
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

Do not infer acceptance from `integrity_ok` alone. Require status `0`,
`ok: true`, the expected `assurance_status`, exact anchors, and a valid receipt
when the result crosses a process boundary.

## Signed verification receipt

The receipt contains a `statement` and `signature`. The statement format is
`invarlock/evidence-verification-receipt-v1`:

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

## Human report

`invarlock report` authenticates the closed inventory, checksums, canonical JSON,
and embedded evidence signature before rendering the comparison as console text
or standalone HTML. Rendering never changes bundle bytes.

A shortened text view is:

```text
# InvarLock comparison report

- **Comparison:** `cmp-example`
- **Metric:** `exact_match`
- **Records:** 20
- **Verdict:** **PASS**
- **Bundle integrity:** embedded evidence signature verified
- **Acceptance path:** `invarlock verify` records the expected signer and independent anchors in a signed receipt
- **Evidence signer:** `sha256:...`

| Measure | Value |
| --- | ---: |
| Baseline mean | 0.75 |
| Subject mean | 0.8 |
| Exact-match delta (pp) | 5 |
| Paired Newcombe 95% interval | [-14.975805, 24.866849] |
| Minimum allowed (pp) | -15 |
| Paired record count | 20 (minimum 20) |
| Confidence-interval width (pp) | 39.842654 (maximum 50) |
```

The exact-match rendering also lists regression and improvement counts, the
discordant-pair count, and the exact two-sided McNemar probability. When the
policy includes sample qualification, console and HTML render the observed and
required record count, interval width, and combined qualification result.

The human view states what it is: a rendering of signature-authenticated
evidence. Independent acceptance comes from `invarlock verify` and its signed
receipt. Automation should parse verified JSON or validate a receipt, never
scrape console or HTML.

## Failure-state interpretation

| State | Integrity | Policy | Portable receipt | Interpretation |
| --- | --- | --- | --- | --- |
| Accepted | True | `pass` | Signed success receipt | Conservative interval bound and any configured sample/precision requirements met the policy under all supplied anchors |
| Policy rejection | True | `fail` | Signed rejection receipt when completion reached | Authentic evidence did not meet the policy |
| Integrity rejection | False | Unavailable or untrusted | Signed rejection receipt when safe completion reached | Pack must not be used |
| Precondition failure | Not completed | Unavailable | May be absent | Caller input or structure prevented completed verification |
| Render success only | Embedded signature authenticated | Stored verdict only | None | Human inspection of evidence, not independent acceptance |

## Related documentation

- [Decision semantics](../assurance/decision-semantics.md) defines the exact
  score, interval, and threshold arithmetic.
- [Evidence artifacts](artifacts.md) maps reports to fixed bundle paths.
- [Public contracts](contracts.md) specifies canonical bytes and signature
  dependency order.
- [Command-line interface](cli.md) documents JSON modes and exit status.
- [Python API](api-guide.md) defines result and signed-receipt types.
