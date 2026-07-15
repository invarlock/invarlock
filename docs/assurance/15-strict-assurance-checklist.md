# Strict Assurance Checklist

> **Plain language:** This is the acceptance checklist for deciding whether a
> strict report and its sibling runtime manifest can be accepted as assurance
> evidence.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Acceptance checklist for strict assurance evidence. |
| **Audience** | Maintainers, release approvers, CI gate owners. |
| **Contract scope** | Current strict assurance behavior, claim set `invarlock-weight-edit-regression-v2`, report v1. |
| **Source of truth** | `src/invarlock/core/assurance_contract.py`, `src/invarlock/reporting/verify_contract.py`, `docs/assurance/14-trust-model.md`. |

Use this checklist before accepting a strict report as assurance evidence.
When a checkbox cannot be ticked, see [Failure Examples](../user-guide/failure-examples.md)
for the matching non-pass shape and [Troubleshooting](../user-guide/troubleshooting.md)
for numbered error codes.

## Quick Start

```bash
TRUSTED_RUNTIME_IMAGE_DIGEST='sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST'
BASELINE_RUN_REPORT='/path/to/retained/baseline/report.json'
ACCEPTANCE_POLICY_PACK='/path/to/acceptance/policy-pack.json'
invarlock verify --profile release --assurance strict \
  --baseline "$BASELINE_RUN_REPORT" \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  reports/eval/evaluation.report.json
```

A green exit from this command satisfies the report/manifest checks that are
machine-checkable from the submitted evidence. The remaining items are human
judgment about policy allowances and bundle contents.

`invarlock report export --format release-review-md` can generate a release
packet from the same report and an optional `invarlock verify --json` result.
Use it as a checklist wrapper; it does not replace this checklist or change the
strict assurance contract.

## Machine-Checked Command Surface

- [ ] `invarlock evaluate` ran with `--assurance strict` or the default strict mode.
- [ ] `--profile` was `ci` or `release`.
- [ ] `--tier` was `balanced` or `conservative`.
- [ ] Runtime execution was declared container-backed and the report is bound
  to its sibling manifest.
- [ ] `--expected-runtime-image-digest` came from reviewed policy or another
  source independent of the submitted manifest.
- [ ] `--baseline` names the complete baseline `report.json` retained from the
  noop baseline run, not a hand-written metric/evaluation fragment or a copy
  reconstructed from the subject.
- [ ] `--policy-pack` names an independently supplied policy pack created outside
  the submitted report bundle, exactly matching the intended resolved policy,
  and binding the expected dataset identity under
  `compatibility.dataset_identity`.
- [ ] `resolved_policy.guard_authority` exactly matches the v2 policy pack and
  is mirrored by `assurance.guard_authority`.
- [ ] Unverified provenance was not allowed.

## Confirmed Policy Context

- [ ] Network and external-code allowances were reviewed and recorded.
- [ ] The original evaluate command and staged bundle contents match the
  release/review intent.

## Guard Chain

- [ ] The observed guard chain is exactly:
  `invariants -> spectral -> rmt -> variance -> invariants`.
- [ ] No guard evidence is missing; the single `invariants` evidence block
  covers both pre/post invariant stages in the current report contract.
- [ ] No guard was skipped, duplicated outside the canonical chain, or marked
  monitor-only for a pass.
- [ ] Unsupported guard/model statuses are explicit and block assurance.
- [ ] Spectral, RMT, and variance evidence is complete and can be independently
  replayed. Findings block when the corresponding authority is `enforce` and
  remain visible when it is `observe`.
- [ ] Primary metrics, drift, invariants, and guard-metric impact pass; these
  mandatory blockers have no observation mode.
- [ ] `guard_metric_impact` is measured, evaluated, and passing; it is not skipped.
- [ ] Its `metric_kind`, `direction`, and `degradation_basis` agree with the
  metric registry. PPL uses `relative_increase`; accuracy uses `absolute_drop`.
- [ ] `degradation` and the display value match recomputation from `bare_value`
  and `guarded_value`; negative values are accepted as improvements.
- [ ] The paired arm evidence and schedule binding are present and consistent,
  and the configured `degradation_limit` is finite and non-negative.

## Guard Fallback Policy

- [ ] Numeric measurement fallbacks are recorded as diagnostics or events; a
  neutral fallback value alone is not acceptable evidence.
- [ ] Spectral estimator failures, non-tensor weights, non-finite weights, and
  quantized-weight skips include structured `spectral_sigma_fallback_*`
  diagnostics.
- [ ] RMT correction failures are emitted as `rmt_correct_failed` error events
  and do not silently erase the original outlier.
- [ ] Variance guard preparation/finalization failures and monitor-only results
  fail closed.
- [ ] Evidence reports expose fallback diagnostics under the relevant
  guard result, and strict assurance blocks unsupported or degraded guard states.

## Metrics And Windows

- [ ] Every strict metric, including accuracy, has an independently supplied
  `--baseline`; omission is a release-blocking verification failure.
- [ ] Final and baseline paired arrays have equal lengths.
- [ ] Window match fraction is `1.0`.
- [ ] Window overlap fraction is `0.0`.
- [ ] `ratio_vs_baseline` equals the exponentiated paired delta log-loss.
- [ ] PPL preview and final raw log-loss/token-count arrays are non-empty,
  equal-length, finite, and positively weighted; both analysis points and both
  displayed perplexities match recomputation.
- [ ] `display_ci` equals `exp(ci)`, bounds are ordered, and the configured
  strict interval contract contains the recomputed baseline log-ratio point.
- [ ] Bootstrap coverage satisfies the selected tier floor.
- [ ] Accuracy reports include positive integer
  `primary_metric.{n_preview,n_final}`, measured preview/final classification
  counts, and matching records/window/coverage counts wherever those surfaces
  are present.
- [ ] Accuracy `delta_vs_baseline_pp` equals
  `100 × (final_accuracy - baseline_accuracy)`; accuracy reports do not
  contain the PPL-only `ratio_vs_baseline` field.
- [ ] Baseline accuracy is recomputed from measured aggregate counts and raw
  per-example correctness, and its final example IDs exactly match the subject
  schedule in order. Raw accuracy surfaces are required in strict baselines.

## Provenance

- [ ] `runtime.manifest.json` is present and report-hash bound.
- [ ] The manifest's image digest matches the independently supplied expected
  runtime image digest. This validates the manifest's claimed image identity and
  report binding; it does not attest that the claimed container executed.
- [ ] Baseline and subject have non-empty, matching provider IDs digests,
  tokenizer digests/hashes, model/adapter identity, dataset provider/split,
  sequence length, and applicable dataset/revision hashes.
- [ ] Policy digest and resolved policy are present in the report, and the
  resolved policy exactly matches the independently supplied policy pack.
- [ ] The policy pack's dataset provider, name, configuration, immutable
  revision (for hosted data), and split match both report arms and the prepared
  evaluation inputs.

## Report Verdict

- [ ] Top-level `assurance.mode` is `strict`.
- [ ] Generated report has `assurance.verdict` set to `pending_verifier`.
- [ ] Generated report has `assurance.report_local_verdict` set to `pass`.
- [ ] Generated report has `assurance.verified_assurance_verdict` set to `pending`.
- [ ] `assurance.fallback_fields_used` is `false`.
- [ ] `assurance.runtime_provenance_verified` is `false` before verifier confirmation.
- [ ] `assurance.blocking_reasons` is empty.
- [ ] `invarlock verify --assurance strict --baseline ... --policy-pack ... --expected-runtime-image-digest ...`
  exits successfully and reports runtime status `expected_image_digest_matched`
  with `binding_verified=true` and `expected_digest_matched=true`.
- [ ] `verify --json` includes the unsigned `verification.receipt` with the exact
  subject/baseline SHA-256 values and the verifier inputs used for review.

## Human/External Trust Checks

- [ ] The expected runtime-image digest was obtained from separately controlled
  policy, not copied from the submitted manifest.
- [ ] If actual execution identity matters, independent rerun or execution
  attestation evidence is present. A compromised evaluation environment can fabricate both
  report and manifest while naming the expected digest.
- [ ] Dataset labels, checkpoint identity, window selection, and inclusion of
  failed runs were reviewed independently; internal count/hash consistency does
  not prove honest or representative sampling.
- [ ] Evidence-pack signer trust and runtime-image policy are evaluated as
  separate anchors. A trusted signer authenticates the signer, not the truth
  of every measurement.

## Related Documentation

- [Trust Model](14-trust-model.md) — Strict pass scope
- [Assurance Case Overview](00-assurance-case.md) — Claims, evidence, and tests
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md) — Manifest requirements
- [Reports Reference](../reference/reports.md) — Release-review export format
- [Failure Examples](../user-guide/failure-examples.md) — Common non-pass shapes
- [Troubleshooting](../user-guide/troubleshooting.md) — Numbered error codes
- [One Run Lifecycle](../reference/one-run-lifecycle.md) — Where each gate runs
