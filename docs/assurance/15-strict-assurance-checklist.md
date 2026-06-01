# Strict Assurance Checklist

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Reviewer checklist for accepting strict assurance evidence. |
| **Audience** | Maintainers, release reviewers, CI gate owners. |
| **Contract scope** | Current strict assurance behavior, claim set `invarlock-weight-edit-regression-v1`, report v1. |
| **Source of truth** | `src/invarlock/core/assurance_contract.py`, `src/invarlock/reporting/verify_contract.py`, `docs/assurance/14-trust-model.md`. |

Use this checklist before accepting a strict report as assurance evidence.
When a checkbox cannot be ticked, see [Failure Examples](../user-guide/failure-examples.md)
for the matching non-pass shape and [Troubleshooting](../user-guide/troubleshooting.md)
for numbered error codes.

## Quick Start

```bash
invarlock verify --assurance strict reports/eval/evaluation.report.json
```

A green exit from this command satisfies the report/manifest checks that are
machine-checkable from the submitted evidence. The remaining items are reviewer
judgment about policy allowances and bundle contents.

## Machine-Checked Command Surface

- [ ] `invarlock evaluate` ran with `--assurance strict` or the default strict mode.
- [ ] `--profile` was `ci` or `release`.
- [ ] `--tier` was `balanced` or `conservative`.
- [ ] Runtime execution was container-backed.
- [ ] Unverified provenance was not allowed.

## Reviewer-Confirmed Policy Context

- [ ] Network and remote-code allowances were reviewed and recorded.
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

## Guard Fallback Policy

- [ ] Numeric measurement fallbacks are recorded as diagnostics or events; a
  neutral fallback value alone is not acceptable evidence.
- [ ] Spectral estimator failures, non-tensor weights, non-finite weights, and
  quantized-weight skips include structured `spectral_sigma_fallback_*`
  diagnostics.
- [ ] RMT correction failures are emitted as `rmt_correct_failed` error events
  and do not silently erase the original outlier.
- [ ] Variance guard preparation/finalization failures fail closed unless an
  explicit monitor-only policy is recorded in the report.
- [ ] Reviewer-facing reports expose fallback diagnostics under the relevant
  guard result, and strict assurance blocks unsupported or degraded guard states.

## Metrics And Windows

- [ ] Final and baseline paired arrays have equal lengths.
- [ ] Window match fraction is `1.0`.
- [ ] Window overlap fraction is `0.0`.
- [ ] `ratio_vs_baseline` equals the exponentiated paired delta log-loss.
- [ ] `display_ci` equals `exp(ci)` for paired ppl-like metrics.
- [ ] Bootstrap coverage satisfies the selected tier floor.

## Provenance

- [ ] `runtime.manifest.json` is present and verified.
- [ ] Runtime image provenance is digest-pinned or explicitly non-assurance.
- [ ] Tokenizer hash and provider digest match the baseline/subject contract.
- [ ] Policy digest and resolved policy are present in the report.

## Report Verdict

- [ ] Top-level `assurance.mode` is `strict`.
- [ ] Generated report has `assurance.verdict` set to `pending_verifier`.
- [ ] Generated report has `assurance.report_local_verdict` set to `pass`.
- [ ] Generated report has `assurance.verified_assurance_verdict` set to `pending`.
- [ ] `assurance.fallback_fields_used` is `false`.
- [ ] `assurance.runtime_provenance_verified` is `false` before verifier confirmation.
- [ ] `assurance.blocking_reasons` is empty.
- [ ] `invarlock verify --assurance strict` exits successfully and reports
  `verification.runtime_provenance.status = verified`.

## Related Documentation

- [Trust Model](14-trust-model.md) — What a strict pass does and does not mean
- [Assurance Case Overview](00-assurance-case.md) — Claims, evidence, and tests
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md) — Manifest requirements
- [Failure Examples](../user-guide/failure-examples.md) — Common non-pass shapes
- [Troubleshooting](../user-guide/troubleshooting.md) — Numbered error codes
- [Reports Reference](../reference/reports.md) — Full v1 schema
- [One Run Lifecycle](../reference/one-run-lifecycle.md) — Where each gate runs
