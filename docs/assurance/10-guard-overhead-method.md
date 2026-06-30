# Guard Overhead Method & Budget

> **Plain language:** We measure how much the GuardChain adds to the primary metric
> using the exact same windows and seeds (paired schedule), then gate against a small budget
> (≤ 1%) when the overhead ratio is evaluated. Report generation soft-passes
> unavailable ratios as `evaluated=false` for tiny/noisy runs; release
> verification requires evaluated overhead evidence unless the run explicitly
> records an overhead skip.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the paired bare-vs-guarded overhead measurement and the release-evidence budget. |
| **Audience** | Release approvers, runtime maintainers, and operators producing overhead evidence. |
| **Contract scope** | Guard overhead ratio/percent reporting, soft-pass diagnostics, and release verifier requirements. |
| **Source of truth** | `src/invarlock/reporting/report_overhead.py`, `src/invarlock/reporting/verify_check_helpers_consistency.py`, and overhead tests. |

## Claim

- overhead_ratio = PM(guarded) / PM(bare) for ppl‑like/lower-is-better primary
  metrics.
- overhead_percent = (overhead_ratio − 1) × 100

PM(bare) is computed with guards disabled; PM(guarded) with the full GuardChain enabled.
Accuracy-style quality deltas are handled outside this ratio gate; do not read
`guard_overhead.overhead_ratio` as an accuracy percentage-point delta.

## Protocol (single toggle, paired schedule)

- Same window plan: identical `seq_len`, `stride`, counts, and window IDs.
- Same seeds: reuse the seed bundle (`python`, `numpy`, `torch`) and bootstrap seed (when applicable).
- Single toggle: run a bare control (guards disabled) and a guarded run on the same model snapshot.
- Deterministic snapshot: prefer snapshot/restore between bare and guarded; otherwise reload deterministically.

## Thresholds

- Release (default): ≤ +1.0% overhead (fraction `0.01`).
- CI: same default unless overridden per profile.

Rationale: the budget must be small relative to sampling noise and locked to a policy digest so it cannot silently drift.

## Runtime Contract (report)

Fields under `/guard_overhead` and `/validation`:

- `guard_overhead.bare_ppl`
- `guard_overhead.guarded_ppl`
- `guard_overhead.overhead_ratio`
- `guard_overhead.overhead_percent`
- `guard_overhead.overhead_threshold` (fraction)
- `validation.guard_overhead_acceptable` (boolean)

The Markdown summary repeats the verdict (PASS/FAIL) and measured values.

Fail conditions (gate evaluated):

- `guard_overhead.overhead_ratio > 1 + guard_overhead.overhead_threshold`.
- If the ratio cannot be computed, the check is marked `evaluated=false` and
  soft-passes (reported in `guard_overhead.diagnostics`) to avoid spurious failures
  in tiny runs.

Release verifier behavior:

- `--profile release` requires a `guard_overhead` section unless the run records
  an explicit skip (`guard_overhead.skipped=true` or `mode: skipped`).
- If release overhead is not skipped, `guard_overhead.evaluated` must be `true`
  and `guard_overhead.overhead_ratio` must be present. Missing or unevaluated
  overhead is a release evidence failure even though report generation can
  soft-pass the unavailable ratio.

## Observability & Provenance

- Seeds and device: `meta.seeds.*`, `meta.device`, paired overhead metrics, and
  the schedule/policy digest are recorded in the final report. Embedded
  bare/guarded arm metadata is sanitized before report publication.
- Policy snapshot & digest: `/resolved_policy`, `/policy_provenance.policy_digest`, `/auto.policy_digest`, and `/policy_digest` (thresholds digest) pin the evaluated policy and floors.

## Remediation (if the gate fails)

- Increase window counts to tighten CI and reduce noise; keep pairing identical.
- Inspect hotspots in guard compute; review guard settings (e.g., spectral caps, epsilon map) relative to tier.
- Confirm the bare run is truly guard‑free and comes from the same snapshot; avoid extra logging/export overhead.
- Consider a local budget override only with documented justification and pilot evidence.

## References

- [Reports Reference](../reference/reports.md) — Guard overhead field list and example JSON
- [Guard Contracts](04-guard-contracts.md) — Overview of guards and expected budgets
