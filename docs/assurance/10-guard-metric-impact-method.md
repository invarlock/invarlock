# Guard Primary-Metric Impact Method and Budget

> **Plain language:** We compare the primary metric from guarded and bare arms
> using the exact same examples and seeds (paired schedule), then gate the
> metric-specific degradation against a small budget (default `0.01`). For
> perplexity this means at most a 1% relative increase; for accuracy it means at
> most a 1 percentage-point absolute drop. `guard_metric_impact` is a
> model-quality comparison, not an elapsed-time, memory, energy, or compute
> measurement. An unavailable or skipped
> comparison is recorded as `evaluated=false`, `passed=false`; it cannot satisfy
> Release or strict assurance.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the paired bare-vs-guarded primary-metric comparison and release-evidence budget. |
| **Audience** | Release approvers, runtime maintainers, and operators producing guard-impact evidence. |
| **Contract scope** | Direction-aware degradation reporting, fail-closed diagnostics, and release verifier requirements. |
| **Source of truth** | `src/invarlock/reporting/report_metric_impact.py`, `src/invarlock/reporting/verify_check_helpers_consistency.py`, and metric-impact tests. |

## Claim

Let `bare_value` be the primary metric with guards disabled and `guarded_value`
the same metric with the full GuardChain enabled. The metric registry determines
the allowed direction and degradation basis:

- PPL-like, lower-is-better metrics use `degradation_basis: relative_increase`:
  `degradation = guarded_value / bare_value - 1`.
- Accuracy, a higher-is-better metric, uses
  `degradation_basis: absolute_drop`:
  `degradation = bare_value - guarded_value`.

For both bases, positive degradation is worse and negative degradation is an
improvement. The gate passes when the values are valid and finite and
`degradation <= degradation_limit`. PPL values must also be positive; accuracy
values must be in `[0, 1]`. The configured default `degradation_limit: 0.01`
therefore has different, explicit units: 1% relative PPL increase or one
accuracy percentage point.

This comparison does not measure runtime cost. A latency or resource-cost
claim requires separate timing, synchronization, warm-up, memory, and hardware
measurement protocols.

## Protocol (single toggle, paired schedule)

- Same evaluation schedule: identical example/window IDs, counts, and ordering;
  windowed metrics also use identical `seq_len` and `stride`.
- Same seeds: reuse the seed bundle (`python`, `numpy`, `torch`) and bootstrap seed (when applicable).
- Single toggle: run a bare control (guards disabled) and a guarded run on the same model snapshot.
- Deterministic snapshot: prefer snapshot/restore between bare and guarded; otherwise reload deterministically.

## Degradation limits

- Release (default): `degradation <= 0.01` using the metric's registered basis.
- CI: same default unless overridden per profile.

The `0.01` value is a shipped policy choice. The repository does not contain a
representative empirical study proving it is below sampling noise or appropriate
for every model/task. Its digest makes policy drift visible, not scientifically
justified.

## Runtime Contract (report)

Fields under `/guard_metric_impact` and `/validation`:

- `guard_metric_impact.metric_kind`: registered primary-metric kind.
- `guard_metric_impact.direction`: `lower` or `higher`, as registered for that
  metric kind.
- `guard_metric_impact.degradation_basis`: `relative_increase` or
  `absolute_drop`, as registered for that metric kind.
- `guard_metric_impact.bare_value` and `guard_metric_impact.guarded_value`:
  measured metric values from the paired arms.
- `guard_metric_impact.bare_facts` and `guard_metric_impact.guarded_facts`:
  arm measurements that verification can recompute. Accuracy retains `correct`,
  `total`, and the
  ordered-example digest; PPL retains `weighted_logloss_sum`, `token_count`,
  and the ordered-window digest.
- `guard_metric_impact.bare_report`: a closed, minimal bare-control envelope
  containing the raw final counts or log losses, token counts, ordered IDs,
  and primary-metric identity needed to reproduce `bare_facts` and
  `bare_value`. The canonical report and evidence-pack signature bind this
  envelope; it is not an unverified self-hash.
- `guard_metric_impact.degradation`: canonical raw degradation in the basis
  above; negative values record improvements.
- `guard_metric_impact.degradation_limit`: maximum permitted degradation in the
  same basis.
- `guard_metric_impact.display_value` and `guard_metric_impact.display_unit`:
  presentation form (`percent` for relative PPL change or `percentage_points`
  for accuracy change).
- `guard_metric_impact.evaluated`, `guard_metric_impact.passed`,
  `guard_metric_impact.checks`, and `guard_metric_impact.diagnostics` record the
  outcome and any failure reason.
- `guard_metric_impact.source` and `guard_metric_impact.schedule_digest` identify
  the comparison source and bind the paired evaluation schedule.
- `validation.guard_metric_impact_acceptable` (boolean)

The Markdown summary repeats the verdict (PASS/FAIL) and measured values.

Fail conditions include:

- `guard_metric_impact.degradation > guard_metric_impact.degradation_limit`.
- A metric kind, direction, or degradation basis that disagrees with the metric
  registry.
- Reported degradation or display values that do not match recomputation from
  the two arm values.
- Arm facts that do not replay from the retained bare envelope and the
  guarded report's actual primary metric and final evaluation windows.
- Missing or differently ordered paired IDs, or a schedule digest that does
  not match the guarded report's final schedule.
- A missing, skipped, or non-finite comparison is marked `evaluated=false`,
  `passed=false` and explained in `guard_metric_impact.diagnostics`.

Verifier behavior:

- Release and strict assurance require a nonempty `guard_metric_impact` block with
  `evaluated=true`, `passed=true`, valid finite arm values and degradation, and
  a finite non-negative degradation limit. Both explicit skips and unavailable
  comparisons fail. Strict verification also recomputes the metric-kind-specific
  values from the bound bare and guarded evidence rather than trusting the
  summary fields alone.

## Observability & Provenance

- Seeds and device: `meta.seeds.*`, `meta.device`, paired impact metrics, and
  the schedule/policy digest are recorded in the final report. The report
  retains only the closed bare-control envelope; guarded evidence is replayed
  from the report's canonical primary metric and final evaluation windows.
- Policy snapshot & digest: `/resolved_policy`, `/policy_provenance.policy_digest`, `/auto.policy_digest`, and `/policy_digest` (thresholds digest) pin the evaluated policy and floors.

## Investigation if the gate fails

- Confirm the bare run is truly guard‑free, the guarded run uses the same
  snapshot, and arm order/state did not change model behavior.
- Check window IDs, token counts, seeds, device state, caching, hooks, and any
  mutation introduced by guard preparation.
- Repeat a paired protocol fixed before observing results if sampling uncertainty is material;
  increasing counts after observing a failure must not be used to cherry-pick a
  passing result.
- Change the policy budget only through a documented, independently reviewed
  decision with task-specific evidence.

## References

- [Reports Reference](../reference/reports.md) — Guard metric impact field list and example JSON
- [Guard Contracts](04-guard-contracts.md) — Overview of guards and expected budgets
