# Guard Overhead Method & Budget

> **Plain language:** We measure how much the GuardChain adds to the primary metric
> using the exact same windows and seeds (paired schedule), then gate against a small budget
> (≤ 1%). If overhead exceeds the budget or provenance is missing, the
> certificate fails.

## Claim

- overhead_ratio = PM(guarded) / PM(bare)  (for ppl‑like kinds this is a ratio; for accuracy use Δ pp)
- overhead_percent = (overhead_ratio − 1) × 100

PM(bare) is computed with guards disabled; PM(guarded) with the full GuardChain enabled.

## Protocol (single toggle, paired schedule)

- Same window plan: identical `seq_len`, `stride`, counts, and window IDs.
- Same seeds: reuse the seed bundle (`python`, `numpy`, `torch`) and bootstrap seed (when applicable).
- Single toggle: run a bare control (guards disabled) and a guarded run on the same model snapshot.
- Deterministic snapshot: prefer snapshot/restore between bare and guarded; otherwise reload deterministically.

## Thresholds

- Release (default): ≤ +1.0% overhead (fraction `0.01`).
- CI: same default unless overridden per profile.

Rationale: the budget must be small relative to sampling noise and locked to a policy digest so it cannot silently drift.

## Runtime Contract (certificate)

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
  soft-passes (reported in `guard_overhead.errors`) to avoid spurious failures
  in tiny runs.

## Observability & Provenance

- Seeds and device: `meta.seeds.*`, `meta.device` recorded for both bare and guarded arms.
- Policy snapshot & digest: `/resolved_policy`, `/policy_provenance.policy_digest`, `/auto.policy_digest`, and `/policy_digest` (thresholds digest) pin the evaluated policy and floors.

## Remediation (if the gate fails)

- Increase window counts to tighten CI and reduce noise; keep pairing identical.
- Inspect hotspots in guard compute; review guard settings (e.g., spectral caps, epsilon map) relative to tier.
- Confirm the bare run is truly guard‑free and comes from the same snapshot; avoid extra logging/export overhead.
- Consider a local budget override only with documented justification and pilot evidence.

## References

- Certificate Schema → Guard Overhead section (field list and example JSON)
- Guard Contracts → Overview of guards and expected budgets
