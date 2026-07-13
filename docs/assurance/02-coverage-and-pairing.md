# Coverage & Pairing Plan

> **Plain language:** Baseline and edited runs reuse identical IDs within each
> arm, while preview and final remain disjoint slices. Fixed seeds and
> `stride == seq_len` prevent schedule drift and sliding-window token overlap.
> Tier-based minima are validated at runtime and surfaced in the report.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the pairing, non-overlap, seed, and tier-floor requirements for evaluation windows. |
| **Audience** | Evaluation pipeline maintainers, release approvers, and operators preparing paired evidence. |
| **Contract scope** | Baseline/subject window reuse, pairing statistics, coverage floors, and report-verifier checks. |
| **Source of truth** | `src/invarlock/core/runner_runtime/pairing.py`, `src/invarlock/eval/window_planning.py`, and report pairing tests. |

## Claim

A valid evaluation schedule uses fixed seeds, reuses baseline IDs for the
corresponding edited-run arm, keeps preview/final ID sets disjoint, and avoids
sliding-window token overlap. The runner enforces tier-based minima. CI/Release
runs hard-fail baseline matching/count drift when a pairing context exists, and
report verification rejects invalid baseline pairing or intersecting slice IDs.

## Window Selection (assumptions)

- **Non‑overlap:** set `seq_len == stride` so windows do not overlap.
- **Deterministic:** record and reuse the seed bundle (`python`, `numpy`, `torch`) and bootstrap seed (when applicable).
- **Dedupe:** deduplication is allowed for pilots/probes; **release evidence uses strict non‑overlap on the full plan**.
- **Exact pairing:** preview/final counts must match; preview and final ID sets
  must be disjoint; and, within each arm, the edited run must reuse the
  baseline IDs. Mixing baseline/subject schedules invalidates the paired Δlog
  assumptions.

## Pairing Reuse (baseline → edited)

- The edited run pins windows via the baseline report.
- report lints baseline-context matching and configured token-window overlap:
  - `dataset.windows.stats.window_match_fraction == 1.0`
  - `dataset.windows.stats.window_overlap_fraction == 0.0`
- Strict verification separately checks that raw
  `evaluation_windows.preview.window_ids` and
  `evaluation_windows.final.window_ids` are disjoint.
- CI/Release abort if counts differ, pairing < 1.0, or overlap > 0.0 when a
  baseline pairing context exists.

## Tier Minima (runner defaults)

Sane defaults enforced by the runner per tier (guard-rail floors; profiles may
request higher counts):

| Tier         | Preview Windows | Final Windows | Bootstrap Replicates |
|--------------|------------------|---------------|----------------------|
| Conservative | 220              | 220           | 1,500                |
| Balanced     | 180              | 180           | 1,200                |
| Aggressive   | 140              | 140           |   800                |

These minima are shipped policy floors chosen to trade evidence volume against
evaluation cost. The half-width calculation in
[Tier Policy v1 Values](09-tier-v1-calibration.md) is a planning heuristic; the
repository does not establish nominal interval coverage or detection power for
these counts. CI/Release profiles treat
shortfalls as hard errors; dev flows surface warnings but also record coverage
in the generated report bundle.

## Runtime Contract (report)

- Window plan: `dataset.windows.stats.{requested_preview,requested_final,actual_preview,actual_final}`
- Pairing/overlap: `dataset.windows.stats.{window_match_fraction,window_overlap_fraction,paired_windows}`
- Coverage floors: `dataset.windows.stats.coverage.{preview,final}` meets/exceeds
  the window tier floor (profiles may request higher counts)
- Bootstrap metadata: `dataset.windows.stats.bootstrap.{method,alpha,replicates,seed}`
  records the baseline-pair method and base RNG configuration;
  `bootstrap.{preview_final_delta_basis,preview_final_delta_method,preview_final_delta_seed}`
  records the separate independent-slice replay contract.

## Observability

- Pairing and coverage appear in both the Markdown report and the JSON report.
  The JSON `evaluation_windows` arrays retain the IDs needed to verify exact
  baseline reuse and confirm that preview/final slices are disjoint.

## Assumptions & Scope

- Applies to **evaluation (inference) schedules**; training/edit algorithms may
  alter data flow and are out of scope here.
- Dataset or tokenizer changes that affect tokenization invalidate recorded
  pairing schedules.
- Baseline/subject ID reuse must be exact; preview/final IDs must be disjoint;
  and the configured stride must avoid token-window overlap. Mixing
  baseline/subject schedules invalidates paired Δlog assumptions.
- This plan is implemented and tested for the documented Linux/macOS profiles.
  Statistical representativeness still depends on the dataset/window sampling
  protocol and must be justified separately.
