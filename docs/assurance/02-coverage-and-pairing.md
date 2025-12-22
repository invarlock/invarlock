# Coverage & Pairing Plan

> **Plain language:** Use non‑overlapping, paired windows with fixed seeds.
> Baseline and edited runs reuse the exact same windows. Tier‑based minima are
> validated at runtime and surfaced in the certificate.

## Claim

A valid evaluation schedule uses non‑overlapping, paired windows with fixed
seeds and reuses the baseline window IDs for edited runs. The runner enforces
tier‑based minima and aborts in CI/Release when pairing or coverage is
insufficient.

## Window Selection (assumptions)

- **Non‑overlap:** set `seq_len == stride` so windows do not overlap.
- **Deterministic:** record and reuse the seed bundle (`python`, `numpy`, `torch`) and bootstrap seed (when applicable).
- **Dedupe:** deduplication is allowed for pilots/probes; **release evidence uses strict non‑overlap on the full plan**.
- **Exact pairing:** preview/final counts must match and the edited run must reuse baseline window IDs; mixing schedules voids the paired Δlog guarantees.

## Pairing Reuse (baseline → edited)

- The edited run pins windows via the baseline report.
- Certificate lints pairing and overlap:
  - `dataset.windows.stats.window_match_fraction == 1.0`
  - `dataset.windows.stats.window_overlap_fraction == 0.0`
- CI/Release abort if counts differ, pairing < 1.0, or overlap > 0.0.

## Tier Minima (runner defaults)

Sane defaults enforced by the runner per tier (guard-rail floors; profiles may
request higher counts):

| Tier         | Preview Windows | Final Windows | Bootstrap Replicates |
|--------------|------------------|---------------|----------------------|
| Conservative | 220              | 220           | 1,500                |
| Balanced     | 180              | 180           | 1,200                |
| Aggressive   | 140              | 140           |   800                |

These minima are derived from half‑width targets on paired Δlog‑loss (see
Tier v1.0 Calibration). CI/Release profiles treat shortfalls as hard errors;
dev flows surface warnings but still record coverage in the cert.

## Runtime Contract (certificate)

- Window plan: `dataset.windows.stats.{requested_preview,requested_final,actual_preview,actual_final}`
- Pairing/overlap: `dataset.windows.stats.{window_match_fraction,window_overlap_fraction,paired_windows}`
- Bootstrap coverage: `dataset.windows.stats.bootstrap.{replicates,seed}` meets/exceeds the tier floor (profiles may request higher counts)

## Observability

- Pairing and coverage appear in both the Markdown report and the JSON
  certificate, enabling auditors to verify schedule integrity.

## Assumptions & Scope

- Applies to **evaluation (inference) schedules**; training/edit algorithms may
  alter data flow and are out of scope here.
- Dataset or tokenizer changes that affect tokenization invalidate recorded
  pairing schedules.
- Window pairing must be exact (ID reuse) and non‑overlapping; mixing schedules
  voids paired Δlog guarantees.
- This plan is calibrated for Linux/macOS environments and the tier profiles
  documented in Tier v1.0 Calibration.
