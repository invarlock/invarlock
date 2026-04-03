---
title: Compare & evaluate (BYOE)
---

## Compare & evaluate (BYOE)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | evaluate two checkpoints (baseline vs subject) with deterministic pairing. |
| **Audience** | Users with existing edit pipelines who want paired evaluation without coupling. |
| **Workflow** | Baseline run → Subject run → report with paired windows. |
| **Network** | Offline by default; `INVARLOCK_ALLOW_NETWORK=1` for model downloads. |
| **Output** | `evaluation.report.json` + `evaluation_report.md` (+ `runtime.manifest.json` for attested outputs). |

InvarLock's primary, most stable path is Compare & evaluate (BYOE): you provide the
baseline and the subject checkpoints, and InvarLock produces a deterministic
report. This avoids coupling to any particular edit stack and keeps your
existing tooling intact.

## TL;DR

- Produce your baseline and edited checkpoints (any external tool).
- Ensure both use the same tokenizer (InvarLock verify lints tokenizer hash when
  present).
- Run `invarlock evaluate --baseline <baseline> --subject <subject> --adapter auto`.

By default, `evaluate` runs inside the runtime container. Use `--assurance trusted-local`
only for trusted local workflows that intentionally run model loading on the
host. If you choose that trusted local path, verify the resulting report with
`invarlock verify --assurance trusted-local ...`.

Example (GPT‑2, CPU/MPS friendly; requires `invarlock[hf]` or equivalent HF extra):

```bash
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline sshleifer/tiny-gpt2 \
  --subject /path/to/your/edited-model \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --out runs/eval_smoke \
  --report-out reports/eval_smoke
```

Outputs:

- JSON report: `reports/eval_smoke/evaluation.report.json`
- Markdown report: `reports/eval_smoke/evaluation_report.md`
- Runtime attestation: `reports/eval_smoke/runtime.manifest.json`

## Reuse a baseline report (skip baseline evaluation)

When evaluating many subjects against the same baseline, you can reuse a single
baseline `report.json` file and skip Phase 1/3 (baseline evaluation) by passing
`--baseline-report` the exact emitted report path.

Requirements:

- Baseline report must be from a no-op run (`edit.name == "noop"`).
- Baseline report must include stored evaluation windows (set `INVARLOCK_STORE_EVAL_WINDOWS=1` when generating it).
- The baseline report must match the intended `--profile`, `--tier`, and adapter family.

Example:

```bash
# 1) Produce a reusable baseline report once (writes runs/baseline_once/source/<timestamp>/report.json)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_STORE_EVAL_WINDOWS=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline sshleifer/tiny-gpt2 \
  --subject sshleifer/tiny-gpt2 \
  --adapter auto \
  --profile ci \
  --tier balanced \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --out runs/baseline_once \
  --report-out reports/eval_baseline_once

# 2) Reuse it for many subjects (skips baseline evaluation)
#    Use the exact report path from step 1, e.g. runs/baseline_once/source/<timestamp>/report.json
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline-report runs/baseline_once/source/<timestamp>/report.json \
  --baseline sshleifer/tiny-gpt2 \
  --subject /path/to/your/edited-model \
  --adapter auto \
  --profile ci \
  --tier balanced \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --out runs/eval_subject_1 \
  --report-out reports/eval_subject_1
```

## Pairing invariants

- InvarLock pairs windows from the baseline run and the edited run. For
  comparability:
  - Sequence length and stride must match.
  - Window counts (preview/final) must match.
  - Tokenizer hash should match; the verify command fails if both hashes are present and differ.

Use the same preset (`--preset`), and keep `seq_len=stride` for deterministic
non-overlapping windows.

## Why Compare & evaluate?

- Stable: your edit stack remains yours; InvarLock focuses on gates and evidence.
- Portable: reports are self-contained artifacts with provenance.
- Low maintenance: you can update your edit tools without waiting for InvarLock updates.

## When to use built-in edits

InvarLock ships a single built-in edit tuned for portable smokes:

- `quant_rtn` — 8‑bit (attention‑only mode available), CPU/MPS friendly

Use it for quick local checks. For production edits, prefer Compare & evaluate (BYOE).

## Determinism & pairing

InvarLock pairs windows deterministically between baseline and subject runs. This
ensures reproducible ratios and CI across machines and re‑runs. Keep these in
mind:

- Match `seq_len` and `stride` between runs; prefer `seq_len = stride` for non‑overlapping windows.
- Keep `preview_n` and `final_n` equal across baseline and subject.
- Use the same tokenizer; `invarlock verify` lints tokenizer hash mismatches when
  present.

Determinism, pairing math, provenance, and runtime attestation are surfaced in
reports and `runtime.manifest.json` and revalidated by `invarlock verify`.

## Related Documentation

- [CLI Reference](../reference/cli.md) — Full `evaluate` command options
- [Reading a report](reading-report.md) — Understanding output reports
- [Coverage & Pairing (Assurance)](../assurance/02-coverage-and-pairing.md) — Window pairing invariants
- [Determinism Contracts (Assurance)](../assurance/08-determinism-contracts.md) — Seed and reproducibility guarantees
