---
title: Compare & evaluate (BYOE)
---

## Compare & evaluate (BYOE)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | evaluate two checkpoints (baseline vs subject) with deterministic pairing. |
| **Audience** | Teams and researchers with existing model-edit workflows who want paired evaluation without coupling to a specific edit stack. |
| **Workflow** | Baseline run → Subject run → report with paired windows. |
| **Network** | Offline by default; use `evaluate --allow-network` when a run needs model downloads. |
| **Output** | `evaluation.report.json` + `evaluation_report.md` (+ `runtime.manifest.json` for container-backed outputs). |

InvarLock's primary, most stable path is Compare & evaluate (BYOE): you provide the
baseline and the subject checkpoints, and InvarLock produces a deterministic
report. This avoids coupling to any particular edit stack and keeps your
existing tooling intact whether you are validating quantization, pruning,
fine-tuning, or other checkpoint-edit workflows.

That is the production boundary in InvarLock: guards and verifier policy inspect
the resulting subject checkpoint and paired metrics, not the external program
that produced the subject. Built-in edit generation is kept to demo/smoke
support; production validation should normally use BYOE.

## TL;DR

- Produce your baseline and edited checkpoints (any external tool).
- Ensure both use the same tokenizer (InvarLock verify lints tokenizer hash when
  present).
- Run `invarlock evaluate --baseline <baseline> --subject <subject> --baseline-adapter auto --subject-adapter auto`.

By default, `evaluate` runs inside the runtime container. Use `--execution-mode host`
only for host-side workflows that intentionally run model loading on the
host. If you choose that host-side path, verify the resulting report with
`invarlock verify --runtime-provenance host --assurance off ...`.

Example (wheel-first, GPT‑2, CPU/MPS friendly; requires `invarlock[hf]` or equivalent HF extra):

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline sshleifer/tiny-gpt2 \
  --subject /path/to/your/edited-model \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --out runs/eval_smoke \
  --report-out reports/eval_smoke
```

Repo maintainers who want a repo-owned preset can replace the flag-only example
above with `--preset configs/presets/...`, but that preset path is not shipped
in wheel installs.

Outputs:

- JSON report: `reports/eval_smoke/evaluation.report.json`
- Markdown report: `reports/eval_smoke/evaluation_report.md`
- Runtime provenance: `reports/eval_smoke/runtime.manifest.json`

## Reuse a baseline report (skip baseline evaluation)

When evaluating many subjects against the same baseline, you can reuse a single
baseline `report.json` file and skip the baseline evaluation portion by passing
`--baseline-report` the exact emitted report path.

Requirements:

- Baseline report must be from a no-op run (`edit.name == "noop"`).
- Baseline report must include stored evaluation windows (set `INVARLOCK_STORE_EVAL_WINDOWS=1` when generating it).
- The baseline report must match the intended baseline model, `--profile`, `--tier`, adapter family,
  assurance mode, and dataset/window-plan fields.

Example:

```bash
# 1) Produce a reusable baseline report once (writes runs/baseline_once/source/<timestamp>/report.json)
INVARLOCK_STORE_EVAL_WINDOWS=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline sshleifer/tiny-gpt2 \
  --subject sshleifer/tiny-gpt2 \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --tier balanced \
  --out runs/baseline_once \
  --report-out reports/eval_baseline_once

# 2) Reuse it for many subjects (skips baseline evaluation)
#    Use the exact report path from step 1, e.g. runs/baseline_once/source/<timestamp>/report.json
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline-report runs/baseline_once/source/<timestamp>/report.json \
  --baseline sshleifer/tiny-gpt2 \
  --subject /path/to/your/edited-model \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --tier balanced \
  --out runs/eval_subject_1 \
  --report-out reports/eval_subject_1
```

## Pairing invariants

- InvarLock pairs windows from the baseline run and the edited run. For
  comparability:
  - Sequence length and stride must match.
  - Window counts (preview/final) must match.
  - Tokenizer hash should match; the verify command fails if both hashes are present and differ.

Use the same dataset/evaluation configuration on both sides, whether that means
repeating the same explicit flags or reusing the same preset (`--preset`), and
keep `seq_len=stride` for deterministic non-overlapping windows.

## Why Compare & evaluate?

- Stable: your edit stack remains yours; InvarLock focuses on gates and evidence.
- Portable: reports are self-contained artifacts with provenance.
- Low maintenance: you can update your edit tools without waiting for InvarLock updates.
- Auditable: public BYOE fixtures under `public_evidence/byoe_examples/` show
  dense magnitude pruning and LoRA-merge style subjects verifying through the
  same release gate.

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

Determinism, pairing math, provenance, and runtime provenance are surfaced in
reports and `runtime.manifest.json` and revalidated by `invarlock verify`.

## Related Documentation

- [CLI Reference](../reference/cli.md) — Full `evaluate` command options
- [Reading a report](reading-report.md) — Understanding output reports
- [Coverage & Pairing (Assurance)](../assurance/02-coverage-and-pairing.md) — Window pairing invariants
- [Determinism Contracts (Assurance)](../assurance/08-determinism-contracts.md) — Seed and reproducibility guarantees
