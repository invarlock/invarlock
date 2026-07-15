---
title: Compare & evaluate (BYOE)
---

## Compare & evaluate (BYOE)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Evaluate two checkpoints (baseline vs subject) with a deterministic pairing schedule. |
| **Audience** | Teams and researchers with existing model-edit workflows who want paired evaluation without coupling to a specific edit stack. |
| **Workflow** | Baseline run → Subject run → report with paired windows. |
| **Network** | Offline by default; use `evaluate --allow-network` when a run needs model downloads. |
| **Output** | `evaluation.report.json` + `evaluation_report.md` (+ `runtime.manifest.json` for container-backed outputs). |

InvarLock's primary, most stable path is Compare & evaluate (BYOE): you provide
the baseline and subject checkpoints, and InvarLock produces a report using a
deterministic window schedule. This avoids coupling to any particular edit stack and keeps your
existing tooling intact whether you are validating quantization, pruning,
fine-tuning, or other checkpoint-edit workflows.

That is the production boundary in InvarLock: guards and verifier policy inspect
the resulting subject checkpoint and paired metrics. The external program that
produced the subject remains upstream provenance context. Built-in edit
generation is kept to demo/smoke support; production validation should normally
use BYOE.

## TL;DR

- Produce your baseline and edited checkpoints (any external tool).
- Ensure both use the same tokenizer (InvarLock verify lints tokenizer hash when
  present).
- Run `invarlock evaluate --baseline <baseline> --subject <subject> --baseline-adapter auto --subject-adapter auto`.

By default, `evaluate` runs inside the runtime container. Use `--execution-mode host`
only for host-side workflows that intentionally run model loading on the
host. If you choose that host-side path, verify the resulting report with
`invarlock verify --runtime-provenance host --assurance off ...`. Host mode is
outside strict runtime assurance.

Example (wheel-first, GPT‑2, CPU/MPS friendly; requires `invarlock[hf]` or equivalent HF extra):

```bash
TINY_GPT2_REVISION=REPLACE_WITH_40_TO_64_CHARACTER_LOWERCASE_COMMIT
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline sshleifer/tiny-gpt2 \
  --subject /path/to/your/edited-model \
  --baseline-revision "$TINY_GPT2_REVISION" \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --out runs/eval_smoke \
  --report-out reports/eval_smoke
```

From a repository checkout, replace the flag-only example above with
`--preset configs/presets/...` when a checked-in preset is appropriate. Preset
paths under `configs/` are not shipped in wheel installs.

Outputs:

- JSON report: `reports/eval_smoke/evaluation.report.json`
- Markdown report: `reports/eval_smoke/evaluation_report.md`
- Runtime manifest: `reports/eval_smoke/runtime.manifest.json`

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
TINY_GPT2_REVISION=REPLACE_WITH_40_TO_64_CHARACTER_LOWERCASE_COMMIT
# 1) Produce a reusable baseline report once (writes runs/baseline_once/source/<timestamp>/report.json)
INVARLOCK_STORE_EVAL_WINDOWS=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline sshleifer/tiny-gpt2 \
  --subject sshleifer/tiny-gpt2 \
  --baseline-revision "$TINY_GPT2_REVISION" \
  --subject-revision "$TINY_GPT2_REVISION" \
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
  --baseline-revision "$TINY_GPT2_REVISION" \
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
- Portable: a report bundle carries recorded metadata and its runtime manifest;
  signer/image trust anchors must come from an independent channel.
- Low maintenance: you can update your edit tools without waiting for InvarLock updates.
- Auditable: the bundle retains the raw baseline, authorized policy pack,
  resolved configuration, model identities, and runtime-image trust anchor
  needed to reproduce the verification decision.

## When to use built-in edits

The evaluator CLI ships a single portable demo edit:

- `quant_rtn` — 8-bit quantize/dequantize simulation (attention-only mode
  available), CPU/MPS friendly; not deployable quantization

Use it for quick local checks. For production edits, prefer Compare & evaluate (BYOE).
The canonical evidence-pack manifest also defines validation-subject scenarios
for magnitude pruning and synthetic low-rank/dense perturbations, plus opt-in
deployable bitsandbytes quantization scenarios on compatible CUDA hosts. The
synthetic generators do not train or merge adapters and do not run a
fine-tuning optimizer.

## Determinism & pairing

InvarLock pairs windows deterministically between baseline and subject runs.
This removes window-selection drift across re-runs. It does not promise
bit-identical metrics across devices, kernels, or dependency versions. Keep
these in mind:

- Match `seq_len` and `stride` between runs; prefer `seq_len = stride` for non‑overlapping windows.
- Keep `preview_n` and `final_n` equal across baseline and subject.
- Use the same tokenizer; `invarlock verify` lints tokenizer hash mismatches when
  present.

Seeds, pairing evidence, report-recorded metadata, and runtime-manifest
binding are surfaced in reports and `runtime.manifest.json` and checked by
`invarlock verify`. Strict verification additionally requires the complete raw
baseline, an independently maintained policy pack, and an independently supplied
`--expected-runtime-image-digest`; matching the digest checks the manifest's
image claim but does not attest actual execution.

## Related Documentation

- [CLI Reference](../reference/cli.md) — Full `evaluate` command options
- [Knowledge & self-edit workflows](knowledge-and-self-edit-workflows.md) — External edit systems as BYOE subject generators
- [Reading a report](reading-report.md) — Understanding output reports
- [Coverage & Pairing (Assurance)](../assurance/02-coverage-and-pairing.md) — Window pairing invariants
- [Determinism Contracts (Assurance)](../assurance/08-determinism-contracts.md) — Seed and reproducibility guarantees
