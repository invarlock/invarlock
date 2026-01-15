---
title: Compare & Certify (BYOE)
---

## Compare & Certify (BYOE)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Certify two checkpoints (baseline vs subject) with deterministic pairing. |
| **Audience** | Users with existing edit pipelines who want certification without coupling. |
| **Workflow** | Baseline run → Subject run → Certificate with paired windows. |
| **Network** | Offline by default; `INVARLOCK_ALLOW_NETWORK=1` for model downloads. |
| **Output** | `evaluation.cert.json` + `evaluation_certificate.md`. |

InvarLock's primary, most stable path is Compare & Certify (BYOE): you provide the
baseline and the subject checkpoints, and InvarLock produces a deterministic
certificate. This avoids coupling to any particular edit stack and keeps your
existing tooling intact.

## TL;DR

- Produce your baseline and edited checkpoints (any external tool).
- Ensure both use the same tokenizer (InvarLock verify lints tokenizer hash when
  present).
- Run `invarlock certify --baseline <baseline> --subject <subject> --adapter auto`.

Example (GPT‑2, CPU/MPS friendly; requires `invarlock[hf]` or equivalent HF extra):

```bash
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline sshleifer/tiny-gpt2 \
  --subject /path/to/your/edited-model \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --out runs/cert_smoke \
  --cert-out reports/cert_smoke
```

Outputs:

- JSON certificate: `reports/cert_smoke/evaluation.cert.json`
- Markdown report: `reports/cert_smoke/evaluation_certificate.md`

## Pairing invariants

- InvarLock pairs windows from the baseline run and the edited run. For
  comparability:
  - Sequence length and stride must match.
  - Window counts (preview/final) must match.
  - Tokenizer hash should match; the verify command fails if both hashes are present and differ.

Use the same preset (`--preset`), and keep `seq_len=stride` for deterministic
non-overlapping windows.

## Why Compare & Certify?

- Stable: your edit stack remains yours; InvarLock focuses on gates and evidence.
- Portable: certificates are self-contained artifacts with provenance.
- Low maintenance: you can update your edit tools without waiting for InvarLock updates.

## When to use built-in edits

InvarLock ships a single built-in edit tuned for portable smokes:

- `quant_rtn` — 8‑bit (attention‑only mode available), CPU/MPS friendly

Use it for quick local checks. For production edits, prefer Compare & Certify (BYOE).

## Determinism & pairing

InvarLock pairs windows deterministically between baseline and subject runs. This
ensures reproducible ratios and CI across machines and re‑runs. Keep these in
mind:

- Match `seq_len` and `stride` between runs; prefer `seq_len = stride` for non‑overlapping windows.
- Keep `preview_n` and `final_n` equal across baseline and subject.
- Use the same tokenizer; `invarlock verify` lints tokenizer hash mismatches when
  present.

Determinism, pairing math, and provenance are surfaced in certificates
(provider and policy digests) and revalidated by `invarlock verify`.
