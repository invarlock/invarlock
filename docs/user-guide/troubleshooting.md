# Troubleshooting

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Consolidated error code reference and troubleshooting guide. |
| **Audience** | Users encountering errors during `certify`, `run`, or `verify` commands. |
| **Exit codes** | `0=success`, `1=generic failure`, `2=schema/config invalid`, `3=hard abort (CI/Release)`. |
| **Source of truth** | `src/invarlock/cli/commands/run.py`, `src/invarlock/cli/commands/verify.py`. |

## Quick Start

```bash
# Check environment and configuration
invarlock doctor --config <config.yaml> --profile ci

# Validate a certificate
invarlock verify reports/cert/evaluation.cert.json --profile ci

# Enable debug output for detailed traces
INVARLOCK_DEBUG_TRACE=1 invarlock run -c <config.yaml>
```

## Error Code Reference

### Pairing Errors (E001–E006)

These errors relate to window pairing, tokenizer consistency, and evidence integrity.

#### E001 — Pairing Schedule Mismatch

| Field | Value |
| --- | --- |
| **Code** | `E001` |
| **Category** | Pairing |
| **Severity** | Hard abort in CI/Release |
| **Exit code** | `3` (CI/Release) or `1` (dev) |

**Triggers:**

- `PAIRING-EVIDENCE-MISSING`: Baseline report path does not exist or cannot be parsed
- `PAIRING-SCHEDULE-MISMATCH`: Window matching fraction ≠ 1.0, window overlap > 0, or counts diverge after stratification
- `PAIRED-WINDOWS-COLLAPSED`: `paired_windows ≤ 0` under paired baseline

**Common causes:**

- Missing or corrupted baseline `report.json`
- Baseline lacks `evaluation_windows` section
- Dataset settings differ between baseline and subject runs
- Sequence length / stride mismatch

**Fixes:**

1. Ensure baseline `report.json` exists and contains `evaluation_windows`
2. Verify dataset settings match: `provider`, `seq_len`, `stride`, `split`
3. Regenerate baseline with the same configuration
4. Check that tokenizer hash matches between runs

**Example error:**

```text
[INVARLOCK:E001] PAIRING-SCHEDULE-MISMATCH: window_match_fraction=0.950
```

---

#### E002 — Tokenizer Digest Mismatch

| Field | Value |
| --- | --- |
| **Code** | `E002` |
| **Category** | Pairing |
| **Severity** | Hard abort in CI/Release |
| **Exit code** | `3` (CI/Release) |

**Triggers:**

- Subject and baseline tokenizers produce different vocabulary hashes

**Common causes:**

- Different tokenizer versions or configurations
- Model updated with new vocabulary
- Trust-remote-code flag inconsistency

**Fixes:**

1. Ensure both runs use the same model checkpoint
2. Pin tokenizer version via `revision` in config
3. Regenerate baseline with current tokenizer

**Example error:**

```text
[INVARLOCK:E002] TOKENIZER-DIGEST-MISMATCH: subject and baseline tokenizers differ
```

---

#### E003 — Mask Parity Mismatch

| Field | Value |
| --- | --- |
| **Code** | `E003` |
| **Category** | Pairing |
| **Severity** | Hard abort in CI/Release |
| **Exit code** | `3` (CI/Release) |

**Triggers:**

- MLM mask positions differ between subject and baseline under identical tokenizers

**Common causes:**

- Different masking seeds between runs
- Baseline generated with different `mask_prob` or `mask_seed`
- Labels in baseline report corrupted or missing

**Fixes:**

1. Ensure `eval.loss.seed` matches between runs
2. Regenerate baseline with consistent masking configuration
3. Verify baseline contains MLM labels in `evaluation_windows`

**Example error:**

```text
[INVARLOCK:E003] MASK-PARITY-MISMATCH: mask positions differ under matched tokenizers
```

---

#### E004 — Provider Digest Missing

| Field | Value |
| --- | --- |
| **Code** | `E004` |
| **Category** | Pairing |
| **Severity** | Hard abort in CI/Release |
| **Exit code** | `3` (CI/Release) |

**Triggers:**

- Subject or baseline is missing the provider digest (ids/tokenizer hash)

**Common causes:**

- Old baseline report generated before digest tracking was added
- Report truncated or missing `provenance` section
- Windows not materialized due to `INVARLOCK_STORE_EVAL_WINDOWS=0`

**Fixes:**

1. Regenerate baseline with current InvarLock version
2. Ensure `INVARLOCK_STORE_EVAL_WINDOWS=1` (default)
3. Check `report.provenance.provider_digest` exists

**Example error:**

```text
[INVARLOCK:E004] PROVIDER-DIGEST-MISSING: subject or baseline missing ids/tokenizer digest
```

---

#### E006 — Window IDs Digest Mismatch

| Field | Value |
| --- | --- |
| **Code** | `E006` |
| **Category** | Pairing |
| **Severity** | Hard abort in CI/Release |
| **Exit code** | `3` (CI/Release) |

**Triggers:**

- Subject and baseline window IDs differ (different windows selected)

**Common causes:**

- Different dataset splits or stratification seeds
- Capacity constraints forced window reduction
- Baseline generated with different `preview_n`/`final_n`

**Fixes:**

1. Use identical `dataset.seed` and window counts
2. Verify capacity allows requested windows
3. Regenerate baseline with matching configuration

**Example error:**

```text
[INVARLOCK:E006] IDS-DIGEST-MISMATCH: subject and baseline window IDs differ
```

---

### Primary Metric Errors (E111)

#### E111 — Primary Metric Degraded

| Field | Value |
| --- | --- |
| **Code** | `E111` |
| **Category** | Quality |
| **Severity** | Certificate emitted, then hard abort in CI/Release |
| **Exit code** | `3` (CI/Release) |

**Triggers:**

- Primary metric (perplexity/accuracy) is non-finite (NaN, Inf)
- Primary metric degraded beyond acceptable ratio

**Common causes:**

- Numerical instability (overflow/underflow)
- Model weights corrupted by edit
- Insufficient precision (try float32)
- Empty or malformed evaluation windows

**Fixes:**

1. Force `dtype: float32` in model config (alias: `torch_dtype`)
2. Reduce batch size if memory-constrained
3. Use an accelerator (CUDA/MPS) for better precision
4. Lower `plan.max_modules` to reduce edit scope
5. Check that evaluation windows contain valid tokens

**Example error:**

```text
[INVARLOCK:E111] Primary metric degraded or non-finite (preview=inf, final=nan)
```

**Notes:**

- `invarlock certify` always emits a certificate before exiting on E111
- `invarlock run` logs a warning for non-finite PM but does not raise E111

---

### Verification Errors (E601)

#### E601 — Certificate Verification Failed

| Field | Value |
| --- | --- |
| **Code** | `E601` |
| **Category** | Verification |
| **Severity** | Hard abort |
| **Exit code** | `3` |

**Triggers:**

- Certificate fails schema validation
- Pairing math recomputation fails
- Gate checks fail in CI/Release profile

**Common causes:**

- Certificate JSON corrupted or hand-edited
- Schema version mismatch (`schema_version` ≠ `v1`)
- Missing required fields (`run_id`, `meta`, `dataset`, etc.)

**Fixes:**

1. Regenerate certificate via `invarlock report --format cert`
2. Ensure certificate is unmodified from generation
3. Check `schema_version` is `"v1"`

**Example error:**

```text
[INVARLOCK:E601] Certificate schema validation failed
```

---

## Exit Code Summary

| Exit Code | Meaning | Typical causes |
| --- | --- | --- |
| `0` | Success | Run/verification passed all gates |
| `1` | Generic failure | Unknown error, missing dependencies |
| `2` | Schema/config invalid | YAML parse error, invalid config keys, `ValidationError` |
| `3` | Hard abort | E001–E006, E111, E601 in CI/Release profile |

## Common Issues

### Network Blocked

**Symptom:** Downloads fail silently or model loading hangs.

**Fix:**

```bash
INVARLOCK_ALLOW_NETWORK=1 invarlock certify --baseline gpt2 --subject gpt2
```

### Dependency Missing

**Symptom:** `ModuleNotFoundError` for `torch`, `transformers`, etc.

**Fix:**

```bash
pip install "invarlock[hf]"      # HF adapters + eval
pip install "invarlock[guards]"  # Guard math
pip install "invarlock[adapters]" # All adapters
```

### Calibration Data Not Indexable

**Symptom:** Runner fails with "calibration data must be indexable."

**Fix:**

```bash
INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE=1 invarlock run -c <config.yaml>
```

### Guard Prepare Failures

**Symptom:** Guard initialization fails in strict mode.

**Fix (local debugging only):**

```bash
INVARLOCK_GUARD_PREPARE_STRICT=0 invarlock run -c <config.yaml>
```

### Non-Finite Metrics on CPU

**Symptom:** Perplexity returns `inf` or `nan` on CPU runs.

**Fixes:**

1. Use an accelerator when available: `--device cuda` or `--device mps`
2. Force higher precision: `dtype: float32` in config (alias: `torch_dtype`)
3. Reduce edit scope or batch size

## Debug Tools

### Doctor Command

Run comprehensive environment checks:

```bash
invarlock doctor --config <config.yaml> --profile ci --strict
```

### Debug Trace

Enable detailed logging:

```bash
INVARLOCK_DEBUG_TRACE=1 invarlock run -c <config.yaml>
```

### Plugins Inspection

List available adapters/guards/edits:

```bash
invarlock plugins list --verbose
invarlock plugins adapters --explain hf_causal
```

## Related Documentation

- [CLI Reference](../reference/cli.md)
- [Configuration Schema](../reference/config-schema.md)
- [Dataset Providers](../reference/datasets.md)
- [Environment Variables](../reference/env-vars.md)
- [Certificates](../reference/certificates.md)
