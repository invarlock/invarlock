# Troubleshooting Guide

This guide covers common issues, error codes, and their solutions when using
InvarLock.

## Error Codes

InvarLock uses structured error codes prefixed with `[INVARLOCK:EXXX]`. These
codes appear in CLI output and help identify specific failure conditions.

### Pairing Errors (E001)

**Error:** `[INVARLOCK:E001] Pairing schedule failure`

**Causes:**

- `window_match_fraction != 1.0` — baseline and subject windows don't align
- `window_overlap_fraction > 0` — unexpected overlap in paired windows
- Window counts diverge after stratification
- `paired_windows <= 0` — no windows paired
- Run is unpaired when baseline is provided

**Solutions:**

1. Ensure baseline `report.json` contains `evaluation_windows` data
2. Use identical `dataset.*` settings for baseline and subject runs
3. Re-run with `--baseline` pointing to a valid baseline report
4. Check `dataset.seed` matches between baseline and subject configs

### Primary Metric Errors (E111)

**Error:** `[INVARLOCK:E111] Primary metric degraded or non-finite`

**Causes:**

- Model outputs contain NaN or Inf values
- Severe numeric instability during evaluation
- Incompatible dtype causing overflow

**Solutions:**

1. Force float32 precision: add `torch_dtype: float32` to model config
2. Reduce evaluation batch size
3. Use accelerator (GPU/MPS) instead of CPU for better precision
4. Lower `plan.max_modules` in edit config to reduce edit severity

### Schema Errors (Exit Code 2)

**Error:** Certificate validation failed with exit code 2

**Causes:**

- Missing required certificate fields
- Invalid `validation.*` keys (not in allow-list)
- Schema version mismatch

**Solutions:**

1. Regenerate certificate with latest InvarLock version
2. Remove custom `validation.*` keys not in the schema allow-list
3. Check `schema_version` is `"v1"`

## Common Issues

### Dependency Missing

**Symptom:** `DEPENDENCY-MISSING` errors during adapter load or eval

**Solution:** Install the required extras:

```bash
# Core HF adapters + evaluation
pip install "invarlock[hf]"

# Individual extras
pip install "invarlock[adapters]"   # Model adapters
pip install "invarlock[guards]"     # Guard math
pip install "invarlock[eval]"       # Dataset providers
pip install "invarlock[gpu]"        # Bitsandbytes quantization
pip install "invarlock[awq]"        # AWQ adapters
pip install "invarlock[gptq]"       # GPTQ adapters
```

### Network Downloads Blocked

**Symptom:** Model or dataset download fails silently

**Solution:** Enable network access for the command:

```bash
INVARLOCK_ALLOW_NETWORK=1 invarlock certify --baseline gpt2 --subject gpt2
```

For cached-only operation, ensure models/datasets are pre-downloaded or use
`HF_DATASETS_OFFLINE=1` to enforce local cache.

### Calibration Data Not Indexable

**Symptom:** `Calibration data not indexable` warning

**Solution:** Either:

1. Pass calibration data as a list/sequence (not an iterator)
2. Set `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE=1` to allow iterator materialization

### Guard Prepare Failures

**Symptom:** Guard prepare fails in CI/Release profile

**Solutions:**

1. For local debugging only: `INVARLOCK_GUARD_PREPARE_STRICT=0`
2. Adjust guard policies in config to be more permissive
3. Check model structure matches guard requirements (e.g., layer count)

### Quantized Model Device Errors

**Symptom:** `.to()` errors with `hf_bnb`, `hf_awq`, or `hf_gptq` adapters

**Solution:** Quantized adapters manage device placement automatically. Avoid
explicit `.to()` calls on loaded models. Let the adapter handle placement via
`device_map="auto"`.

### Plugin List Empty

**Symptom:** `invarlock plugins list` shows no plugins

**Solution:** Unset environment variables that disable discovery:

```bash
unset INVARLOCK_DISABLE_PLUGIN_DISCOVERY
unset INVARLOCK_MINIMAL
```

### Spectral/RMT Instability

**Symptom:** Frequent spectral or RMT guard warnings/failures

**Solutions:**

1. Lower `sigma_quantile` (e.g., from 0.95 to 0.90)
2. Increase `deadband` to tolerate more noise
3. Narrow `scope` to monitor only relevant families (e.g., `scope: ffn`)
4. Review calibrated thresholds in `tier-policy-catalog.md`

### Variance Guard Never Enables

**Symptom:** VE correction never applies despite A/B testing

**Solution:** Check the predictive gate results in the report:

```bash
jq '.guards[] | select(.name == "variance") | .metrics' runs/*/report.json
```

Look for `predictive_gate` and `ab_gain` values. The A/B gate may require more
calibration windows or a different `min_effect_lognll` threshold.

## Exit Codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Generic failure |
| 2 | Schema/config/validation failure |
| 3 | Hard abort (`[INVARLOCK:EXXX]` in CI/Release profile) |

## Diagnostic Commands

### Check Environment

```bash
invarlock doctor --profile ci
```

### List Available Plugins

```bash
invarlock plugins list --verbose
```

### Explain Gate Decisions

```bash
invarlock report explain \
  --report runs/subject/report.json \
  --baseline runs/baseline/report.json
```

### Validate Certificate

```bash
invarlock verify reports/cert/evaluation.cert.json --profile ci
```

### Get JSON Output for Scripting

```bash
invarlock verify reports/cert/evaluation.cert.json --json
```

## Getting Help

If you encounter issues not covered here:

1. Check the [CLI Reference](../reference/cli.md) for command options
2. Review [Environment Variables](../reference/env-vars.md) for toggles
3. Consult [Guard Contracts](../assurance/04-guard-contracts.md) for gate logic
4. Open an issue with:
   - InvarLock version (`invarlock version`)
   - Full command and config used
   - Error message and exit code
   - Relevant `report.json` excerpts (redact sensitive data)

## Related Documentation

- [CLI Reference](../reference/cli.md)
- [Environment Variables](../reference/env-vars.md)
- [Configuration Schema](../reference/config-schema.md)
- [Certificate Schema](../reference/certificate-schema.md)
- [Guard Contracts](../assurance/04-guard-contracts.md)
