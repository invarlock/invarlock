# Configuration Schema

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | YAML configuration structure for `invarlock run` and presets. |
| **Audience** | CLI users authoring presets or overrides. |
| **Source of truth** | `src/invarlock/cli/config.py`, runtime profiles under `invarlock/_data/runtime`. |
| **Network** | Offline by default; enable downloads via `INVARLOCK_ALLOW_NETWORK=1`. |

## Quick Start

```yaml
model:
  id: gpt2
  adapter: hf_causal
  device: auto

dataset:
  provider: wikitext2
  seq_len: 512
  stride: 512
  preview_n: 200
  final_n: 200

edit:
  name: quant_rtn
  plan: { bitwidth: 8, clamp_ratio: 0.005 }

guards:
  order: ["invariants", "spectral", "rmt", "variance", "invariants"]

output:
  dir: runs/example
```

## Concepts

- **Profiles and tiers**: `--profile` selects runtime window counts; `--tier`
  resolves guard thresholds from `tiers.yaml`.
- **Defaults merging**: the optional top-level `defaults` mapping is merged into
  the config before execution.
- **Deprecated keys**: `assurance.*` and `guards.{spectral,rmt}.mode` are rejected
  to keep measurement contracts explicit.

**Precedence (highest → lowest)**

1. CLI flags (e.g. `--device`, `--tier`, `--probes`).
2. Profile selection (`--profile ci|release`) — window counts + determinism knobs.
3. YAML config (`-c config.yaml`).
4. `defaults:` block in YAML (DRY base).
5. Packaged runtime defaults (fallback).

### Key override matrix

| Setting | CLI | Profile | YAML | defaults | Winner rule |
| --- | --- | --- | --- | --- | --- |
| `model.device` | `--device` | — | ✅ | ✅ | CLI wins. |
| `dataset.preview_n/final_n` | — | ✅ | ✅ | ✅ | Profile wins. |
| `auto.tier` | `--tier` | — | ✅ | ✅ | CLI wins. |
| `auto.probes` | `--probes` | — | ✅ | ✅ | CLI wins. |

Confirm in `report.meta.device`, `report.meta.auto`, and `report.data.preview_n/final_n`.

**Worked example**: if YAML sets `preview_n: 64` and you run `--profile ci`, the
report shows `preview_n=200` because the CI profile overrides the YAML counts.

### Config → Report → Certificate → Verify

| Config area | Report fields | Certificate fields | Verify gates |
| --- | --- | --- | --- |
| `model.*` | `report.meta.{model_id,adapter,device}` | `certificate.meta.{model_id,adapter,device}` | Schema only. |
| `dataset.*` | `report.data.*`, `report.dataset.windows.stats`, `report.provenance.provider_digest` | `certificate.dataset.*`, `certificate.provenance.provider_digest` | Pairing + provider digest checks (CI/Release). |
| `eval.*` | `report.metrics.primary_metric` | `certificate.primary_metric`, `validation.*`, `primary_metric_tail` | Ratio/counts + drift band (CI/Release). |
| `guards.*` | `report.guards[]`, `report.guard_overhead` | `certificate.spectral/rmt/variance`, `resolved_policy.*`, `guard_overhead` | Measurement contracts + overhead (Release). |
| `auto.*` / `--profile` | `report.meta.auto`, `report.context.profile` | `certificate.auto`, `certificate.meta.profile` | Schema only. |
| `output.*` | `report.artifacts.*` | `certificate.artifacts.*` | Schema only. |

## Reference

### Model

```yaml
model:
  id: <hf_id_or_path>
  adapter: auto
  device: auto
  # extra adapter kwargs (passed to load_model)
  dtype: float16
  trust_remote_code: false
  # Optional: v5-native HF quantization config (e.g., bitsandbytes)
  # quantization_config:
  #   quant_method: bitsandbytes
  #   bits: 8
```

### Dataset

```yaml
dataset:
  provider: wikitext2
  split: validation
  seq_len: 512
  stride: 512
  preview_n: 200
  final_n: 200
  seed: 42
```

Supported providers: `wikitext2`, `synthetic`, `hf_text`, `local_jsonl`,
`hf_seq2seq`, `local_jsonl_pairs`, `seq2seq`.

### Edit (built-in quant_rtn)

```yaml
edit:
  name: quant_rtn
  plan:
    bitwidth: 8
    per_channel: true
    group_size: 128
    clamp_ratio: 0.005
    scope: attn
    max_modules: 12
```

`edit.parameters` is accepted as an alias for `edit.plan` (the CLI normalizes it).

### Auto policy hints

```yaml
auto:
  enabled: true
  tier: balanced
  probes: 0
  target_pm_ratio: 2.0
```

### Guards

```yaml
guards:
  order: ["invariants", "spectral", "rmt", "variance", "invariants"]
  spectral:
    sigma_quantile: 0.95
  rmt:
    epsilon_by_family: { ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01 }
  variance:
    min_gain: 0.0
```

### Context (snapshot controls)

```yaml
context:
  snapshot:
    mode: auto
    ram_fraction: 0.4
    threshold_mb: 768
    disk_free_margin_ratio: 1.2
    temp_dir: /tmp
```

### Output

```yaml
output:
  dir: runs/example
  save_model: false
  model_dir: runs/exports/my_model  # optional
  model_subdir: model               # optional
```

### Metrics

```yaml
eval:
  max_pm_ratio: 1.5
  metric:
    kind: auto            # auto|ppl_causal|ppl_mlm|ppl_seq2seq|accuracy|vqa_accuracy
    reps: 2000
    ci_level: 0.95
```

## Troubleshooting

- **Unknown keys rejected**: remove deprecated `assurance.*` or guard `mode` keys.
- **Provider not found**: verify `dataset.provider` and install `invarlock[eval]`.
- **Preset drift**: run `python scripts/check_config_schema_sync.py` after edits.

## Observability

- `report.meta.config` captures the `RunConfig` applied by the runner.
- `report.context` records `profile`/`auto` context used for tier resolution.
- Certificates include resolved policy snapshots under `resolved_policy.*`.

## Related Documentation

- [CLI Reference](cli.md)
- [Dataset Providers](datasets.md)
- [Tier Policy Catalog](tier-policy-catalog.md)
- [Environment Variables](env-vars.md)
