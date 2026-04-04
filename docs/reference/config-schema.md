# Configuration Schema

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | YAML configuration structure for `invarlock evaluate --preset` and advanced/internal preset-driven flows. |
| **Audience** | CLI users authoring presets or overrides. |
| **Source of truth** | `src/invarlock/core/config_runtime.py`, `src/invarlock/core/config_dependencies.py`, runtime profiles under `invarlock/_data/runtime`. |
| **Network** | Offline by default; use `evaluate --allow-network` when a preset-driven run needs downloads. |
| **Execution** | Model-loading commands run in the runtime container by default; trusted local `invarlock evaluate` runs use `--assurance trusted-local`, while advanced/internal flows may use `INVARLOCK_ALLOW_HOST_EXECUTION=1` or `--allow-host-execution`. |

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
  preview_n: 240
  final_n: 240

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
- **Programmatic access**: `load_config()` returns an explicit mapping-backed
  `InvarLockConfig`. Use `cfg["model"]["id"]` or
  `cfg.require_section("model")["id"]`; attribute-style access is unsupported.
- **Unsupported keys**: `edit.kind`, `edit.parameters`, `assurance.*`, and
  `guards.{spectral,rmt}.mode` are rejected to keep the config surface explicit.

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
report shows `preview_n=240` because the CI profile overrides the YAML counts.

### Config → Report → report → Verify

| Config area | Report fields | report fields | Verify gates |
| --- | --- | --- | --- |
| `model.*` | `report.meta.{model_id,adapter,device}` | `report.meta.{model_id,adapter,device}` | Schema only. |
| `dataset.*` | `report.data.*`, `report.dataset.windows.stats`, `report.provenance.provider_digest` | `report.dataset.*`, `report.provenance.provider_digest` | Pairing + provider digest checks (CI/Release). |
| `eval.*` | `report.metrics.primary_metric` | `report.primary_metric`, `validation.*`, `primary_metric_tail` | Ratio/counts + drift band (CI/Release). |
| `guards.*` | `report.guards[]`, `report.guard_overhead` | `report.spectral/rmt/variance`, `resolved_policy.*`, `guard_overhead` | Measurement contracts + overhead (Release). |
| `auto.*` / `--profile` | `report.meta.auto`, `report.context.profile` | `report.auto`, `report.meta.profile` | Schema only. |
| `output.*` | `report.artifacts.*` | `report.artifacts.*` | Schema only. |

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
  #   bitwidth: 8
```

### Dataset

```yaml
dataset:
  provider: wikitext2
  split: validation
  seq_len: 512
  stride: 512
  preview_n: 240
  final_n: 240
  seed: 42
```

Supported providers: `wikitext2`, `synthetic`, `hf_text`, `local_jsonl`,
`vision_text`, `hf_seq2seq`, `local_jsonl_pairs`, `seq2seq`.

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

Only `edit.plan` is supported for built-in edit configuration.

### Auto policy hints

```yaml
auto:
  enabled: true
  tier: balanced
  probes: 0
  target_pm_ratio: 2.0
```

### Primary metric policy hints

```yaml
primary_metric:
  acceptance_range: {min: 0.95, max: 1.10}
  drift_band: {min: 0.90, max: 1.20}
  overhead_threshold: 0.01
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
  run:
    strict_guard_prepare: true
    strict_eval: true
    skip_overhead_check: false   # release/ci explicit skip marker
    tiny_relax: false            # dev/demo-only relaxed gating
  eval:
    strict_errors: true
    tiny_relax: false
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

- **Unsupported keys rejected**: remove `edit.kind`, `edit.parameters`,
  `assurance.*`, or guard `mode` keys.
- **Provider not found**: verify `dataset.provider` and install `invarlock[eval]`.
- **Preset drift**: run `python scripts/check_config_schema_sync.py` after edits.

## Observability

- `report.meta.config` captures the `RunConfig` applied by the runner.
- `report.context` records `profile`/`auto` context used for tier resolution.
- reports include resolved policy snapshots under `resolved_policy.*`.

## Related Documentation

- [CLI Reference](cli.md)
- [Dataset Providers](datasets.md)
- [Tier Policy Catalog](tier-policy-catalog.md)
- [Environment Variables](env-vars.md)
