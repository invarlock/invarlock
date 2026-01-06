# Configuration Schema

InvarLock pipelines are defined with YAML. The core sections are summarized below:

```yaml
model:                # Model identifier and adapter
  id: gpt2
  adapter: hf_gpt2
  device: auto

dataset:              # Data source and evaluation windows
  provider: wikitext2
  seq_len: 768
  stride: 768
  preview_n: 200
  final_n: 200
  seed: 42

edit:                 # Edit operation and plan parameters
  name: quant_rtn
  plan:
    bitwidth: 8                 # INT8 only (built-in RTN)
    per_channel: true           # per‑channel quantization
    group_size: 128             # reserved; currently ignored for built-in RTN
    clamp_ratio: 0.005          # percentile clamp for outliers
    scope: attn                 # attn | ffn | all
    max_modules: 12             # optional cap for CI smokes

auto:                 # Auto-tuning surface
  enabled: true
  tier: balanced
  probes: 0

guards:               # Guard chain and per-guard overrides
  order: ["invariants", "spectral", "rmt", "variance", "invariants"]
  # Optional per-guard measurement knobs (recorded into the certificate):
  # spectral:
  #   estimator: {type: power_iter, iters: 4, init: ones}
  # rmt:
  #   epsilon_by_family: {ffn: 0.10, attn: 0.08, embed: 0.12, other: 0.12}
  #   estimator: {type: power_iter, iters: 3, init: ones}
  #   activation:
  #     sampling:
  #       windows: {count: 8, indices_policy: evenly_spaced}

output:               # Artifact destinations
  dir: runs/examples/quant_rtn
  save_model: false
  save_report: true
```

Refer to `src/invarlock/config.py` for the authoritative defaults.

---

## Context (snapshot controls)

Retries reuse a single loaded model and reset its state via snapshot/restore between attempts. Control the snapshot strategy here (config takes precedence over env):

```yaml
context:
  snapshot:
    mode: auto                 # auto | bytes | chunked
    ram_fraction: 0.4          # choose chunked when snapshot ≥ fraction × available RAM
    threshold_mb: 768          # fallback threshold when RAM not detectable
    disk_free_margin_ratio: 1.2  # require 20% headroom on disk for chunked
    temp_dir: /tmp             # where to place chunked snapshots
```

Environment overrides (lower precedence): `INVARLOCK_SNAPSHOT_MODE`, `INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION`, `INVARLOCK_SNAPSHOT_THRESHOLD_MB`.

Note: if you force `mode: bytes` and the adapter's in-memory snapshot fails, the CLI will attempt `chunked` snapshotting when supported; otherwise it falls back to reload-per-attempt.

---

## Dataset (provider-specific keys)

Dataset providers may accept extra keys. For example, the generic HF text provider:

```yaml
dataset:
  provider: hf_text
  dataset_name: wikitext         # HF dataset name
  config_name: wikitext-2-raw-v1 # optional
  text_field: text               # defaults to 'text'
  split: validation
  preview_n: 64
  final_n: 64
```

Online mode requires `INVARLOCK_ALLOW_NETWORK=1` for the first run to populate the HuggingFace cache. For offline usage, pre-download via `datasets` and set `HF_DATASETS_OFFLINE=1`.

---

## Metric and Provider Selection

Configure task-agnostic evaluation via `eval.metric.*` and pick the dataset provider kind explicitly. Config takes precedence over env flags; if both are set, config wins and a deprecation notice is logged for `INVARLOCK_METRIC_V1`.

```yaml
eval:
  metric:
    kind: auto            # auto|ppl_causal|ppl_mlm|ppl_seq2seq|accuracy|vqa_accuracy
    reps: 2000            # optional bootstrap reps for display/meta
    ci_level: 0.95        # optional CI level for display/meta

dataset:
  provider: wikitext2     # wikitext2|hf_text|synthetic
  # provider-specific keys may follow (e.g., dataset_name, config_name, ...)
```

Resolution order for the metric and provider:

- CLI args → config → `ModelProfile` defaults

Family defaults:

- GPT/LLaMA → ppl_causal + wikitext2
- BERT → ppl_mlm + hf_text
- T5/Enc‑Dec → ppl_seq2seq + hf_text (task-specific)

---

## Quant RTN Edit (built-in)

RTN (Round-To-Nearest) quantization is the only built‑in edit. Use it for portable smokes and CI checks; prefer Compare & Certify (BYOE) for production workflows.

```yaml
edit:
  name: quant_rtn
  plan:
    bitwidth: 8
    per_channel: true
    group_size: 128
    clamp_ratio: 0.005
    scope: attn
```


## Output Export (HF loadable)

Control export of a HF‑loadable model directory from runs. When enabled, the
pipeline saves a `save_pretrained` directory you can pass later to
`--baseline/--subject` in `invarlock certify`.

Precedence for destination (first present wins):

1. `output.model_dir` (absolute or relative to the run directory)
2. `INVARLOCK_EXPORT_DIR` (env; absolute or relative)
3. `output.model_subdir` (subdirectory under the run directory)
4. default: `model` under the run directory

```yaml
output:
  save_model: true          # enable exporting a HF directory
  model_dir: runs/exports/my_quant  # optional explicit path (abs or relative)
  # or, provide a subdirectory name under the run dir
  # model_subdir: model
```

Environment override:

```bash
INVARLOCK_EXPORT_MODEL=1 INVARLOCK_EXPORT_DIR=exports/quant invarlock run -c ...
```



## Keeping Docs and CLI in Sync

When updating the CLI help or YAML configuration options, run
`scripts/check_config_schema_sync.py` to ensure the rendered schema fragments
remain present in the documentation set (either here or in `README.md`). The
script exits non-zero if any of the top-level configuration sections are
missing, which keeps MkDocs content and CLI help text aligned.

```bash
python scripts/check_config_schema_sync.py
```

Add this check to your local workflow (or automation) whenever new config keys,
CLI options, or documentation examples land to avoid drift between the CLI help
and the published schema.
