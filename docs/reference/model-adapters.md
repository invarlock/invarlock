# Model Adapters

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Load models, describe structure, and snapshot/restore state for edits and guards. |
| **Audience** | CLI users choosing `model.adapter` and Python callers instantiating adapters. |
| **Supported surface** | Core HF adapters, auto-match adapters, and Linux-only quantized adapters. |
| **Requires** | `invarlock[adapters]` or `invarlock[hf]` for core HF adapters; `invarlock[onnx]` for `hf_onnx`; `invarlock[gpu]`, `invarlock[awq]`, `invarlock[gptq]` for quantized adapters. |
| **Network** | Offline by default; set `INVARLOCK_ALLOW_NETWORK=1` for model downloads. |
| **Inputs** | `model.id` (HF repo or local path), adapter name, device. |
| **Outputs / Artifacts** | Loaded model object; optional snapshots; exported model directories when enabled. |
| **Source of truth** | `src/invarlock/adapters/*`, `src/invarlock/plugins/hf_*_adapter.py`. |

## Quick Start

```bash
# Install core HF adapters + evaluation stack
pip install "invarlock[hf]"

# Inspect adapter availability
invarlock plugins adapters

# Compare & Certify with adapter auto-selection
INVARLOCK_ALLOW_NETWORK=1 invarlock certify \
  --baseline gpt2 \
  --subject gpt2 \
  --adapter auto
```

```python
from invarlock.adapters import HF_Causal_Auto_Adapter

adapter = HF_Causal_Auto_Adapter()
model = adapter.load_model("gpt2", device="auto")
print(adapter.describe(model)["model_type"])
```

## Concepts

- **Adapters hide model-specific logic**: they handle loading, structure description,
  and snapshot/restore so edits/guards stay model-agnostic.
- **Auto adapters** (`hf_causal_auto`, `hf_mlm_auto`) resolve a concrete adapter
  from the model’s `config.json`. Remote models still require network to fetch
  the config on first use.
- **Quantized adapters** (`hf_bnb`, `hf_awq`, `hf_gptq`) handle their own device
  placement; avoid calling `.to(...)` on the loaded model.
- **Snapshot strategy**: HF adapters expose `snapshot`/`restore` and
  `snapshot_chunked`/`restore_chunked` (large-model friendly). The CLI selects the
  strategy automatically via `context.snapshot.*`.

### Auto adapter mapping

| `model_type` family | Adapter |
| --- | --- |
| llama / mistral / qwen / yi | `hf_llama` |
| gpt2 / opt / neo-x | `hf_gpt2` |
| bert / roberta | `hf_bert` |

Auto inspects `config.model_type`; remote models may need network for config.

Capability matrix (at a glance)

| Adapter family | Snapshot/restore | Guard compatibility | Platform |
| --- | --- | --- | --- |
| HF PyTorch (`hf_gpt2`, `hf_llama`, `hf_bert`, `hf_t5`) | Yes | Full | All |
| Quantized (`hf_bnb`, `hf_awq`, `hf_gptq`) | Best-effort | Full when modules exposed | Linux |
| ONNX (`hf_onnx`) | No | Eval-only | All |

## Reference

### Supported adapters

| Adapter | Models / Purpose | Requires | Platform support | Notes |
| --- | --- | --- | --- | --- |
| `hf_gpt2` | GPT-2/OPT/GPT‑Neo-X style causal LMs | `invarlock[adapters]` | All platforms with torch | Default causal LM adapter. |
| `hf_llama` | LLaMA/Mistral/Qwen/Yi causal LMs | `invarlock[adapters]` | All platforms with torch | Handles RMSNorm + RoPE/GQA. |
| `hf_bert` | BERT/RoBERTa/DeBERTa MLMs | `invarlock[adapters]` | All platforms with torch | Loads `AutoModelForMaskedLM` when possible. |
| `hf_t5` | T5/encoder‑decoder models | `invarlock[adapters]` | All platforms with torch | For seq2seq evaluation. |
| `hf_onnx` | Optimum/ONNXRuntime causal LMs | `invarlock[onnx]` | All platforms | Inference-only; snapshot/restore not supported. |
| `hf_causal_auto` | Auto-select causal adapter | `invarlock[adapters]` | All platforms with torch | Resolves to `hf_gpt2`/`hf_llama`. |
| `hf_mlm_auto` | Auto-select MLM adapter | `invarlock[adapters]` | All platforms with torch | Resolves to `hf_bert`. |
| `hf_bnb` | Bitsandbytes quantized LMs | `invarlock[gpu]` | Linux only | Uses `device_map="auto"`; no `.to()`. |
| `hf_awq` | AWQ quantized LMs | `invarlock[awq]` | Linux only | Requires `autoawq`/`triton`. |
| `hf_gptq` | GPTQ quantized LMs | `invarlock[gptq]` | Linux only | Requires `auto-gptq`/`triton`. |

### Adapter capabilities

| Adapter class | Snapshot/restore | Guard compatibility | Notes |
| --- | --- | --- | --- |
| PyTorch HF adapters (`hf_gpt2`, `hf_llama`, `hf_bert`, `hf_t5`) | Yes | Full (module access) | Uses `HFAdapterMixin` snapshots. |
| Quantized HF adapters (`hf_bnb`, `hf_awq`, `hf_gptq`) | Yes (best-effort) | Full when modules are exposed | Avoid explicit `.to()` calls. |
| ONNX adapter (`hf_onnx`) | No | Eval-only | Use `edit: noop` and expect guard limitations. |

### Adapter selection (`adapter: auto`)

Automatic resolution uses the model’s `config.model_type` and structure checks.
The CLI’s `adapter: auto` shortcut resolves to `hf_causal_auto` or `hf_mlm_auto`
before plugin discovery.

- LLaMA/Mistral/Qwen/Yi → `hf_llama`
- BERT/RoBERTa/DeBERTa/ALBERT → `hf_bert`
- GPT‑2/OPT/GPT‑Neo-X → `hf_gpt2`

```yaml
model:
  id: meta-llama/Llama-2-7b-hf
  adapter: auto
  device: auto
```

### Configuration examples

```yaml
# Standard causal LM run
model:
  id: gpt2
  adapter: hf_gpt2
  device: auto
```

```yaml
# Bitsandbytes quantized load (Linux + gpu extra)
model:
  id: mistralai/Mistral-7B-v0.1
  adapter: hf_bnb
  load_in_8bit: true
```

```yaml
# ONNX Runtime inference-only adapter (use with edit: noop)
model:
  id: /path/to/onnx-model
  adapter: hf_onnx
edit:
  name: noop
```

### Adapter load arguments

Adapter loaders pass through standard Hugging Face `from_pretrained` arguments:

| Key | Common use | Applies to |
| --- | --- | --- |
| `torch_dtype` | Force `float16`/`bfloat16` | HF adapters |
| `device_map` | Sharding/placement | HF adapters |
| `trust_remote_code` | Enable custom model code | HF adapters |
| `revision` | Pin model revision | HF adapters |
| `cache_dir` | Cache location | HF adapters |

### Adapter describe fields

`adapter.describe(model)` returns a dictionary containing:

- `n_layer`, `heads_per_layer`, `mlp_dims`, `tying` (required for guard gates)
- `model_type`, `model_class`, and adapter-specific metadata

### Snapshot strategy

```python
snapshot = adapter.snapshot(model)
try:
    # mutate model
    ...
    adapter.restore(model, snapshot)
finally:
    pass
```

For large models, use chunked snapshots:

```python
snap_dir = adapter.snapshot_chunked(model)
try:
    adapter.restore_chunked(model, snap_dir)
finally:
    import shutil
    shutil.rmtree(snap_dir, ignore_errors=True)
```

## Troubleshooting

- **Adapter missing from `invarlock plugins adapters`**: install the required extra
  (`invarlock[adapters]`, `invarlock[gpu]`, `invarlock[gptq]`, `invarlock[awq]`).
- **Linux-only adapters not available**: `hf_bnb`, `hf_awq`, and `hf_gptq` are
  only published for Linux in `pyproject.toml`.
- **Quantized model `.to()` errors**: avoid explicit `.to()`; load with the adapter
  and let it manage device placement.
- **ONNX adapter guard failures**: `hf_onnx` is inference-only; use `edit: noop`
  and avoid guards that require PyTorch module access.

## Observability

- `invarlock plugins adapters --json` reports readiness and missing extras.
- `report.context["plugins"]` and certificate `plugins.adapters` record adapter
  discovery for audit trails.

## Related Documentation

- [CLI Reference](cli.md)
- [Configuration Schema](config-schema.md)
- [Dataset Providers](datasets.md)
- [Environment Variables](env-vars.md)
- [Certificates](certificates.md) — Schema, telemetry, and HTML export
