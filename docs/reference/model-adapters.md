# Model Adapters

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Load models, describe structure, and snapshot/restore state for edits and guards. |
| **Audience** | CLI users choosing `model.adapter` and Python callers instantiating adapters. |
| **Supported surface** | Core HF text and image-text adapters, auto-match adapters, platform-dependent BNB, and GPTQModel-backed AWQ/GPTQ quantized adapters. |
| **Requires** | `invarlock[adapters]` or `invarlock[hf]` for core HF adapters; `invarlock[gpu]`, `invarlock[awq]`, `invarlock[gptq]` for quantized adapters. |
| **Network** | Offline by default; use `evaluate --allow-network` when a run needs model downloads. |
| **Inputs** | `model.id` (HF repo or local path), adapter name, device. |
| **Outputs / Artifacts** | Loaded model object; optional snapshots; exported model directories when enabled. |
| **Source of truth** | `src/invarlock/adapters/*`, `src/invarlock/plugins/hf_*_adapter.py`. |

## Quick Start

```bash
# Install core HF adapters + evaluation stack
pip install "invarlock[hf]"

# Inspect adapter availability
invarlock advanced plugins adapters

# Compare & evaluate with adapter auto-selection
invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject gpt2 \
  --baseline-adapter auto --subject-adapter auto
```

The CLI example above uses the runtime container by default. Add
`--execution-mode host` only for host-side compare/evaluate workflows that
intentionally bypass that boundary.

```python
from invarlock.adapters.auto import HF_Auto_Adapter

adapter = HF_Auto_Adapter()
model = adapter.load_model("gpt2", device="auto")
print(adapter.describe(model)["model_type"])
```

> Adapter availability is broader than the published assurance basis. GPT-2 and
> BERT back the published calibrated basis; repo-included pilot configs
> for Mistral 7B, Ministral 3 text-only, Qwen2 7B, Qwen2.5 7B, Qwen2.5 14B, and additional experimental families are for
> experimentation until supporting artifacts are attached. See the Model Family
> Catalog for the authoritative family-by-family inventory.

## Concepts

- **Adapters hide model-specific logic**: they handle loading, structure description,
  and snapshot/restore so edits/guards stay model-agnostic.
- **Auto selection**: use `adapter: auto` in a single-run config, or use
  `--baseline-adapter auto --subject-adapter auto` for paired evaluation to
  choose concrete role adapters (`hf_causal`, `hf_mlm`,
  `hf_seq2seq`) plus quant adapters when detected. Local paths can use
  `config.json`; remote IDs fall back to name heuristics and default to
  `hf_causal` when unsure. Image-text models use the explicit
  `hf_multimodal` adapter rather than adapter auto.
- **Quantized adapters** (`hf_bnb`, `hf_awq`, `hf_gptq`) handle their own device
  placement; avoid calling `.to(...)` on the loaded model.
- **Containerized quant evidence** requires a runtime image with the optional
  quant backends installed. For remote CUDA evidence-pack setup, set
  `PACK_RUNTIME_IMAGE_FLAVOR=quant` to select/build
  `invarlock-runtime:cuda-quant` instead of the default CUDA runtime image.
  This opt-in image retains the compiler toolchain because GPTQModel-backed
  GPTQ/AWQ kernels may JIT-compile at model-load time.
  Strict release-review evidence still needs normal runtime provenance: use a
  local InvarLock runtime image tag or set `INVARLOCK_RUNTIME_IMAGE_DIGEST` for
  custom image references.
- **Snapshot strategy**: HF adapters expose `snapshot`/`restore` and
  `snapshot_chunked`/`restore_chunked` (large-model friendly). The CLI selects the
  strategy automatically via `context.snapshot.*`.

### Auto adapter mapping

| `model_type` family | Adapter |
| --- | --- |
| llama / mistral / mistral3 / mixtral / qwen / gemma / OLMo / yi | `hf_causal` |
| gpt2 / gpt_oss / opt / neo-x / phi | `hf_causal` |
| bert / roberta | `hf_mlm` |
| t5 / bart | `hf_seq2seq` |

Auto inspects `config.model_type`; remote models may need network for config.

Capability matrix (at a glance)

| Adapter family | Snapshot/restore | Guard compatibility | Platform |
| --- | --- | --- | --- |
| HF text (`hf_causal`, `hf_mlm`, `hf_seq2seq`) | Yes | Full | All |
| HF image-text (`hf_multimodal`) | Yes | Full when decoder layers are exposed | All |
| Quantized (`hf_bnb`) | Best-effort | Full when modules exposed | Platform-dependent |
| Quantized (`hf_awq`, `hf_gptq`) | Best-effort | Full when modules exposed | GPTQModel-supported platforms |

Machine-readable adapter capability metadata is published at
`contracts/adapter_capabilities.json` and surfaced through
`invarlock advanced plugins adapters --json`.

## Reference

### Supported adapters

| Adapter | Models / Purpose | Requires | Platform support | Notes |
| --- | --- | --- | --- | --- |
| `hf_causal` | Decoder-only causal LMs (dense + MoE + GPT2-like) | `invarlock[adapters]` | All platforms with torch | Default causal LM adapter. |
| `hf_mlm` | BERT/RoBERTa/DeBERTa MLMs | `invarlock[adapters]` | All platforms with torch | Loads `AutoModelForMaskedLM` when possible. |
| `hf_multimodal` | Image-text generation models exposed through HF `AutoModelForImageTextToText` | `invarlock[adapters]` | All platforms with torch | Single-image `vision_text` evaluation with explicit adapter selection. |
| `hf_seq2seq` | T5/encoder‑decoder models | `invarlock[adapters]` | All platforms with torch | For seq2seq evaluation. |
| `hf_auto` | Auto-select HF adapter | `invarlock[adapters]` | All platforms with torch | Delegates to a role adapter; prefers quant adapters when detected. |
| `hf_bnb` | Bitsandbytes quantized LMs | `invarlock[gpu]` | Platform-dependent | Uses `device_map="auto"`; no `.to()`. Latest bitsandbytes wheels can work outside Linux/CUDA when the runtime imports cleanly. |
| `hf_awq` | AWQ quantized LMs | `invarlock[awq]` | GPTQModel-supported platforms | Uses the Transformers AWQ loader backed by GPTQModel; GPU recommended for quantized inference. |
| `hf_gptq` | GPTQ quantized LMs | `invarlock[gptq]` | GPTQModel-supported platforms | Uses GPTQModel for GPTQ subject loading; GPU recommended for quantized inference. |

### Adapter capabilities

| Adapter class | Snapshot/restore | Guard compatibility | Notes |
| --- | --- | --- | --- |
| PyTorch HF adapters (`hf_causal`, `hf_mlm`, `hf_multimodal`, `hf_seq2seq`) | Yes | Full (module access) / multimodal full when decoder layers are exposed | Uses `HFAdapterMixin` snapshots. |
| Quantized HF adapters (`hf_bnb`, `hf_awq`, `hf_gptq`) | Yes (best-effort) | Full when modules are exposed | Avoid explicit `.to()` calls. |

### Adapter selection (`adapter: auto`)

Automatic resolution uses local `config.json` (if `model.id` is a directory) and
simple heuristics to choose a concrete built-in adapter name.

- Decoder-only causal → `hf_causal`
- BERT/RoBERTa/DeBERTa/ALBERT → `hf_mlm`
- T5/BART → `hf_seq2seq`

```yaml
model:
  id: mistralai/Mistral-7B-v0.1
  adapter: auto
  device: auto
```

### Configuration examples

```yaml
# Standard causal LM run
model:
  id: gpt2
  adapter: hf_causal
  device: auto
```

```yaml
# Bitsandbytes quantized load (Linux + gpu extra)
model:
  id: mistralai/Mistral-7B-v0.1
  adapter: hf_bnb
  quantization_config:
    quant_method: bitsandbytes
    bits: 8
```

### Adapter load arguments

Adapter loaders pass through standard Hugging Face `from_pretrained` arguments:

| Key | Common use | Applies to |
| --- | --- | --- |
| `dtype` | Force `float16`/`bfloat16` | HF adapters |
| `device_map` | Sharding/placement | HF adapters |
| `trust_remote_code` | Enable custom model code only with explicit `--allow-remote-code` / `INVARLOCK_ALLOW_REMOTE_CODE=1` | HF adapters |
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

- **Adapter missing from `invarlock advanced plugins adapters`**: install the required extra
  (`invarlock[adapters]`, `invarlock[gpu]`, `invarlock[gptq]`, `invarlock[awq]`).
- **GPTQModel-backed adapters unavailable**: `hf_awq` and `hf_gptq` use
  GPTQModel-backed loading; verify the selected GPTQModel wheel supports your
  Python, PyTorch, and accelerator stack.
- **Container report fails with missing quant extra**: build or select the quant
  CUDA runtime image (`make runtime-image-cuda-quant`, or
  `PACK_RUNTIME_IMAGE_FLAVOR=quant` for the remote setup helper). The default
  CUDA runtime image is intentionally limited to the core evaluation stack.
- **Bitsandbytes not detected**: `hf_bnb` is platform-dependent. If the backend
  imports cleanly, `invarlock advanced plugins adapters` will report it as ready even on
  non-CUDA hosts.
- **Quantized model `.to()` errors**: avoid explicit `.to()`; load with the adapter
  and let it manage device placement.

## Observability

- `invarlock advanced plugins adapters --json` reports readiness and missing extras.
- `report.context["plugins"]` and report `plugins.adapters` record adapter
  discovery for audit trails.

## Related Documentation

- [CLI Reference](cli.md)
- [Configuration Schema](config-schema.md)
- [Dataset Providers](datasets.md)
- [Environment Variables](env-vars.md)
- [reports](reports.md) — Schema, telemetry, and HTML export
