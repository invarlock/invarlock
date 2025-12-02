# Model Adapters Reference

Model adapters enable InvarLock to work with different neural network architectures
and frameworks. This guide covers all available adapters and how to use them
effectively.

## Overview

Model adapters provide a standardized interface for InvarLock to interact with
different model architectures. They handle:

- Model loading and validation
- Architecture-specific optimizations
- Device management and memory efficiency
- State serialization and restoration
- Weight tying preservation

Adapters live under the `invarlock.adapters` namespace and are enabled via
optional extras (for example, `pip install "invarlock[hf]"` or
`pip install "invarlock[adapters]"`). The core install
(`pip install invarlock`) remains torch‑free.

## Available Adapters

### HuggingFace GPT-2 Adapter (`hf_gpt2`)

**Purpose**: Support for HuggingFace GPT-2 and GPT-2 variants

**Supported Models**:



- `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- DistilGPT-2, CodeParrot, and other GPT-2 variants

**Configuration**:

```yaml
model:
  id: "gpt2"
  adapter: "hf_gpt2"
  device: "auto"
```

**Features**:

- Automatic device placement (CPU, CUDA, MPS)
- Weight tying preservation between embeddings and LM head
- Efficient state management for large models
- Support for custom tokenizers

### HuggingFace LLaMA Adapter (`hf_llama`)

**Purpose**: Support for Meta's LLaMA and LLaMA-2 models

**Supported Models**:

- LLaMA (7B, 13B, 30B, 65B)
- LLaMA-2 (7B, 13B, 70B)
- Code Llama variants
- Alpaca and other LLaMA fine-tunes

**Configuration**:

```yaml
model:
  id: "meta-llama/Llama-2-7b-hf"
  adapter: "hf_llama"
  device: "auto"
  load_in_8bit: false  # Optional: Use 8-bit loading
```

**Key Features**:



- **Group Query Attention (GQA)**: Optimized handling of multi-head attention
- **RMSNorm Support**: Proper handling of Root Mean Square normalization
- **RoPE Integration**: Rotary Position Embedding support
- **SwiGLU Activation**: Swish-Gated Linear Unit handling
- **Device-Aware Loading**: Automatic memory optimization

**Architecture Details**:

```python
# LLaMA-specific features detected by adapter
{
    "attention_type": "gqa",           # Group Query Attention
    "normalization_type": "rmsnorm",   # RMS normalization
    "activation": "swiglu",            # SwiGLU activation
    "positional_encoding": "rope",     # Rotary Position Embedding
    "gqa_config": {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,      # GQA configuration
        "head_dim": 128
    }
}
```

### HuggingFace BERT Adapter (`hf_bert`)

**Purpose**: Support for BERT-family encoder models

**Supported Models**:

- BERT (base, large)
- RoBERTa (base, large)
- DistilBERT
- ALBERT, ELECTRA
- Domain-specific BERT variants (BioBERT, SciBERT, etc.)

**Configuration**:

```yaml
model:
  id: "bert-base-uncased"
  adapter: "hf_bert"
  device: "auto"
  task_type: "classification"  # Optional: specify task type
```

**Key Features**:

- **Bidirectional Attention**: Proper handling of BERT's attention mechanism
- **Token Type Embeddings**: Support for segment embeddings
- **Classification Heads**: Integration with downstream task heads
- **Pooling Layer Support**: Proper handling of CLS token pooling
- **Multi-Task Support**: Sequence classification, token classification, QA

**Architecture Details**:

```python
# BERT-specific features detected by adapter
{
    "attention_type": "bidirectional",
    "layer_norm_type": "standard",
    "activation": "gelu",
    "positional_encoding": "learned",
    "use_token_type_embeddings": True,
    "max_position_embeddings": 512,
    "embeddings_info": {
        "vocab_size": 30522,
        "hidden_size": 768,
        "type_vocab_size": 2
    }
}
```

## Adapter Selection

InvarLock automatically selects the appropriate adapter based on model
characteristics via the auto adapters and the CLI `adapter: auto` setting:

### Automatic Selection (CLI)

```yaml
# In a preset or context override
model:
  id: meta-llama/Llama-2-7b-hf
  adapter: auto  # resolves to hf_llama under the hood
```

### Automatic Selection (Python)

```python
from invarlock.adapters import HF_Causal_Auto_Adapter

adapter = HF_Causal_Auto_Adapter()
model = adapter.load_model("meta-llama/Llama-2-7b-hf", device="auto")
# Under the hood, this delegates to HF_LLaMA_Adapter
```

### Manual Selection (YAML)

```yaml
# Explicitly specify adapter in configuration
model:
  id: "bert-base-uncased"
  adapter: "hf_bert"  # Force specific adapter
```

### Selection Criteria

1. **Model Type Detection**: Checks `config.model_type`
2. **Structural Analysis**: Examines layer patterns and components
3. **Class Name Matching**: Matches HuggingFace class names
4. **Architecture Validation**: Verifies expected components exist

## API Reference

### Base Adapter Interface (core)

Core adapters expose the following abstract interface (see `invarlock.core.api.ModelAdapter`):

```python
class ModelAdapter:
    def can_handle(self, model: Any) -> bool: ...
    def describe(self, model: Any) -> dict[str, Any]: ...
    def snapshot(self, model: Any) -> bytes: ...
    def restore(self, model: Any, blob: bytes) -> None: ...
```

Many concrete adapters also provide convenience helpers:

```python
# Optional conveniences implemented by HF adapters via HFAdapterMixin
adapter.load_model(model_id: str, device: str = "auto") -> Any
adapter.snapshot_chunked(model: nn.Module, *, prefix: str = "invarlock-snap-") -> str
adapter.restore_chunked(model: nn.Module, snapshot_path: str) -> None
```

### Model Description Format

All adapters return standardized model descriptions:

```python
{
    # Required fields for InvarLock operation
    "n_layer": 12,                           # Number of transformer layers
    "heads_per_layer": [12, 12, 12, ...],    # Attention heads per layer
    "mlp_dims": [3072, 3072, 3072, ...],     # MLP hidden dimensions
    "tying": {"lm_head.weight": "transformer.wte.weight"},  # Weight tying map

    # Model metadata
    "model_type": "gpt2",                    # Architecture type
    "model_class": "GPT2LMHeadModel",        # HuggingFace class name
    "total_params": 124439808,               # Total parameters
    "device": "cuda:0",                      # Current device

    # Architecture-specific details
    "architecture": {
        "attention_type": "causal",          # Attention mechanism
        "normalization_type": "layernorm",   # Normalization type
        "activation": "gelu",                # Activation function
        "positional_encoding": "learned"     # Position encoding type
    }
}
```

## Usage Examples

### Loading Different Model Types

```python
from invarlock.adapters import HF_GPT2_Adapter, HF_LLaMA_Adapter, HF_BERT_Adapter

# GPT-2 model
gpt2_adapter = HF_GPT2_Adapter()
gpt2_model = gpt2_adapter.load_model("gpt2", device="cuda")

# LLaMA model
llama_adapter = HF_LLaMA_Adapter()
llama_model = llama_adapter.load_model("meta-llama/Llama-2-7b-hf", device="cuda")

# BERT model
bert_adapter = HF_BERT_Adapter()
bert_model = bert_adapter.load_model("bert-base-uncased", device="cuda")
```

### Model Analysis

```python
# Get detailed model description
description = adapter.describe(model)

print(f"Model has {description['n_layer']} layers")
print(f"Total parameters: {description['total_params']:,}")
print(f"Architecture: {description['architecture']['attention_type']}")

# Check for specific features
if description['architecture']['attention_type'] == 'gqa':
    print("Model uses Group Query Attention")
    gqa_config = description['gqa_config']
    print(f"GQA ratio: {gqa_config['gqa_ratio']}")
```

### State Management

```python
# Create model snapshot for rollback
snapshot = adapter.snapshot(model)

# Modify model (e.g., apply edits)
# ... model modifications ...

# Restore original state if needed
adapter.restore(model, snapshot)

# For large models, prefer chunked snapshot to reduce peak memory use:
snap_dir = adapter.snapshot_chunked(model)
try:
    # ... modify model ...
    adapter.restore_chunked(model, snap_dir)
finally:
    import shutil
    shutil.rmtree(snap_dir, ignore_errors=True)
```

> The CLI automatically reuses a single loaded model across retries and chooses
> between bytes or chunked snapshots based on the estimated snapshot size vs
> available RAM and disk. You can override this behavior via
> `context.snapshot.*` config or `INVARLOCK_SNAPSHOT_MODE`.

### Device Management

```python
# Automatic device selection
model = adapter.load_model("gpt2", device="auto")

# Manual device specification
model = adapter.load_model("gpt2", device="cuda:1")

# Check current device
description = adapter.describe(model)
print(f"Model on device: {description['device']}")
```

## Configuration Examples

### Conservative Configuration

```yaml
model:
  id: "meta-llama/Llama-2-7b-hf"
  adapter: "hf_llama"
  device: "auto"
  load_in_8bit: true          # Memory optimization
  trust_remote_code: false    # Security setting

guards:
  spectral:
    sigma_quantile: 0.90         # Conservative spectral bound
  rmt:
    margin: 1.3               # Tight outlier detection
```

### Performance Configuration

```yaml
model:
  id: "gpt2-xl"
  adapter: "hf_gpt2"
  device: "cuda"
  torch_dtype: "float16"      # Half precision
  low_cpu_mem_usage: true     # Memory optimization

  # Compare & Certify (BYOE) or use built-in quant_rtn demo in a separate preset
```

### Multi-Model Pipeline

```yaml
models:
  base_model:
    id: "gpt2"
    adapter: "hf_gpt2"

  comparison_model:
    id: "bert-base-uncased"
    adapter: "hf_bert"

evaluation:
  compare_models: true
```

## Advanced Features

### Custom Adapter Development

Create custom adapters by extending the base class:

```python
class CustomAdapter(BaseAdapter):
    name = "custom_adapter"

    def can_handle(self, model: nn.Module) -> bool:
        # Custom model detection logic
        return hasattr(model, 'custom_attribute')

    def describe(self, model: nn.Module) -> dict:
        # Custom model analysis
        return {
            "n_layer": self._count_layers(model),
            "heads_per_layer": self._get_heads(model),
            # ... other required fields
        }
```

### Weight Tying Analysis

```python
# Analyze weight tying relationships
tying_info = adapter._extract_weight_tying_info(model)

for tied_param, source_param in tying_info.items():
    print(f"{tied_param} is tied to {source_param}")

# Verify weight tying preservation
original_tying = adapter._extract_weight_tying_info(model)
# ... apply edits ...
final_tying = adapter._extract_weight_tying_info(model)

assert original_tying == final_tying, "Weight tying was broken!"
```

### Memory Optimization

```python
# Load large models efficiently
adapter = HF_LLaMA_Adapter()

# Option 1: 8-bit loading
model = adapter.load_model(
    "meta-llama/Llama-2-70b-hf",
    device="auto",
    load_in_8bit=True
)

# Option 2: Device mapping for multi-GPU
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 1,
    # ... distribute layers across GPUs
}
```

## Troubleshooting

### Common Issues

**1. Import Errors**

```python
# Check if transformers is available
try:
    from invarlock.adapters.hf_llama import HF_LLaMA_Adapter
except ImportError:
    print("Install transformers: pip install transformers")
```

**2. Model Not Recognized**

```python
# Debug adapter selection
for adapter_class in [HF_GPT2_Adapter, HF_LLaMA_Adapter, HF_BERT_Adapter]:
    adapter = adapter_class()
    if adapter.can_handle(model):
        print(f"Model can be handled by {adapter.name}")
```

**3. Device Mismatch**

```python
# Ensure model and data are on same device
model_device = next(model.parameters()).device
print(f"Model device: {model_device}")

# Move data to model device
data = data.to(model_device)
```

**4. Memory Issues**

```python
# Monitor memory usage
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

### Performance Tips

1. **Use appropriate precision**: `float16` for inference, `float32` for training
2. **Enable optimizations**: Set `low_cpu_mem_usage=True` for large models
3. **Batch processing**: Process multiple samples together when possible
4. **Device placement**: Use fastest available device (CUDA > MPS > CPU)

## Integration with InvarLock Pipeline

Adapters integrate seamlessly with the InvarLock pipeline:

```yaml
# Complete pipeline configuration
model:
  id: "meta-llama/Llama-2-7b-hf"
  adapter: "hf_llama"

dataset:
  provider: "wikitext2"
  seq_len: 512

  # edit:
  #   name: "quant_rtn"

guards:
  order: ["invariants", "spectral", "rmt"]
  spectral:
    sigma_quantile: 0.95
    family_caps:
      ffn: 2.5
      attn: 2.8
      embed: 3.0
      other: 3.0
  rmt:
    margin: 1.5

evaluation:
  metrics: ["perplexity", "accuracy"]
```

The adapter automatically handles model-specific optimizations and ensures
compatibility with InvarLock's edit and guard systems.

### Adapter Auto-Detection (`adapter: auto`)

Set `model.adapter: auto` to let InvarLock resolve the correct built-in adapter from
the model's `config.json` (no network). Typical resolution:

- LLaMA/Mistral/Qwen/Yi → `hf_llama`
- BERT/RoBERTa/DeBERTa/ALBERT → `hf_bert`
- GPT-2/OPT/GPT‑NeoX variants → `hf_gpt2`

Example:

```yaml
model:
  id: "meta-llama/Llama-2-7b-hf"
  adapter: auto
  device: auto
```
