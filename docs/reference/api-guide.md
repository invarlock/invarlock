# InvarLock API Reference Guide

This guide provides documentation for using InvarLock programmatically. The
programmatic surface is narrower than the CLI and still evolving. Prefer CLI
usage for production workflows and use the types in `invarlock.core.api` together
with built‑in adapters and edits for light programmatic integration.

> Notes
>
> - Import from the unified namespace `invarlock.*` (e.g., `invarlock.guards`,
>   `invarlock.adapters`, `invarlock.edits`).
> - The runner entry point is `invarlock.core.runner.CoreRunner` (method
>   `execute(...)`). Types such as `RunConfig` and `ModelAdapter` live in
>   `invarlock.core.api`.
> - Examples below are illustrative and may omit error handling; use the CLI
>   for end‑to‑end flows.

See also

- Compare & Certify (BYOE): ../user-guide/compare-and-certify.md
- Primary Metric Smoke (tiny examples): ../user-guide/primary-metric-smoke.md
- Programmatic Quickstart (minimal script): ./programmatic-quickstart.md

## Certification API (stable public surface)

Use the assurance facade for certificate operations. This surface remains
stable even if internals move.

```python
from invarlock.assurance import make_certificate, validate_certificate, render_certificate_markdown

# Given a guarded run report and a baseline report (or metrics), build a certificate
certificate = make_certificate(report, baseline)
assert validate_certificate(certificate)

md = render_certificate_markdown(certificate)
print(md.splitlines()[0])  # "# InvarLock Safety Certificate"
```

Note: Internals live under `invarlock.reporting.certificate`, but import from
`invarlock.assurance` in user code to avoid coupling to private layouts.

## Public vs Internal API

- Public (stable import path): `invarlock.assurance`
  - Exposes `CERTIFICATE_SCHEMA_VERSION`, `make_certificate`,
    `validate_certificate`, `render_certificate_markdown`.
  - Preferred stable surface for certificate operations; import from this
    facade in user code.
- Public (core types used in examples): `invarlock.core.api` and top‑level
  convenience namespaces (`invarlock.adapters`, `invarlock.edits`, `invarlock.guards`).
  These are intended for light programmatic integration shown in this guide.
- Internal (subject to change without notice): `invarlock.reporting.*`,
  `invarlock.eval.*`, CLI wiring under `invarlock.cli.*`, and guard policy internals.
  If you currently import from `invarlock.reporting.certificate`, migrate to
  `invarlock.assurance` to avoid breakage.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Model Loading](#model-loading)
4. [Edit Operations](#edit-operations)
5. [Guard System](#guard-system)
6. [Pipeline Configuration](#pipeline-configuration)
7. [Custom Extensions](#custom-extensions)
8. [Error Handling](#error-handling)
9. [Best Practices](#best-practices)

## Quick Start

### Basic Usage

```python
from invarlock.core.runner import CoreRunner
from invarlock.core.api import RunConfig
from invarlock.adapters import HF_Causal_Auto_Adapter
from invarlock.edits import RTNQuantEdit
from invarlock.guards import InvariantsGuard

# Minimal programmatic example (demo edit)
adapter = HF_Causal_Auto_Adapter()
model = adapter.load_model('gpt2', device='auto')
edit = RTNQuantEdit(bitwidth=8, per_channel=True, group_size=128, clamp_ratio=0.005)
guards = [InvariantsGuard()]
cfg = RunConfig(device='auto')

runner = CoreRunner()
report = runner.execute(model, adapter, edit, guards, cfg)
print(report.meta.get('status', 'done'))
```

### Advanced Pipeline

```python
from invarlock.core.runner import CoreRunner
from invarlock.core.api import RunConfig
from invarlock.guards import SpectralGuard, RMTGuard, InvariantsGuard
from invarlock.adapters import HF_GPT2_Adapter
from invarlock.edits import RTNQuantEdit

# Create custom pipeline
adapter = HF_GPT2_Adapter()
edit = RTNQuantEdit(bitwidth=8, per_channel=True, group_size=128, clamp_ratio=0.005)
guards = [
    InvariantsGuard(strict_mode=True),
    SpectralGuard(sigma_quantile=0.95, deadband=0.10),
    RMTGuard(margin=1.5, correct=False)
]

runner = CoreRunner()
model = adapter.load_model('gpt2', device='auto')
cfg = RunConfig(device='auto')
report = runner.execute(model, adapter, edit, guards, cfg)
```

## Core Components

### CoreRunner

The main orchestrator for InvarLock operations.

```python
class CoreRunner:
    def __init__(self,
                 adapter=None,
                 edit=None,
                 guards=None,
                 evaluator=None):
        """Initialize InvarLock runner.

        Args:
            adapter: Model adapter instance
            edit: Edit operation instance
            guards: List of guard instances
            evaluator: Evaluation instance
        """

    def run(self, config: dict) -> dict:
        """Run complete InvarLock pipeline.

        Args:
            config: Configuration dictionary

        Returns:
            Result dictionary with metrics and outputs
        """

    def run_with_model(self, model, config: dict) -> dict:
        """Run pipeline with pre-loaded model.

        Args:
            model: Pre-loaded model instance
            config: Configuration dictionary

        Returns:
            Result dictionary
        """
```

### RunConfig

Configuration management and validation.

```python
from invarlock.core.api import RunConfig

# Create configuration from dictionary
config_dict = {
    'model': {'id': 'gpt2', 'adapter': 'hf_gpt2'},
    # 'edit': {'name': 'quant_rtn', 'parameters': {'bitwidth': 8}}
}

config = RunConfig()  # use fields directly; see invarlock.core.api.RunConfig
```

## Model Loading

### Automatic Model Loading

```python
from invarlock.adapters import HF_Causal_Auto_Adapter

adapter = HF_Causal_Auto_Adapter()
model = adapter.load_model('gpt2', device='auto')
print(f"Auto adapter resolved; device: {next(model.parameters()).device}")
```

### Manual Adapter Selection

```python
from invarlock.adapters import HF_GPT2_Adapter, HF_LLaMA_Adapter, HF_BERT_Adapter

# Load specific model types
gpt2_adapter = HF_GPT2_Adapter()
gpt2_model = gpt2_adapter.load_model('gpt2', device='cuda')

llama_adapter = HF_LLaMA_Adapter()
llama_model = llama_adapter.load_model('meta-llama/Llama-2-7b-hf')

bert_adapter = HF_BERT_Adapter()
bert_model = bert_adapter.load_model('bert-base-uncased')

# Verify compatibility
assert gpt2_adapter.can_handle(gpt2_model)
assert llama_adapter.can_handle(llama_model)
assert bert_adapter.can_handle(bert_model)
```

### Model Analysis

```python
# Get detailed model description
description = adapter.describe(model)

print(f"Architecture: {description['model_type']}")
print(f"Layers: {description['n_layer']}")
print(f"Parameters: {description['total_params']:,}")
print(f"Attention heads per layer: {description['heads_per_layer']}")
print(f"MLP dimensions: {description['mlp_dims']}")

# Check for specific features
arch = description['architecture']
if arch['attention_type'] == 'gqa':
    print("Model uses Group Query Attention")
    gqa = description['gqa_config']
    print(f"GQA ratio: {gqa['gqa_ratio']}")
```

### Model State Management

```python
# Create snapshot for rollback (small/medium models)
snapshot = adapter.snapshot(model)

# Apply modifications
# ... edit operations ...

# Restore if needed
adapter.restore(model, snapshot)

# For large models, use chunked snapshots to minimize peak memory:
snap_dir = adapter.snapshot_chunked(model)
try:
    # ... mutate model and test ...
    adapter.restore_chunked(model, snap_dir)
finally:
    import shutil
    shutil.rmtree(snap_dir, ignore_errors=True)

# Verify restoration
restored_description = adapter.describe(model)
assert restored_description == original_description
```

## Edit Operations

### Quant RTN (built-in)

```python
from invarlock.edits import RTNQuantEdit

edit = RTNQuantEdit(
    bitwidth=8,
    per_channel=True,
    group_size=128,
    clamp_ratio=0.005,
    scope="attn",
)

preview = edit.preview(model, adapter, calibration_data=None)
result = edit.apply(model, adapter, preview.get('plan', {}))
```

### Custom Guard Development

```python
from invarlock.core.api import Guard

class CustomGuard(Guard):
    name = "custom_guard"

    def prepare(self, model, adapter, calib, policy):
        return {"ready": True}

    def check(self, model, adapter, report):
        return {"passed": True, "metrics": {}}
```

## Guard System

### Guard Configuration

```python
from invarlock.guards import (
    SpectralGuard, RMTGuard, InvariantsGuard, VarianceGuard
)
from invarlock.guards.policies import get_variance_policy

# Create guards with different policies
guards = [
    InvariantsGuard(strict_mode=True, on_fail="abort"),

    SpectralGuard(
        sigma_quantile=0.95,        # Target percentile for spectral norm
        deadband=0.10,           # Tolerance
        scope="all",             # Target all layers
        correction_enabled=False # Monitor-only in balanced tier
    ),

    RMTGuard(
        margin=1.5,              # Safety margin
        deadband=0.10,           # Tolerance
        correct=False,           # Monitor-only in balanced tier
        q="auto"                 # Auto-detect aspect ratio
    ),

    VarianceGuard(
        get_variance_policy("balanced")
    )
]
```

### Guard Chain Execution

```python
from invarlock.core.api import GuardChain

# Create guard chain
guard_chain = GuardChain(guards)

# Prepare all guards
preparation_results = guard_chain.prepare_all(
    model, adapter, calibration_data, policy={}
)

# Execute before edit hooks
guard_chain.before_edit_all(model)

# Apply edit operations
# ... edit operations ...

# Execute after edit hooks
guard_chain.after_edit_all(model)

# Finalize and get results
finalization_results = guard_chain.finalize_all(model)

# Check if all guards passed
all_passed = guard_chain.all_passed(finalization_results)
if not all_passed:
    print("Some guards failed!")
    for guard_name, result in finalization_results.items():
        if not result.passed:
            print(f"{guard_name}: {result.violations}")
```

### Individual Guard Usage

```python
# Spectral guard detailed usage
spectral_guard = SpectralGuard(sigma_quantile=0.95, deadband=0.10)

# Capture baseline spectral properties
from invarlock.guards.spectral import capture_baseline_sigmas
baseline_sigmas = capture_baseline_sigmas(model)

# Validate after modifications
result = spectral_guard.validate(
    model, adapter,
    {"baseline_metrics": baseline_sigmas}
)

if not result["passed"]:
    print(f"Spectral violation: {result['message']}")
    if result["action"] == "abort":
        # Handle failure
        pass

# RMT guard detailed usage
rmt_guard = RMTGuard(margin=1.5, correct=False)

# Prepare with baseline
preparation = rmt_guard.prepare(model, adapter, calibration_data, {})
if preparation["ready"]:
    # Apply edit
    # ...

    # Check for outliers (balanced tier monitors only)
    rmt_guard.after_edit(model)

    # Get final results
    outcome = rmt_guard.finalize(model)
    print(f"RMT outliers detected: {outcome.get('has_outliers', False)}")

# Enable conservative correction if desired
conservative_rmt = RMTGuard(margin=1.3, deadband=0.05, correct=True)
```

## Pipeline Configuration

### Programmatic Configuration

```python
# Complete pipeline configuration
config = {
    'model': {
        'id': 'meta-llama/Llama-2-7b-hf',
        'adapter': 'hf_llama',
        'device': 'auto',
        'torch_dtype': 'float16',
        'load_in_8bit': False
    },

    'dataset': {
        'provider': 'wikitext2',
        'seq_len': 512,
        'batch_size': 4,
        'num_samples': 1000,
        'seed': 42
    },

    # 'edit': {
    #     'name': 'quant_rtn',
    #     'parameters': {
    #         'bitwidth': 8,
    #         'per_channel': True
    #     }
    # },

    'guards': {
        'order': ['invariants', 'spectral', 'rmt', 'variance', 'invariants'],
        'invariants': {
            'strict_mode': True,
            'on_fail': 'abort'
        },
        'spectral': {
            'sigma_quantile': 0.95,
            'deadband': 0.10,
            'scope': 'all'
        },
        'rmt': {
            'margin': 1.5,
            'deadband': 0.10,
            'correct': True
        },
        'variance': {
            'min_gain': 0.0,
            'scope': 'ffn'
        }
    },

    'evaluation': {
        'enabled': True,
        'metrics': ['perplexity', 'accuracy'],
        'test_datasets': ['wikitext2'],
        'comparison_baseline': True
    },

    'output': {
        'dir': './results',
        'save_model': True,
        'save_metrics': True,
        'generate_report': True,
        'formats': ['json', 'html']
    }
}
```

### Dynamic Configuration

> Dynamic builders are not part of the stable public API. Compose configs as
> plain dictionaries/YAML and pass the relevant pieces to `CoreRunner.execute`.

### Configuration Validation

> Validate configuration using your own loader or schema tooling; the core
> types in `invarlock.core.api` intentionally keep validation light to avoid
> coupling programmatic callers to internal schemas.

## Custom Extensions

### Custom Adapter Development

```python
from invarlock.core.api import ModelAdapter

class CustomAdapter(ModelAdapter):
    name = "custom_adapter"

    def can_handle(self, model):
        # Check if this adapter can handle the model
        return hasattr(model, 'custom_architecture')

    def describe(self, model):
        # Return model description in standard format
        return {
            'n_layer': len(model.layers),
            'heads_per_layer': [8] * len(model.layers),
            'mlp_dims': [2048] * len(model.layers),
            'tying': {},
            'model_type': 'custom',
            'total_params': sum(p.numel() for p in model.parameters()),
            'device': str(next(model.parameters()).device)
        }

    def snapshot(self, model):
        # Create serialized snapshot
        state_dict = model.state_dict()
        return pickle.dumps(state_dict)

    def restore(self, model, snapshot):
        # Restore from snapshot
        state_dict = pickle.loads(snapshot)
        model.load_state_dict(state_dict)

    def load_model(self, model_id, device="auto"):
        # Load model implementation
        pass
```

### Custom Guard Development

```python
from invarlock.core.api import Guard

class CustomGuard(Guard):
    name = "custom_guard"

    def __init__(self, threshold=1.0, strict=False):
        super().__init__()
        self.threshold = threshold
        self.strict = strict
        self.baseline = None

    def prepare(self, model, adapter, calib_data, policy):
        # Initialize baseline measurements
        self.baseline = self._compute_baseline(model)
        return {
            "ready": True,
            "baseline_metrics": {"custom_metric": self.baseline}
        }

    def validate(self, model, adapter, context):
        # Validate current model state
        current_value = self._compute_current(model)
        violation = abs(current_value - self.baseline) > self.threshold

        return {
            "passed": not violation,
            "action": "abort" if (violation and self.strict) else "warn",
            "message": f"Custom metric: {current_value:.3f} (baseline: {self.baseline:.3f})",
            "metrics": {
                "current": current_value,
                "baseline": self.baseline,
                "difference": abs(current_value - self.baseline)
            }
        }

    def _compute_baseline(self, model):
        # Custom baseline computation
        pass

    def _compute_current(self, model):
        # Custom current state computation
        pass
```

### Plugin Registration

InvarLock discovers adapters, edits, and guards via [setuptools entry points](https://setuptools.pypa.io/en/latest/userguide/entry_point.html). To ship your own plugin:

1. Expose the implementation class (e.g. `MyCoolEdit`).
2. Add the entry point in your `pyproject.toml`:

```toml
[project.entry-points."invarlock.guards"]
my_guard = "my_package.my_module:MyGuard"
```

- Install the package in the CLI environment and confirm discovery:

```bash
invarlock plugins guards
#  … shows my_guard
```

Within tests or tooling you can access the registry programmatically:

```python
from invarlock.core.registry import CoreRegistry

registry = CoreRegistry()
available_guards = registry.list_guards()
info = registry.get_plugin_info("my_guard", "guards")
```

> Note: The sample `hello_guard` (`invarlock.plugins.hello_guard`) demonstrates guard entry-point wiring end-to-end. Use it as a template when authoring new plug-ins.

## Error Handling

### Exception Types

```python
from invarlock.core.exceptions import (
    InvarlockError,
    ModelLoadError,
    AdapterError,
    EditError,
    GuardError,
    ConfigError,
)

try:
    result = runner.run(config)
except ModelLoadError as e:
    # Loading/weights errors
    print(f"Failed to load model: {e}")
    print(f"code={e.code} recoverable={e.recoverable}")

except AdapterError as e:
    # Adapter resolution/device mapping problems
    print(f"Adapter error: {e}")
    print(f"code={e.code} details={e.details}")

except EditError as e:
    # Edit/transform failures
    print(f"Edit operation failed: {e}")
    if e.recoverable:
        print("Operation is recoverable; consider fallback parameters")

except GuardError as e:
    # Guard execution or policy failures
    print(f"Guard validation failed: {e}")
    print(f"code={e.code}")

except ConfigError as e:
    # Configuration parsing/validation issues
    print(f"Configuration error: {e}")

except InvarlockError as e:
    # Catch-all for InvarLock domain errors
    print(f"InvarLock error: {e}")
```

Typical categories and exit codes:

| Category | Typical causes | CLI exit code |
| --- | --- | --- |
| Success | No errors | `0` |
| Generic failure | Unexpected exceptions, non-domain errors | `1` |
| Schema/shape invalid | Invalid RunReport or report schema failures | `2` |
| Hard abort (CI/Release) | Any `InvarlockError` raised during CI/Release profiles | `3` |

### Graceful Error Recovery

```python
def robust_pipeline_execution(config):
    """Execute pipeline with comprehensive error handling."""

    try:
        # Initialize runner
        runner = CoreRunner()

        # Validate configuration
        validated_config = validate_config(config)

        # Run pipeline
        result = runner.run(validated_config)

        return result

    except ModelLoadError:
        # Try alternative models
        fallback_models = ['gpt2', 'distilgpt2']
        for model_id in fallback_models:
            try:
                config['model']['id'] = model_id
                return runner.run(config)
            except ModelLoadError:
                continue
        raise RuntimeError("No compatible models available")

    except GuardError as e:
        # Handle guard failures
        if e.recoverable:
            # Adjust guard policies and retry
            config['guards'] = get_fallback_guard_config()
            return runner.run(config)
        else:
            # Log failure and return error state
            return {'success': False, 'error': str(e)}

    except EditError as e:
        # Handle edit failures; try less aggressive parameters
        if 'sparsity' in config['edit']['parameters']:
            original_sparsity = config['edit']['parameters']['sparsity']
            config['edit']['parameters']['sparsity'] = original_sparsity * 0.5
            return runner.run(config)

        return {'success': False, 'error': str(e)}
```

## Best Practices

### Performance Optimization

```python
# 1. Use appropriate device placement
config['model']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Enable memory optimizations for large models
config['model']['low_cpu_mem_usage'] = True
config['model']['torch_dtype'] = 'float16'

# 3. Limit calibration data size
config['dataset']['num_samples'] = min(1000, config['dataset']['num_samples'])

# 4. Use efficient guard configurations
config['guards']['spectral']['scope'] = 'ffn'  # Focus on FFN layers
config['guards']['variance']['max_calib'] = 50  # Limit variance calibration

# 5. Enable checkpointing for long operations
config['output']['save_checkpoints'] = True
config['output']['checkpoint_interval'] = 100

# Snapshot controls for retry loops (CLI honors these in run_context)
config.setdefault('context', {}).setdefault('snapshot', {
    'mode': 'auto',                # auto | bytes | chunked
    'ram_fraction': 0.4,           # choose chunked when snapshot ≥ fraction × available RAM
    'threshold_mb': 768,           # fallback threshold when RAM not detectable
    'disk_free_margin_ratio': 1.2, # 20% headroom for chunked
})
```

### Memory Management

```python
import torch
import gc

def memory_efficient_pipeline(config):
    """Execute pipeline with memory management."""

    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Run pipeline
        runner = CoreRunner()
        result = runner.run(config)

        # Clean up after successful run
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        # Clean up on failure
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e
```

### Configuration Management

```python
# 1. Use configuration inheritance
# Example pattern (pseudo-code)
# base_config = yaml.safe_load(open('base_config.yaml'))
# experiment_config = yaml.safe_load(open('experiment_config.yaml'))
# final_config = deep_merge(base_config, experiment_config)

# 2. Validate configurations early
validate_config(config)

# 3. Use environment variables for deployment
config['model']['device'] = os.getenv('INVARLOCK_DEVICE', 'auto')
config['output']['dir'] = os.getenv('INVARLOCK_OUTPUT_DIR', './results')

# 4. Log configuration for reproducibility
import json
with open('used_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### Monitoring and Logging

```python
import logging
from invarlock.observability import MonitoringManager

# Set up observability
monitor = MonitoringManager()
monitor.start()

# Monitor pipeline execution
logger = logging.getLogger('invarlock.pipeline')

def monitored_pipeline(config):
    logger.info("Starting InvarLock pipeline")
    logger.info(f"Configuration: {config}")

    try:
        runner = CoreRunner()
        result = runner.run(config)

        # Log metrics
        metrics = monitor.get_metrics()
        logger.info("Pipeline completed successfully")
        logger.info(f"Performance metrics: {metrics}")

        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(f"Configuration: {config}")
        raise
```

This API guide provides comprehensive coverage of InvarLock's programmatic interface, enabling advanced users to integrate InvarLock into their applications and customize the pipeline for specific needs.
