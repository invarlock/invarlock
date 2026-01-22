# Plugin Workflow: Adapters and Guards

InvarLock's plugin system extends model loading and guard capabilities. Plugins do
not add additional edit algorithms beyond the built‑in RTN quantization.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Extend InvarLock with custom adapters and guards. |
| **Audience** | Developers adding model support or custom validation. |
| **Plugin types** | Adapters (model loading), Guards (validation checks). |
| **Registration** | Via `pyproject.toml` entry points. |
| **Source of truth** | `src/invarlock/plugins/hello_guard.py` (example). |

## Contents

1. [List Installed Plugins](#list-installed-plugins)
2. [Your First Plugin](#your-first-plugin)
3. [Guard Plugin (Complete Example)](#guard-plugin-complete-example)
4. [Adapter Plugin Example](#adapter-plugin-example)
5. [Plugin Debugging](#plugin-debugging)
6. [Related Documentation](#related-documentation)

## List Installed Plugins

```bash
# List all plugins
invarlock plugins

# List by category
invarlock plugins adapters
invarlock plugins guards
invarlock plugins edits

# JSON output for scripting
invarlock plugins list --json
```

## Your First Plugin

This walkthrough creates a minimal guard plugin from scratch.

### Step 1: Create Package Structure

```text
my_invarlock_plugin/
├── pyproject.toml
├── src/
│   └── my_plugin/
│       ├── __init__.py
│       └── my_guard.py
└── tests/
    └── test_my_guard.py
```

### Step 2: Implement the Guard

```python
# src/my_plugin/my_guard.py
"""A simple guard that checks for NaN values in weights."""

from invarlock.core.api import Guard


class NaNCheckGuard(Guard):
    """Guard that fails if any model weight contains NaN."""
    
    name = "nan_check"
    
    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, abort on NaN. If False, warn only.
        """
        self.strict = strict
    
    def prepare(self, model, adapter, calib, policy):
        """Called before the edit is applied."""
        return {"ready": True, "checked_params": 0}
    
    def validate(self, model, adapter, context):
        """Called after the edit is applied."""
        import torch
        
        nan_params = []
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if torch.isnan(param).any():
                nan_params.append(name)
        
        if nan_params:
            action = "abort" if self.strict else "warn"
            return {
                "passed": False,
                "action": action,
                "message": f"Found NaN in {len(nan_params)} parameters",
                "metrics": {
                    "nan_param_count": len(nan_params),
                    "nan_params": nan_params[:5],  # First 5 for brevity
                },
            }
        
        return {
            "passed": True,
            "action": "continue",
            "message": f"All {total_params} parameters are finite",
            "metrics": {"checked_params": total_params},
        }
```

### Step 3: Register in pyproject.toml

```toml
[project]
name = "my-invarlock-plugin"
version = "0.1.0"
dependencies = ["invarlock>=0.1.0"]

[project.entry-points."invarlock.guards"]
nan_check = "my_plugin.my_guard:NaNCheckGuard"
```

### Step 4: Install and Test

```bash
# Install in editable mode
pip install -e ./my_invarlock_plugin

# Verify registration
invarlock plugins guards
# Should show: nan_check | Plugin | Guard | — | — | ✅ Ready

# Use in a run
invarlock run -c config.yaml
```

### Step 5: Add to Config

```yaml
guards:
  order: ["invariants", "nan_check", "spectral", "variance"]
  nan_check:
    strict: true
```

## Guard Plugin (Complete Example)

This example shows a production-ready guard with policy support and tests.

### Guard Implementation

```python
# src/my_plugin/threshold_guard.py
"""Guard that monitors weight magnitude changes."""

from typing import Any

from invarlock.core.api import Guard


class ThresholdGuard(Guard):
    """
    Monitors maximum weight magnitude after edits.
    
    Fails if any weight exceeds the configured threshold, which can
    indicate numerical instability from aggressive quantization.
    """
    
    name = "threshold"
    
    def __init__(
        self,
        max_magnitude: float = 100.0,
        warn_magnitude: float = 50.0,
        scope: str = "all",
    ):
        """
        Args:
            max_magnitude: Abort threshold for weight magnitude.
            warn_magnitude: Warning threshold for weight magnitude.
            scope: Which layers to check ("all", "ffn", "attn").
        """
        self.max_magnitude = max_magnitude
        self.warn_magnitude = warn_magnitude
        self.scope = scope
        self._baseline_magnitudes: dict[str, float] = {}
    
    def prepare(self, model, adapter, calib, policy) -> dict[str, Any]:
        """Capture baseline weight magnitudes."""
        import torch
        
        self._baseline_magnitudes = {}
        for name, param in model.named_parameters():
            if self._in_scope(name):
                self._baseline_magnitudes[name] = param.abs().max().item()
        
        return {
            "ready": True,
            "baseline_params": len(self._baseline_magnitudes),
            "scope": self.scope,
        }
    
    def validate(self, model, adapter, context) -> dict[str, Any]:
        """Check weight magnitudes after edit."""
        import torch
        
        violations = []
        warnings = []
        max_seen = 0.0
        
        for name, param in model.named_parameters():
            if not self._in_scope(name):
                continue
                
            magnitude = param.abs().max().item()
            max_seen = max(max_seen, magnitude)
            
            if magnitude > self.max_magnitude:
                violations.append({
                    "param": name,
                    "magnitude": magnitude,
                    "baseline": self._baseline_magnitudes.get(name, 0),
                })
            elif magnitude > self.warn_magnitude:
                warnings.append({
                    "param": name,
                    "magnitude": magnitude,
                })
        
        if violations:
            return {
                "passed": False,
                "action": "abort",
                "message": f"{len(violations)} params exceed magnitude threshold",
                "warnings": [w["param"] for w in warnings],
                "metrics": {
                    "max_magnitude": max_seen,
                    "violations": violations[:3],
                    "violation_count": len(violations),
                },
            }
        
        if warnings:
            return {
                "passed": True,
                "action": "warn",
                "message": f"{len(warnings)} params near threshold",
                "warnings": [w["param"] for w in warnings],
                "metrics": {
                    "max_magnitude": max_seen,
                    "warning_count": len(warnings),
                },
            }
        
        return {
            "passed": True,
            "action": "continue",
            "message": "All weight magnitudes within bounds",
            "metrics": {"max_magnitude": max_seen},
        }
    
    def _in_scope(self, name: str) -> bool:
        """Check if parameter is in configured scope."""
        if self.scope == "all":
            return True
        if self.scope == "ffn":
            return "mlp" in name.lower() or "ffn" in name.lower()
        if self.scope == "attn":
            return "attn" in name.lower() or "attention" in name.lower()
        return True
```

### Test Suite

```python
# tests/test_threshold_guard.py
"""Tests for ThresholdGuard."""

import pytest
import torch
from torch import nn

from my_plugin.threshold_guard import ThresholdGuard


class MockAdapter:
    """Minimal adapter for testing."""
    def describe(self, model):
        return {"model_type": "test"}


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.attn = nn.Linear(10, 10)


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def adapter():
    return MockAdapter()


def test_prepare_captures_baselines(model, adapter):
    """Test that prepare() captures baseline magnitudes."""
    guard = ThresholdGuard()
    result = guard.prepare(model, adapter, None, None)
    
    assert result["ready"] is True
    assert result["baseline_params"] > 0


def test_validate_passes_normal_weights(model, adapter):
    """Test validation passes for normal weight magnitudes."""
    guard = ThresholdGuard(max_magnitude=100.0)
    guard.prepare(model, adapter, None, None)
    
    result = guard.validate(model, adapter, {})
    
    assert result["passed"] is True
    assert result["action"] == "continue"


def test_validate_fails_large_weights(model, adapter):
    """Test validation fails when weights exceed threshold."""
    guard = ThresholdGuard(max_magnitude=0.01)  # Very low threshold
    guard.prepare(model, adapter, None, None)
    
    result = guard.validate(model, adapter, {})
    
    assert result["passed"] is False
    assert result["action"] == "abort"
    assert "violations" in result["metrics"]


def test_validate_warns_near_threshold(model, adapter):
    """Test validation warns when weights approach threshold."""
    # Set warn threshold just below actual magnitudes
    guard = ThresholdGuard(max_magnitude=100.0, warn_magnitude=0.01)
    guard.prepare(model, adapter, None, None)
    
    result = guard.validate(model, adapter, {})
    
    assert result["passed"] is True
    assert result["action"] == "warn"


def test_scope_filtering(model, adapter):
    """Test that scope correctly filters parameters."""
    guard = ThresholdGuard(scope="attn")
    guard.prepare(model, adapter, None, None)
    
    # Only attn layer should be captured
    assert len(guard._baseline_magnitudes) == 2  # weight + bias
    assert all("attn" in k for k in guard._baseline_magnitudes)
```

Run tests:

```bash
pytest tests/test_threshold_guard.py -v
```

## Adapter Plugin Example

Adapters handle model loading for specific formats. This example shows a
skeleton for a custom adapter.

```python
# src/my_plugin/custom_adapter.py
"""Adapter for custom model format."""

from pathlib import Path
from typing import Any

from invarlock.core.api import ModelAdapter


class CustomFormatAdapter(ModelAdapter):
    """
    Adapter for loading models in a custom format.
    
    This adapter demonstrates the required interface. Replace the
    implementation with your actual loading logic.
    """
    
    name = "custom_format"
    
    def load_model(
        self,
        model_id: str,
        device: str = "auto",
        **kwargs,
    ) -> Any:
        """
        Load a model from the custom format.
        
        Args:
            model_id: Path to model directory or identifier.
            device: Target device ("auto", "cpu", "cuda", "mps").
            **kwargs: Additional loading arguments.
        
        Returns:
            Loaded model instance.
        """
        import torch
        
        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Load your model here
        model_path = Path(model_id)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        
        # Example: load a state dict
        state_dict = torch.load(model_path / "model.pt", map_location=device)
        
        # Create and load model architecture
        # model = YourModelClass()
        # model.load_state_dict(state_dict)
        # model.to(device)
        # return model
        
        raise NotImplementedError("Replace with actual loading logic")
    
    def describe(self, model) -> dict[str, Any]:
        """
        Describe model structure for guards and reporting.
        
        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_type": "custom",
            "model_class": type(model).__name__,
            "n_layer": getattr(model, "n_layer", None),
            "n_head": getattr(model, "n_head", None),
            # Add other relevant metadata
        }
    
    def snapshot(self, model) -> bytes:
        """Create an in-memory snapshot for retry loops."""
        import io
        import torch
        
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()
    
    def restore(self, model, snapshot: bytes) -> None:
        """Restore model state from snapshot."""
        import io
        import torch
        
        buffer = io.BytesIO(snapshot)
        state_dict = torch.load(buffer, map_location="cpu")
        model.load_state_dict(state_dict)
```

Register in `pyproject.toml`:

```toml
[project.entry-points."invarlock.adapters"]
custom_format = "my_plugin.custom_adapter:CustomFormatAdapter"
```

## Plugin Debugging

### Check Registration

```bash
# Verify plugin is discovered
invarlock plugins list --verbose

# Get details for specific plugin
invarlock plugins adapters --explain custom_format
# For guards, use --verbose to show module and entry point details
invarlock plugins guards --verbose
```

### Debug Loading Issues

```bash
# Enable debug logging
INVARLOCK_DEBUG_TRACE=1 invarlock plugins list

# Check for import errors
python -c "from my_plugin.my_guard import NaNCheckGuard; print('OK')"
```

### Test in Isolation

```python
# test_integration.py
from my_plugin.my_guard import NaNCheckGuard
from invarlock.adapters import HF_Causal_Adapter

adapter = HF_Causal_Adapter()
model = adapter.load_model("gpt2", device="cpu")

guard = NaNCheckGuard()
prep = guard.prepare(model, adapter, None, None)
print(f"Prepare: {prep}")

result = guard.validate(model, adapter, {})
print(f"Validate: {result}")
```

### Common Issues

| Issue | Cause | Fix |
| --- | --- | --- |
| Plugin not listed | Entry point not found | Check `pyproject.toml` syntax and reinstall. |
| Import error | Missing dependency | Add to `project.dependencies`. |
| `passed` key missing | Incomplete return dict | Include `passed`, `action`, `message`. |
| Guard skipped | Not in `guards.order` | Add guard name to order list. |

## Related Documentation

- [Guards Reference](../reference/guards.md) — Built-in guard configuration
- [Model Adapters](../reference/model-adapters.md) — Built-in adapter reference
- [CLI Reference](../reference/cli.md#plugins-entry-points) — Plugin CLI commands
- [Configuration Schema](../reference/config-schema.md) — YAML config for guards
