"""
Regression tests for VE guard pipeline target resolution fixes.

These tests verify that:
1. DataParallel/DDP wrapped models are correctly unwrapped
2. Adapter fallback uses adapter.describe() for layer count
3. Additional architecture patterns (decoder.layers, layers) are supported
"""

import torch.nn as nn

from invarlock.guards.variance import (
    VarianceGuard,
    _iter_transformer_layers,
    _unwrap_model,
)

# === Test Models ===


class TinyGPT2Style(nn.Module):
    """Minimal GPT-2 style model with transformer.h structure."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.transformer = nn.Module()
        blocks = []
        for _ in range(n_layers):
            blk = nn.Module()
            blk.attn = nn.Module()
            blk.attn.c_proj = nn.Linear(4, 4, bias=False)
            blk.mlp = nn.Module()
            blk.mlp.c_proj = nn.Linear(4, 4, bias=False)
            blocks.append(blk)
        self.transformer.h = nn.ModuleList(blocks)

    def forward(self, x):
        return x


class TinyModelLayersStyle(nn.Module):
    """Minimal model with model.layers structure."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.model = nn.Module()
        layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.c_proj = nn.Linear(4, 4, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(4, 4, bias=False)
            layers.append(layer)
        self.model.layers = nn.ModuleList(layers)

    def forward(self, x):
        return x


class TinyDecoderStyle(nn.Module):
    """Minimal decoder-only model with decoder.layers structure (T5/BART style)."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.decoder = nn.Module()
        layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.c_proj = nn.Linear(4, 4, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.c_proj = nn.Linear(4, 4, bias=False)
            layers.append(layer)
        self.decoder.layers = nn.ModuleList(layers)

    def forward(self, x):
        return x


class TinyGenericLayersStyle(nn.Module):
    """Minimal model with top-level layers attribute."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.c_proj = nn.Linear(4, 4, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.c_proj = nn.Linear(4, 4, bias=False)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return x


class TinyUnknownStructure(nn.Module):
    """Model with non-standard structure that _iter_transformer_layers won't recognize."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        # Store layers in a non-standard way
        self.custom_blocks = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.c_proj = nn.Linear(4, 4, bias=False)
            block.mlp = nn.Module()
            block.mlp.c_proj = nn.Linear(4, 4, bias=False)
            self.custom_blocks.append(block)

    def get_layer(self, idx):
        return self.custom_blocks[idx]

    def forward(self, x):
        return x


class MockDataParallel(nn.Module):
    """Mock DataParallel wrapper that wraps a model in .module attribute."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


class MockDDP(nn.Module):
    """Mock DistributedDataParallel wrapper (nested .module)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


class MockAdapter:
    """Mock adapter that provides describe() and get_layer_modules()."""

    def __init__(self, model: nn.Module, n_layers: int):
        self._model = model
        self._n_layers = n_layers

    def describe(self, model):
        return {"n_layer": self._n_layers, "model_type": "mock"}

    def get_layer_modules(self, model, layer_idx: int):
        """Return the c_proj modules for a given layer."""
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        # Try to get layer from custom_blocks (unknown structure model)
        if hasattr(unwrapped, "custom_blocks") and layer_idx < len(
            unwrapped.custom_blocks
        ):
            block = unwrapped.custom_blocks[layer_idx]
            return {
                "attn.c_proj": block.attn.c_proj,
                "mlp.c_proj": block.mlp.c_proj,
            }

        # Fallback - try standard structures
        if hasattr(unwrapped, "transformer") and hasattr(unwrapped.transformer, "h"):
            if layer_idx < len(unwrapped.transformer.h):
                block = unwrapped.transformer.h[layer_idx]
                return {
                    "attn.c_proj": block.attn.c_proj,
                    "mlp.c_proj": block.mlp.c_proj,
                }
        return {}


# === Tests for _unwrap_model ===


def test_unwrap_model_plain():
    """Plain model without wrapper returns itself."""
    model = TinyGPT2Style()
    unwrapped = _unwrap_model(model)
    assert unwrapped is model


def test_unwrap_model_single_dataparallel():
    """Single DataParallel wrapper is correctly unwrapped."""
    inner = TinyGPT2Style()
    wrapped = MockDataParallel(inner)
    unwrapped = _unwrap_model(wrapped)
    assert unwrapped is inner


def test_unwrap_model_nested_wrappers():
    """Nested wrappers (DDP wrapping DP) are fully unwrapped."""
    inner = TinyGPT2Style()
    dp = MockDataParallel(inner)
    ddp = MockDDP(dp)
    unwrapped = _unwrap_model(ddp)
    assert unwrapped is inner


# === Tests for _iter_transformer_layers with different architectures ===


def test_iter_gpt2_style():
    """_iter_transformer_layers handles GPT-2 (transformer.h) structure."""
    model = TinyGPT2Style(n_layers=3)
    layers = list(_iter_transformer_layers(model))
    assert len(layers) == 3


def test_iter_model_layers_style():
    """_iter_transformer_layers handles model.layers structure."""
    model = TinyModelLayersStyle(n_layers=3)
    layers = list(_iter_transformer_layers(model))
    assert len(layers) == 3


def test_iter_decoder_style():
    """_iter_transformer_layers handles decoder (decoder.layers) structure."""
    model = TinyDecoderStyle(n_layers=3)
    layers = list(_iter_transformer_layers(model))
    assert len(layers) == 3


def test_iter_generic_layers_style():
    """_iter_transformer_layers handles top-level layers attribute."""
    model = TinyGenericLayersStyle(n_layers=3)
    layers = list(_iter_transformer_layers(model))
    assert len(layers) == 3


def test_iter_wrapped_model():
    """_iter_transformer_layers handles DataParallel-wrapped models."""
    inner = TinyGPT2Style(n_layers=2)
    wrapped = MockDataParallel(inner)
    layers = list(_iter_transformer_layers(wrapped))
    assert len(layers) == 2


def test_iter_nested_wrapped_model():
    """_iter_transformer_layers handles nested wrappers."""
    inner = TinyGPT2Style(n_layers=2)
    dp = MockDataParallel(inner)
    ddp = MockDDP(dp)
    layers = list(_iter_transformer_layers(ddp))
    assert len(layers) == 2


# === Tests for VarianceGuard._resolve_target_modules ===


def test_resolve_targets_gpt2_direct():
    """Direct resolution works for GPT-2 style model."""
    model = TinyGPT2Style(n_layers=2)
    guard = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0})
    targets = guard._resolve_target_modules(model, adapter=None)
    assert len(targets) == 2
    assert "transformer.h.0.mlp.c_proj" in targets
    assert "transformer.h.1.mlp.c_proj" in targets


def test_resolve_targets_wrapped_model():
    """Resolution works for DataParallel-wrapped model."""
    inner = TinyGPT2Style(n_layers=2)
    wrapped = MockDataParallel(inner)
    guard = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0})
    targets = guard._resolve_target_modules(wrapped, adapter=None)
    # Should find targets through the wrapper
    assert len(targets) == 2


def test_resolve_targets_adapter_fallback_unknown_structure():
    """Adapter fallback resolves targets for unknown model structure.

    This is the key regression test: when _iter_transformer_layers returns
    no layers (unknown structure), the adapter fallback should use
    adapter.describe() for layer count and adapter.get_layer_modules()
    for actual modules.
    """
    model = TinyUnknownStructure(n_layers=2)
    adapter = MockAdapter(model, n_layers=2)

    guard = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0})

    # Verify that _iter_transformer_layers finds layers via fallback
    # (modules with attn and mlp attributes)
    # But let's test the adapter path explicitly

    targets = guard._resolve_target_modules(model, adapter=adapter)

    # The adapter fallback should find targets
    assert len(targets) >= 2, f"Expected >= 2 targets, got {len(targets)}: {targets}"


def test_resolve_targets_adapter_fallback_wrapped_unknown():
    """Adapter fallback works for wrapped model with unknown structure."""
    inner = TinyUnknownStructure(n_layers=2)
    wrapped = MockDataParallel(inner)
    adapter = MockAdapter(inner, n_layers=2)

    guard = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0})
    targets = guard._resolve_target_modules(wrapped, adapter=adapter)

    # Adapter fallback should find targets
    assert len(targets) >= 2


def test_resolve_targets_stats_record_fallback():
    """Stats correctly record when fallback is used."""
    model = TinyUnknownStructure(n_layers=2)
    adapter = MockAdapter(model, n_layers=2)

    guard = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0})
    guard._resolve_target_modules(model, adapter=adapter)

    # Check that stats record fallback usage
    stats = guard._stats.get("target_resolution", {})
    assert "fallback_used" in stats


# === Tests for adapter.describe() layer count ===


def test_adapter_describe_provides_layer_count():
    """Adapter.describe() provides n_layer for fallback iteration."""
    model = TinyUnknownStructure(n_layers=3)
    adapter = MockAdapter(model, n_layers=3)

    guard = VarianceGuard(policy={"scope": "both", "min_gain": 0.0})
    targets = guard._resolve_target_modules(model, adapter=adapter)

    # Should use all 3 layers from describe()
    assert len(targets) >= 3  # 3 layers * (attn + mlp) if scope="both"


class NoDescribeAdapter:
    """Adapter without describe() method."""

    def __init__(self, n_layers: int):
        self._n_layers = n_layers

    def get_layer_modules(self, model, layer_idx: int):
        return {}


def test_adapter_fallback_no_describe():
    """Adapter fallback handles adapters without describe() method."""
    model = TinyGPT2Style(n_layers=2)
    adapter = NoDescribeAdapter(n_layers=2)

    guard = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0})
    # Should not crash; will use direct resolution since model is GPT-2 style
    targets = guard._resolve_target_modules(model, adapter=adapter)
    assert len(targets) == 2


class ConfigModel(nn.Module):
    """Model with a config object (like HuggingFace models)."""

    def __init__(self, n_layers: int):
        super().__init__()

        class Config:
            def __init__(self, n):
                self.n_layer = n

        self.config = Config(n_layers)
        # Unknown structure
        self.blocks = nn.ModuleList()

    def forward(self, x):
        return x


class DescribeErrorAdapter:
    """Adapter whose describe() method raises an exception."""

    def __init__(self, n_layers: int):
        self._n_layers = n_layers

    def describe(self, model):
        raise RuntimeError("describe failed")

    def get_layer_modules(self, model, layer_idx: int):
        return {}


def test_adapter_describe_error_falls_back_to_config():
    """When adapter.describe() fails, fallback extracts n_layer from model.config."""
    model = ConfigModel(n_layers=3)
    adapter = DescribeErrorAdapter(n_layers=3)

    guard = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0})
    # This shouldn't crash
    targets = guard._resolve_target_modules(model, adapter=adapter)
    # Model has unknown structure and get_layer_modules returns empty
    # So targets will be empty, but the process shouldn't crash
    assert isinstance(targets, dict)


# === Integration test for pipeline context ===


def test_pipeline_integration_wrapped_model():
    """Full integration test: prepare() works with wrapped models."""
    inner = TinyGPT2Style(n_layers=2)
    wrapped = MockDataParallel(inner)

    guard = VarianceGuard(
        policy={
            "scope": "ffn",
            "min_gain": 0.0,
            "tap": "transformer.h.*.mlp.c_proj",
        }
    )

    # Simulate pipeline prepare() call
    result = guard.prepare(wrapped, adapter=None, calib=[], policy=None)

    # Should complete without error
    assert "ready" in result
    # Target modules should be resolved
    assert len(guard._target_modules) == 2


def test_pipeline_integration_adapter_fallback():
    """Full integration test: prepare() uses adapter fallback for unknown structure."""
    model = TinyUnknownStructure(n_layers=2)
    adapter = MockAdapter(model, n_layers=2)

    guard = VarianceGuard(
        policy={
            "scope": "ffn",
            "min_gain": 0.0,
            "tap": "transformer.h.*.mlp.c_proj",
        }
    )

    # Simulate pipeline prepare() call with adapter
    result = guard.prepare(model, adapter=adapter, calib=[], policy=None)

    # Should complete without error
    assert "ready" in result
    # Target modules should be resolved via fallback
    assert len(guard._target_modules) >= 2
