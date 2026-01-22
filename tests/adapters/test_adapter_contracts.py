from __future__ import annotations

import pytest

from invarlock.core.registry import get_registry


def test_plugin_adapters_present_in_registry():
    reg = get_registry()
    adapters = set(reg.list_adapters())
    # Built-ins
    for name in {"hf_causal", "hf_mlm", "hf_seq2seq", "hf_causal_onnx", "hf_auto"}:
        assert name in adapters
    # Optional plugins should be discoverable (even if not importable)
    for name in {"hf_gptq", "hf_awq", "hf_bnb"}:
        assert name in adapters


@pytest.mark.unit
def test_registry_metadata_shows_entry_points_without_loading():
    reg = get_registry()
    # Accessing list does not instantiate; expect readable names
    names = reg.list_adapters()
    assert isinstance(names, list)
    assert any(n.startswith("hf_") for n in names)
