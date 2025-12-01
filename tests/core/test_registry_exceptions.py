from __future__ import annotations

import pytest

import invarlock.core.registry as reg


def test_get_guard_typed_missing_plugin_raises_plugin_error(monkeypatch):
    r = reg.CoreRegistry()
    # Ensure the plugin name is unknown
    missing = "__totally_missing_guard__"
    with pytest.raises(Exception) as ei:
        r.get_guard_typed(missing)
    exc = ei.value
    from invarlock.core.exceptions import PluginError

    assert isinstance(exc, PluginError)
    assert exc.code == "E701"
    assert exc.details and exc.details.get("name") == missing
    assert exc.details.get("kind") == "guard"
