from __future__ import annotations


def test_provider_network_wikitext2_label() -> None:
    from invarlock.cli.constants import PROVIDER_NETWORK

    assert PROVIDER_NETWORK.get("wikitext2") == "cache"


def test_hf_causal_onnx_adapter_needs_extra_when_missing(monkeypatch) -> None:
    # Monkeypatch registry to expose only hf_causal_onnx as a core adapter
    class _FakeRegistry:
        def list_adapters(self):
            return ["hf_causal_onnx"]

        def get_plugin_info(self, name: str, kind: str):
            assert kind == "adapters"
            return {
                "module": "invarlock.adapters.hf_causal_onnx",
                "entry_point": "HF_Causal_ONNX_Adapter",
            }

        def list_guards(self):
            return []

        def list_edits(self):
            return []

    def _fake_get_registry():
        return _FakeRegistry()

    monkeypatch.setattr(
        "invarlock.core.registry.get_registry", _fake_get_registry, raising=True
    )

    # Import doctor and fetch adapter rows via the exposed helper
    from invarlock.cli import doctor_helpers as doctor_mod

    monkeypatch.setattr(
        doctor_mod, "onnx_causal_runtime_available", lambda: False, raising=False
    )
    rows = doctor_mod.get_adapter_rows()
    # Expect exactly one row for hf_causal_onnx and that it needs extras with an enable hint
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "hf_causal_onnx"
    assert row["status"] == "needs_extra"
    assert "invarlock[onnx]" in (row.get("enable") or "")
