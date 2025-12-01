from pathlib import Path

from invarlock.core.registry import get_registry


def test_hf_onnx_adapter_is_registered():
    registry = get_registry()
    info = registry.get_plugin_info("hf_onnx", "adapters")
    assert info["module"] in {"invarlock.adapters"}
    assert info["available"] is True


def test_auto_resolver_detects_local_onnx(tmp_path: Path):
    # Create a fake local ONNX export directory
    (tmp_path / "model.onnx").write_bytes(b"fake-onnx")
    # Add a minimal config for completeness (not required for detection)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    from invarlock.cli.adapter_auto import resolve_auto_adapter

    resolved = resolve_auto_adapter(str(tmp_path))
    assert resolved == "hf_onnx"
