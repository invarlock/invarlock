from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runtime_tools():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "proof_packs" / "python" / "runtime_tools.py"
    spec = importlib.util.spec_from_file_location("proof_pack_runtime_tools", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_tools_python_helpers_use_portable_utc() -> None:
    runtime_tools = _load_runtime_tools()
    assert runtime_tools.iso_to_epoch("2025-01-01T00:00:10Z") == 1735689610
    assert runtime_tools.iso_to_epoch("") == 0
    assert runtime_tools.now_iso_plus_seconds(0).endswith("Z")
