from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "model_evidence" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = module
    spec.loader.exec_module(module)
    return module
