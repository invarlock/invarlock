from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_evidence_contracts_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "release" / "evidence_contracts.py"
    script_dir = str(module_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "empirical_guard_evidence_contracts_under_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
