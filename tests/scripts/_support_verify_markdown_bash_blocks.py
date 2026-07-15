from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "docs" / "verify_markdown_bash_blocks.py"
    spec = importlib.util.spec_from_file_location(
        "tests_verify_markdown_bash_blocks", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
