from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "verify_notebooks_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "tests_verify_notebooks_smoke", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_script_rewrites_notebook_shell_commands(tmp_path: Path) -> None:
    module = _load_script_module()
    nb_path = tmp_path / "demo.ipynb"
    nb_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["!invarlock doctor --json || true\n"],
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "%%bash\n",
                            "invarlock evaluate --baseline gpt2 --subject gpt2\n",
                            "test -f reports/eval/runtime.manifest.json\n",
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_py = tmp_path / "demo.py"

    module.write_script(nb_path=nb_path, out_py=out_py, skip_pip=True)

    rendered = out_py.read_text(encoding="utf-8")
    assert "sys.executable" in rendered
    assert 'replacement = f"{indent}{env_prefix}{py} -m invarlock"' in rendered
    assert "return f\"echo '[skip-host] {stripped}'\"" in rendered
    assert "_run_bash('invarlock doctor --json || true')" in rendered
    assert (
        "_run_bash('invarlock evaluate --baseline gpt2 --subject gpt2\\n"
        "test -f reports/eval/runtime.manifest.json\\n')"
    ) in rendered
