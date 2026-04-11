from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType

import pytest


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "security" / "run_pip_audit.py"
    spec = importlib.util.spec_from_file_location(
        "tests_run_pip_audit",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_allowlist(
    path: Path,
    *,
    tracking_issue: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "owner": "security-maintainers",
                "entries": [
                    {
                        "advisory": "GHSA-test-test-test",
                        "owner": "security-maintainers",
                        "expires": (date.today() + timedelta(days=7)).isoformat(),
                        "tracking_issue": tracking_issue,
                        "reason": "test fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_allowlist_rejects_non_issue_github_url(tmp_path: Path) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://github.com/pypa/pip/pull/13607",
    )

    with pytest.raises(SystemExit, match="must link to a GitHub tracking issue"):
        module._load_allowlist(allowlist)


def test_load_allowlist_rejects_non_github_tracking_url(tmp_path: Path) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://example.com/issues/13607",
    )

    with pytest.raises(SystemExit, match="must link to a GitHub tracking issue"):
        module._load_allowlist(allowlist)
