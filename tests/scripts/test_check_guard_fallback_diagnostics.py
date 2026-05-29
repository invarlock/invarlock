from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_checker():
    path = Path("scripts/checks/check_guard_fallback_diagnostics.py")
    spec = importlib.util.spec_from_file_location(
        "check_guard_fallback_diagnostics", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_guard(root: Path, text: str) -> None:
    guard_dir = root / "src" / "invarlock" / "guards"
    guard_dir.mkdir(parents=True)
    (guard_dir / "bad_guard.py").write_text(text, encoding="utf-8")


def test_guard_fallback_checker_rejects_silent_numeric_except_return(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    _write_guard(
        tmp_path,
        """
def measure(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0
""",
    )

    ok, errors = checker.check_guard_fallbacks(root=tmp_path)

    assert not ok
    assert "risky numeric fallback" in errors[0]


def test_guard_fallback_checker_accepts_explicit_rationale(tmp_path: Path) -> None:
    checker = _load_checker()
    _write_guard(
        tmp_path,
        """
def pvalue(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        # guard-fallback-ok: invalid values are conservative no-reject p-values.
        return 1.0
""",
    )

    ok, errors = checker.check_guard_fallbacks(root=tmp_path)

    assert ok, errors
