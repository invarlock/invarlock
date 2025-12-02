from __future__ import annotations

from scripts import golden_runs


def test_golden_script_passes() -> None:
    rc = golden_runs.main()
    assert isinstance(rc, int)
    assert rc == 0
