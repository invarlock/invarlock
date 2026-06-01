from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSPECTOR_PATH = (
    REPO_ROOT
    / "examples"
    / "integrations"
    / "optimum_onnx_export"
    / "inspect_optimum_onnx_export.py"
)


def _load_inspector():
    spec = importlib.util.spec_from_file_location(
        "optimum_onnx_inspector", INSPECTOR_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_file_rows_sorts_and_hashes_files(tmp_path: Path) -> None:
    inspector = _load_inspector()
    (tmp_path / "z.txt").write_text("last\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first\n", encoding="utf-8")

    rows = inspector.collect_file_rows(tmp_path)

    assert [row["path"] for row in rows] == ["a.txt", "z.txt"]
    assert [row["bytes"] for row in rows] == [6, 5]
    assert rows[0]["sha256"] == inspector.sha256_file(tmp_path / "a.txt")
    assert rows[1]["sha256"] == inspector.sha256_file(tmp_path / "z.txt")


def test_probe_error_records_stable_error_shape() -> None:
    inspector = _load_inspector()

    result = inspector.probe_error(ValueError("first line\nsecond line"))

    assert result == {
        "ok": False,
        "error_type": "ValueError",
        "error_message": "first line",
    }
