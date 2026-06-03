from __future__ import annotations

from pathlib import Path


def _slice_section(text: str, start_marker: str, end_marker: str | None) -> str:
    start = text.find(start_marker)
    assert start != -1, f"Missing section start: {start_marker!r}"
    if end_marker is None:
        return text[start:]
    end = text.find(end_marker, start)
    assert end != -1, f"Missing section end: {end_marker!r}"
    return text[start:end]


def test_task_evaluate_edit_passes_baseline_report() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (
        repo_root / "scripts/evidence_packs/lib/tasks/task_edit_lifecycle.sh"
    ).read_text(encoding="utf-8")
    section = _slice_section(
        text,
        "task_evaluate_edit() {",
        "# ============ TASK: CLEANUP_EDIT",
    )
    assert "--baseline-report" in section


def test_task_evaluate_error_passes_baseline_report() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (
        repo_root / "scripts/evidence_packs/lib/tasks/task_error_lifecycle.sh"
    ).read_text(encoding="utf-8")
    section = _slice_section(text, "task_evaluate_error() {", None)
    assert "--baseline-report" in section
