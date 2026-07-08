from __future__ import annotations

import json
from pathlib import Path


def test_events_dump_matches_window_count(tmp_path: Path):
    # Arrange: create a tiny golden report with preview/final windows
    ev_path = tmp_path / "events.jsonl"
    # 3 preview + 2 final = 5 total lines expected
    with ev_path.open("a", encoding="utf-8") as events:
        for _ in range(5):
            events.write("{}\n")  # simple JSONL placeholders
    report = {
        "artifacts": {"events_path": str(ev_path)},
        "evaluation_windows": {
            "preview": {"window_ids": [1, 2, 3]},
            "final": {"window_ids": [10, 20]},
        },
    }
    rpt_path = tmp_path / "report.json"
    rpt_path.write_text(json.dumps(report))

    # Act
    rpt = json.loads(rpt_path.read_text())
    got = sum(1 for _ in open(rpt["artifacts"]["events_path"], encoding="utf-8"))
    win = rpt["evaluation_windows"]
    expected = len(win["preview"]["window_ids"]) + len(win["final"]["window_ids"])

    # Assert
    assert got == expected, f"events JSONL rows {got} != expected {expected}"
