from __future__ import annotations

from invarlock.reporting.dataset_hashing import _compute_actual_window_hashes


def test_compute_actual_window_hashes_from_input_ids():
    report = {
        "data": {},
        "evaluation_windows": {
            "preview": {"input_ids": [[1, 2, 3]]},
            "final": {"input_ids": [[4, 5], [6]]},
        },
    }
    hashes = _compute_actual_window_hashes(report)
    assert hashes["preview"].startswith("sha256:")
    assert hashes["final"].startswith("sha256:")
    assert hashes["total_tokens"] == 3 + 2 + 1
