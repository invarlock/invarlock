from __future__ import annotations

from pathlib import Path

from invarlock.reporting import verify_output


def test_build_verify_json_result_item_omits_bool_numeric_fields() -> None:
    item = verify_output.build_verify_json_result_item(
        Path("report.json"),
        {
            "primary_metric": {
                "kind": "accuracy",
                "ratio_vs_baseline": True,
                "display_ci": [True, False],
            }
        },
        ok=True,
        reason="ok",
        tolerance=1e-9,
    )

    assert item["ratio_vs_baseline"] is None
    assert item["ci"] is None


def test_build_verify_success_line_omits_bool_counts_and_point() -> None:
    line = verify_output.build_verify_success_line(
        {
            "primary_metric": {
                "kind": "accuracy",
                "ratio_vs_baseline": True,
                "display_ci": [True, False],
            },
            "ppl": {
                "stats": {
                    "coverage": {"preview": {"used": True}, "final": {"used": False}}
                }
            },
        }
    )

    assert line == "VERIFY OK metric=accuracy"
