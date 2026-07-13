from __future__ import annotations

from pathlib import Path

from invarlock.reporting import verify_output


def test_build_verify_json_result_item_omits_bool_numeric_fields() -> None:
    item = verify_output.build_verify_json_result_item(
        Path("report.json"),
        {
            "primary_metric": {
                "kind": "accuracy",
                "delta_vs_baseline_pp": True,
                "display_ci": [True, False],
            }
        },
        ok=True,
        reason="ok",
        tolerance=1e-9,
    )

    assert item["delta_vs_baseline_pp"] is None
    assert "ratio_vs_baseline" not in item
    assert item["ci"] is None
    assert item["recompute"] == {
        "family": "accuracy",
        "performed": False,
        "ok": None,
        "reason": "missing_evidence",
    }


def test_build_verify_success_line_omits_bool_counts_and_point() -> None:
    line = verify_output.build_verify_success_line(
        {
            "primary_metric": {
                "kind": "accuracy",
                "delta_vs_baseline_pp": True,
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


def test_build_verify_json_result_rejects_malformed_accuracy_counts() -> None:
    cases = [
        ({"n_correct": "bad", "n_total": 2}, "malformed_evidence"),
        ({"n_correct": 1, "n_total": 0}, "zero_denominator"),
        ({"n_correct": -1, "n_total": 2}, "malformed_evidence"),
        ({"n_correct": 3, "n_total": 2}, "malformed_evidence"),
    ]
    for classification, expected_reason in cases:
        item = verify_output.build_verify_json_result_item(
            Path("report.json"),
            {
                "primary_metric": {"kind": "accuracy", "final": 0.5},
                "metrics": {"classification": classification},
            },
            ok=False,
            reason="invalid",
            tolerance=1e-9,
        )
        assert item["recompute"]["reason"] == expected_reason


def test_build_verify_json_result_rejects_malformed_ppl_windows() -> None:
    malformed_windows = [
        None,
        {},
        {"logloss": "bad", "token_counts": [1]},
        {"logloss": [], "token_counts": []},
        {"logloss": [1, 2], "token_counts": [1]},
        {"logloss": [float("nan")], "token_counts": [1]},
        {"logloss": [1], "token_counts": [-1]},
        {"logloss": [1], "token_counts": [0]},
    ]
    observed = []
    for final in malformed_windows:
        item = verify_output.build_verify_json_result_item(
            Path("report.json"),
            {
                "primary_metric": {"kind": "ppl_causal", "final": 1.0},
                "evaluation_windows": {"final": final},
            },
            ok=False,
            reason="invalid",
            tolerance=1e-9,
        )
        observed.append(item["recompute"]["reason"])
    assert observed == [
        "missing_evidence",
        "missing_evidence",
        "malformed_evidence",
        "missing_evidence",
        "malformed_evidence",
        "malformed_evidence",
        "malformed_evidence",
        "zero_denominator",
    ]


def test_build_verify_json_result_reports_recompute_mismatch_and_warning_fallback() -> (
    None
):
    item = verify_output.build_verify_json_result_item(
        Path("report.json"),
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 99.0,
                "ratio_vs_baseline": 2.0,
            },
            "evaluation_windows": {"final": {"logloss": [0.0], "token_counts": [1]}},
            "guard_warnings": {"warning_count": object(), "warnings": ["one"]},
        },
        ok=False,
        reason="mismatch",
        tolerance=1e-9,
        verification={"strict": True},
    )
    assert item["recompute"] == {
        "family": "ppl",
        "performed": True,
        "ok": False,
        "reason": "mismatch",
    }
    assert item["warning_count"] == 1
    assert item["guard_warnings_present"] is True
    assert item["verification"] == {"strict": True}
