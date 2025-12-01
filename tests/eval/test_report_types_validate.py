from invarlock.reporting.report_types import create_empty_report, validate_report


def test_validate_report_true_minimal():
    r = create_empty_report()
    # Fill minimal required values to satisfy checks (PM-only)
    r["meta"]["model_id"] = "m"
    r["data"]["dataset"] = "d"
    r["data"]["split"] = "val"
    r["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "final": 10.0,
        "preview": 10.0,
        "ratio_vs_baseline": 1.0,
        "display_ci": (10.0, 10.0),
    }
    assert validate_report(r) is True


def test_validate_report_missing_top_level():
    r = create_empty_report()
    r.pop("flags")
    assert validate_report(r) is False


def test_validate_report_type_checks():
    r = create_empty_report()
    r["metrics"]["primary_metric"] = {"kind": "ppl_causal", "final": "not-a-number"}
    assert validate_report(r) is False
    r = create_empty_report()
    r["meta"]["seed"] = "not-int"
    assert validate_report(r) is False
    r = create_empty_report()
    r["guards"] = {"not": "a list"}  # wrong type
    assert validate_report(r) is False
