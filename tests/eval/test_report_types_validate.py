import pytest

from invarlock.reporting.report_types import create_empty_report, validate_report


def _make_valid_report():
    report = create_empty_report()
    report["meta"]["model_id"] = "m"
    report["data"]["dataset"] = "d"
    report["data"]["split"] = "val"
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "final": 10.0,
        "preview": 10.0,
        "ratio_vs_baseline": 1.0,
        "display_ci": (10.0, 10.0),
    }
    return report


def test_validate_report_true_minimal():
    r = _make_valid_report()
    assert validate_report(r) is True


def test_validate_report_true_when_primary_metric_final_is_none():
    r = _make_valid_report()
    r["metrics"]["primary_metric"]["final"] = None
    assert validate_report(r) is True


def test_validate_report_missing_top_level():
    r = _make_valid_report()
    r.pop("flags")
    assert validate_report(r) is False


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda report: 42, False),
        (lambda report: report.__setitem__("guards", {"not": "a list"}), False),
        (lambda report: report.__setitem__("metrics", []), False),
        (
            lambda report: report["metrics"].__setitem__("primary_metric", []),
            False,
        ),
        (
            lambda report: report["metrics"].__setitem__("primary_metric", {}),
            False,
        ),
        (
            lambda report: report["metrics"].__setitem__(
                "primary_metric",
                {"kind": "", "final": 10.0},
            ),
            False,
        ),
        (
            lambda report: report["metrics"].__setitem__(
                "primary_metric",
                {"kind": 123, "final": 10.0},
            ),
            False,
        ),
        (
            lambda report: report["metrics"].__setitem__(
                "primary_metric",
                {"kind": "ppl_causal", "final": "not-a-number"},
            ),
            False,
        ),
        (lambda report: report.__setitem__("meta", []), False),
        (lambda report: report["meta"].__setitem__("seed", "not-int"), False),
    ],
)
def test_validate_report_type_checks(mutation, expected):
    report = _make_valid_report()
    mutated = mutation(report)
    if mutated is not None:
        report = mutated
    assert validate_report(report) is expected
