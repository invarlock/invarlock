from __future__ import annotations

from dataclasses import asdict

import pytest

from invarlock.core.api import RunReport as CoreRunReport
from invarlock.reporting.report_normalization import normalize_and_validate_run_report
from invarlock.reporting.report_types import create_empty_report, validate_report


def _valid_primary_metric() -> dict[str, object]:
    return {
        "kind": "ppl_causal",
        "final": 10.0,
        "preview": 10.0,
        "ratio_vs_baseline": 1.0,
    }


def test_core_lifecycle_guard_mapping_cannot_normalize_away_a_failure() -> None:
    core_report = CoreRunReport(
        meta={"model_id": "m", "seed": 7},
        guards={
            "spectral": {
                "passed": False,
                "decision": "block",
                "violations": ["spectral regression"],
            }
        },
        metrics={"primary_metric": _valid_primary_metric()},
        status="failed",
    )

    with pytest.raises(ValueError, match="Invalid canonical RunReport structure"):
        normalize_and_validate_run_report(asdict(core_report))


def test_canonical_guard_list_passes_through_without_losing_failure() -> None:
    report = create_empty_report()
    report["meta"].update({"model_id": "m", "seed": 7})
    report["metrics"]["primary_metric"] = _valid_primary_metric()
    failing_guard = {
        "name": "spectral",
        "passed": False,
        "decision": "block",
        "policy": {"correction_enabled": True},
        "metrics": {},
        "diagnostics": [],
        "violations": ["spectral regression"],
    }
    report["guards"] = [failing_guard]

    normalized = normalize_and_validate_run_report(report)

    assert normalized["guards"] == [failing_guard]
    assert normalized["guards"][0]["passed"] is False
    assert validate_report(normalized) is True


def test_canonical_policy_receipt_survives_ingress_without_reconstruction() -> None:
    report = create_empty_report()
    report["metrics"]["primary_metric"] = _valid_primary_metric()
    report["resolved_policy"] = {"spectral": {"max_caps": 2}}
    report["policy_resolution"] = {
        "format_version": "invarlock.runtime-policy-receipt.v1",
        "source": "runtime",
    }

    normalized = normalize_and_validate_run_report(report)

    assert normalized["resolved_policy"] == report["resolved_policy"]
    assert normalized["policy_resolution"] == report["policy_resolution"]
    assert normalized is not report


@pytest.mark.parametrize(
    "guard_entry",
    [
        42,
        {},
        {"name": "", "passed": False},
        {"name": "spectral", "passed": "false"},
    ],
)
def test_validate_report_rejects_malformed_guard_entries(guard_entry: object) -> None:
    report = create_empty_report()
    report["metrics"]["primary_metric"] = _valid_primary_metric()
    report["guards"] = [guard_entry]  # type: ignore[list-item]

    assert validate_report(report) is False
