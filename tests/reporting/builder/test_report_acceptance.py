import math
from pathlib import Path

from invarlock.public_contracts import load_json_contract
from invarlock.reporting.report_make import REPORT_SCHEMA_VERSION, make_report
from tests.reporting.builder._support_report_acceptance import (
    mock_baseline,
    mock_report_with_windows,
)


def test_v1_required_keys_and_shapes(tmp_path: Path) -> None:
    report = mock_report_with_windows()
    baseline = mock_baseline(report)
    cert = make_report(report, baseline)

    # schema version present and correct
    assert cert.get("schema_version") == REPORT_SCHEMA_VERSION == "v1"

    # primary metric present with required shape
    pm = cert.get("primary_metric", {})
    assert isinstance(pm, dict) and pm
    # display_ci must be a 2-length array of numbers
    dci = pm.get("display_ci")
    assert isinstance(dci, list | tuple) and len(dci) == 2
    assert all(isinstance(x, int | float) for x in dci)
    # kind must be in the allow-list when contracts are available
    kinds = load_json_contract("metric_kinds.json")
    assert str(pm.get("kind", "")).lower() in set(kinds)
    # ratio_vs_baseline must be finite when baseline present
    rvb = pm.get("ratio_vs_baseline")
    assert isinstance(rvb, int | float) and math.isfinite(float(rvb))


def test_validation_keys_subset_only() -> None:
    report = mock_report_with_windows()
    baseline = mock_baseline(report)
    cert = make_report(report, baseline)
    allowed = set(load_json_contract("validation_keys.json"))
    observed = set((cert.get("validation") or {}).keys())
    # All emitted validation keys must be within the allow-list
    assert observed.issubset(allowed), (
        f"Unexpected validation keys: {sorted(observed - allowed)}"
    )


def test_no_top_level_ppl_keys() -> None:
    report = mock_report_with_windows()
    baseline = mock_baseline(report)
    cert = make_report(report, baseline)
    offenders = [
        k for k in cert.keys() if isinstance(k, str) and k.lower().startswith("ppl")
    ]
    assert not offenders, (
        f"Evaluation Report contains disallowed top-level ppl keys: {offenders}"
    )
