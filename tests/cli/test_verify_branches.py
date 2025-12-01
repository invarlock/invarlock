from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from invarlock.cli.commands import verify as V


def _base_cert() -> dict:
    return {
        "schema_version": "v1",
        "baseline_ref": {"primary_metric": {"final": 10.0}},
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "preview": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [10.0, 10.0],
        },
        "dataset": {
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "paired_windows": 1,
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                },
            }
        },
        "meta": {},
    }


def test_validate_ratio_branches_no_logloss_block_and_bad_ci_entries_and_nonfinite():
    # Primary metric validation: baseline <= 0 and mismatch
    cert = _base_cert()
    cert["baseline_ref"]["primary_metric"]["final"] = 0.0
    errs = V._validate_primary_metric(cert)
    assert any("baseline final" in e.lower() for e in errs)

    cert = _base_cert()
    cert["primary_metric"]["ratio_vs_baseline"] = 2.0
    errs = V._validate_primary_metric(cert)
    assert any("ratio mismatch" in e.lower() for e in errs)


def test_validate_pairing_bootstrap_fallback_and_missing_metrics():
    # (bootstrap fallback path removed in PM-only; stats must be present)
    cert = _base_cert()
    stats = cert["dataset"]["windows"]["stats"]
    stats.pop("window_match_fraction", None)
    stats.pop("window_overlap_fraction", None)
    errs = V._validate_pairing(cert)
    assert any("missing window_match_fraction" in e for e in errs)
    assert any("missing window_overlap_fraction" in e for e in errs)


def test_validate_counts_branches_none_expected_and_missing_coverage():
    cert = _base_cert()
    # Missing coverage blocks -> missing used error branches
    cert["dataset"]["windows"]["stats"]["coverage"] = {}
    errs = V._validate_counts(cert)
    assert any("coverage.preview.used" in e for e in errs)
    assert any("coverage.final.used" in e for e in errs)


def test_apply_profile_lints_equals_mismatch_and_numeric_conversion_errors():
    cert = _base_cert()
    cert["meta"]["model_profile"] = {
        "cert_lints": [
            {"type": "equals", "path": "ppl.final", "value": 999.0, "message": "M"},
            {"type": "gte", "path": "ppl.final", "value": "bad", "message": "G"},
            {"type": "lte", "path": "ppl.final", "value": "bad", "message": "L"},
            {"type": "unknown", "path": "ppl.final", "value": 1.0, "message": "U"},
            "invalid_entry",
        ]
    }
    errs = V._apply_profile_lints(cert)
    assert any("Expected ppl.final ==" in e for e in errs)
    assert any("Expected numeric comparison" in e for e in errs)


def test_validate_certificate_payload_schema_fail(tmp_path: Path):
    cert_path = tmp_path / "c.json"
    cert_path.write_text(json.dumps(_base_cert()))
    with patch("invarlock.cli.commands.verify.validate_certificate", lambda c: False):
        errs = V._validate_certificate_payload(cert_path)
        assert any("schema validation failed" in e for e in errs)


def test_verify_command_multiple_files_mixed_results(tmp_path: Path, capsys):
    good = _base_cert()
    bad = _base_cert()
    bad["primary_metric"]["ratio_vs_baseline"] = 2.0

    p1 = tmp_path / "good.json"
    p2 = tmp_path / "bad.json"
    p1.write_text(json.dumps(good))
    p2.write_text(json.dumps(bad))

    with patch("invarlock.cli.commands.verify.validate_certificate", lambda c: True):
        errs1 = V._validate_certificate_payload(p1)
        errs2 = V._validate_certificate_payload(p2)
        assert errs1 == [] and errs2 != []
