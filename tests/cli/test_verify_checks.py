import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from invarlock.cli.commands import verify as v


def _cert_base(pm: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": {"model_id": "m", "adapter": "hf", "seed": 1, "device": "cpu"},
        "primary_metric": pm,
        "dataset": {
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "paired_windows": 2,
                },
            }
        },
        "baseline_ref": {
            "primary_metric": {"kind": pm.get("kind", ""), "final": 100.0}
        },
    }


def test_coerce_float_rejects_non_numeric() -> None:
    assert v._coerce_float(object()) is None
    assert v._coerce_float("1.25") == 1.25
    assert v._coerce_float("nan") is None


def test_validate_primary_metric_ppl_ratio_ok_and_mismatch():
    pm = {"kind": "ppl_causal", "final": 110.0, "ratio_vs_baseline": 1.1}
    cert = _cert_base(pm)
    assert v._validate_primary_metric(cert) == []

    cert_bad = _cert_base(
        {"kind": "ppl_causal", "final": 110.0, "ratio_vs_baseline": 1.2}
    )
    errs = v._validate_primary_metric(cert_bad)
    assert any("Primary metric ratio mismatch" in e for e in errs)


def test_validate_primary_metric_ppl_baseline_final_non_numeric_is_ignored() -> None:
    cert = _cert_base({"kind": "ppl_causal", "final": 110.0, "ratio_vs_baseline": 1.1})
    cert["baseline_ref"]["primary_metric"]["final"] = "100"
    assert v._validate_primary_metric(cert) == []


def test_validate_primary_metric_non_ppl_requires_ratio():
    cert = _cert_base({"kind": "accuracy", "final": 0.9})
    errs = v._validate_primary_metric(cert)
    assert any("missing primary_metric.ratio_vs_baseline" in e for e in errs)


def test_pairing_and_counts_checks():
    cert = _cert_base({"kind": "ppl_causal", "final": 100.0, "ratio_vs_baseline": 1.0})
    assert v._validate_pairing(cert) == []
    assert v._validate_counts(cert) == []

    # Mismatches
    cert_bad = _cert_base(
        {"kind": "ppl_causal", "final": 100.0, "ratio_vs_baseline": 1.0}
    )
    cert_bad["dataset"]["windows"]["stats"]["coverage"]["preview"]["used"] = 1
    errs = v._validate_counts(cert_bad)
    assert any("Preview window count mismatch" in e for e in errs)


def test_drift_band_and_tokenizer_hash():
    cert = _cert_base(
        {
            "kind": "ppl_causal",
            "preview": 100.0,
            "final": 101.0,
            "ratio_vs_baseline": 1.0,
        }
    )
    assert v._validate_drift_band(cert) == []
    cert_bad = _cert_base(
        {
            "kind": "ppl_causal",
            "preview": 100.0,
            "final": 120.0,
            "ratio_vs_baseline": 1.0,
        }
    )
    errs = v._validate_drift_band(cert_bad)
    assert any("drift ratio out of band" in e for e in errs)

    # Tokenizer hash mismatch only when both present
    cert_hash = _cert_base(
        {"kind": "ppl_causal", "final": 100.0, "ratio_vs_baseline": 1.0}
    )
    cert_hash["meta"]["tokenizer_hash"] = "aaa"
    cert_hash["baseline_ref"]["tokenizer_hash"] = "bbb"
    errs = v._validate_tokenizer_hash(cert_hash)
    assert any("Tokenizer hash mismatch" in e for e in errs)


def test_profile_lints_helpers(tmp_path: Path):
    cert = _cert_base({"kind": "ppl_causal", "final": 100.0, "ratio_vs_baseline": 1.0})
    cert["meta"]["model_profile"] = {
        "cert_lints": [
            {
                "type": "equals",
                "path": "meta.device",
                "value": "cpu",
                "message": "Device must be CPU",
            },
            {
                "type": "gte",
                "path": "dataset.windows.preview",
                "value": 1,
                "message": "Preview windows too few",
            },
            {
                "type": "lte",
                "path": "dataset.windows.final",
                "value": 10,
                "message": "Final windows too many",
            },
        ]
    }
    assert v._apply_profile_lints(cert) == []

    cert_bad = _cert_base(
        {"kind": "ppl_causal", "final": 100.0, "ratio_vs_baseline": 1.0}
    )
    cert_bad["meta"]["model_profile"] = {
        "cert_lints": [
            {
                "type": "equals",
                "path": "meta.adapter",
                "value": "other",
                "message": "Adapter must match",
            },
            {
                "type": "gte",
                "path": "dataset.windows.preview",
                "value": 10,
                "message": "Too few",
            },
            {
                "type": "lte",
                "path": "dataset.windows.final",
                "value": 1,
                "message": "Too many",
            },
        ]
    }
    errs = v._apply_profile_lints(cert_bad)
    assert len(errs) == 3
    # Non-numeric comparison path
    cert_bad2 = _cert_base(
        {"kind": "ppl_causal", "final": 100.0, "ratio_vs_baseline": 1.0}
    )
    cert_bad2["meta"]["model_profile"] = {
        "cert_lints": [
            {
                "type": "gte",
                "path": "dataset.windows.preview",
                "value": "x",
                "message": "Bad numeric",
            },
        ]
    }
    errs2 = v._apply_profile_lints(cert_bad2)
    assert any("numeric" in e.lower() for e in errs2)


def test_resolve_path_and_warn_adapter_family(tmp_path: Path):
    cert = _cert_base({"kind": "ppl_causal", "final": 100.0, "ratio_vs_baseline": 1.0})
    # Resolve path utility
    assert v._resolve_path(cert, "dataset.windows.preview") == 2

    # Write a baseline report file with adapter provenance
    baseline = {
        "meta": {
            "plugins": {
                "adapter": {
                    "provenance": {
                        "family": "hf",
                        "library": "transformers",
                        "version": "0.0",
                    }
                }
            }
        }
    }
    with NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(baseline, fh)
        fh.flush()
        cert.setdefault("provenance", {})
        cert["provenance"]["baseline"] = {"report_path": fh.name}

    # Should not raise; soft warning path covered
    v._warn_adapter_family_mismatch(tmp_path / "cert.json", cert)


def test_validate_certificate_payload_success_and_fail(tmp_path: Path, monkeypatch):
    # Monkeypatch schema validator to succeed so we can cover our checks
    monkeypatch.setattr(v, "validate_certificate", lambda c: True)

    cert_ok = _cert_base(
        {
            "kind": "ppl_causal",
            "preview": 100.0,
            "final": 101.0,
            "ratio_vs_baseline": 1.01,
        }
    )
    p_ok = tmp_path / "c_ok.json"
    p_ok.write_text(json.dumps(cert_ok))
    assert v._validate_certificate_payload(p_ok) == []

    cert_bad = _cert_base({"kind": "ppl_causal", "final": 101.0})  # missing ratio
    p_bad = tmp_path / "c_bad.json"
    p_bad.write_text(json.dumps(cert_bad))
    errs = v._validate_certificate_payload(p_bad)
    assert errs and any("missing" in e.lower() or "mismatch" in e.lower() for e in errs)
