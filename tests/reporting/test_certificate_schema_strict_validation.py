from __future__ import annotations

import math

import pytest

from invarlock.reporting.certificate import (
    CERTIFICATE_JSON_SCHEMA,
    make_certificate,
    validate_certificate,
)


def _mock_report_with_windows():
    # Deterministic synthetic windows for ppl_causal
    preview = {
        "window_ids": [1, 2],
        "logloss": [1.00, 1.06],
        "token_counts": [100, 200],
    }
    final = {
        "window_ids": [3, 4],
        "logloss": [1.05, 1.15],
        "token_counts": [100, 200],
    }
    ppl_prev = math.exp((1.00 * 100 + 1.06 * 200) / 300)
    ppl_fin_subj = math.exp((1.05 * 100 + 1.15 * 200) / 300)
    report = {
        "meta": {
            "model_id": "stub",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "seed": 7,
            "seeds": {"python": 7, "numpy": 7, "torch": 7},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": ppl_prev,
                "final": ppl_fin_subj,
                "ratio_vs_baseline": 1.0,
            },
            "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {"preview": preview, "final": final},
        "edit": {"name": "structured"},
        "artifacts": {"events_path": "", "logs_path": ""},
        "guards": [],
    }
    return report


def _mock_baseline(report):
    prev = report["evaluation_windows"]["preview"]
    fin = report["evaluation_windows"]["final"]
    ppl_fin_base = math.exp((1.00 * 100 + 1.10 * 200) / 300)
    return {
        "run_id": "baseline",
        "model_id": report["meta"]["model_id"],
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": ppl_fin_base},
            "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {"preview": prev, "final": fin},
    }


@pytest.mark.unit
def test_validation_schema_rejects_unknown_keys():
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_certificate(report, baseline)
    # Inject an unknown validation key; strict schema must reject it
    cert.setdefault("validation", {})["foo_acceptable"] = True
    try:
        import jsonschema  # type: ignore

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=cert, schema=CERTIFICATE_JSON_SCHEMA)
    except Exception:
        # Fallback to helper which uses jsonschema when present
        assert validate_certificate(cert) is False


@pytest.mark.unit
def test_validation_schema_accepts_allowlisted_keys():
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_certificate(report, baseline)
    cert.setdefault("validation", {})["hysteresis_applied"] = False
    # Helper should accept allow-listed keys.
    assert validate_certificate(cert) is True
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(instance=cert, schema=CERTIFICATE_JSON_SCHEMA)
    except Exception:
        # jsonschema optional; validate_certificate already checked above
        pass
