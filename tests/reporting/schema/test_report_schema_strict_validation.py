from __future__ import annotations

import math

import pytest

from invarlock.reporting.report_schema import (
    REPORT_JSON_SCHEMA,
    validate_report,
)
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _import_jsonschema():
    try:
        import jsonschema
    except (ImportError, ModuleNotFoundError):
        return None
    return jsonschema


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
    report = canonical_run_report(
        {
            "meta": {
                "model_id": "stub",
                "adapter": "hf_causal",
                "auto": {"tier": "balanced"},
                "device": "cpu",
                "seed": 7,
                "seeds": {"python": 7, "numpy": 7, "torch": 7},
            },
            "context": {"profile": "dev"},
            "data": {
                "dataset": "dummy",
                "split": "validation",
                "seq_len": 8,
                "stride": 4,
                "preview_n": 2,
                "final_n": 2,
            },
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": ppl_prev,
                    "final": ppl_fin_subj,
                    "ratio_vs_baseline": ppl_fin_subj
                    / math.exp((1.00 * 100 + 1.10 * 200) / 300),
                },
                "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
            },
            "evaluation_windows": {"preview": preview, "final": final},
            "edit": {"name": "structured"},
            "artifacts": {"events_path": "", "logs_path": ""},
            "guards": [],
        }
    )
    return report


def _mock_baseline(report):
    prev = report["evaluation_windows"]["preview"]
    ppl_fin_base = math.exp((1.00 * 100 + 1.10 * 200) / 300)
    return canonical_baseline(
        {
            "run_id": "baseline",
            "model_id": report["meta"]["model_id"],
            "meta": {
                "model_id": report["meta"]["model_id"],
                "adapter": "hf_causal",
                "auto": {"tier": "balanced"},
            },
            "context": {"profile": "dev"},
            "data": {
                "dataset": "dummy",
                "split": "validation",
                "seq_len": 8,
                "stride": 4,
                "preview_n": 2,
                "final_n": 2,
            },
            "edit": {"name": "noop"},
            "guards": [],
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": math.exp((1.00 * 100 + 1.06 * 200) / 300),
                    "final": ppl_fin_base,
                },
                "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
            },
            "evaluation_windows": {
                "preview": prev,
                "final": {
                    "window_ids": [3, 4],
                    "logloss": [1.00, 1.10],
                    "token_counts": [100, 200],
                },
            },
        }
    )


@pytest.mark.unit
def test_validation_schema_rejects_unknown_keys():
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_report(report, baseline)
    # Inject an unknown validation key; strict schema must reject it
    cert.setdefault("validation", {})["foo_acceptable"] = True
    assert validate_report(cert) is False

    jsonschema = _import_jsonschema()
    if jsonschema is None:
        return

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=cert, schema=REPORT_JSON_SCHEMA)


@pytest.mark.unit
def test_validation_schema_accepts_allowlisted_keys():
    report = _mock_report_with_windows()
    baseline = _mock_baseline(report)
    cert = make_report(report, baseline)
    cert.setdefault("validation", {})["hysteresis_applied"] = False
    # Helper should accept allow-listed keys.
    assert validate_report(cert) is True

    jsonschema = _import_jsonschema()
    if jsonschema is None:
        return

    jsonschema.validate(instance=cert, schema=REPORT_JSON_SCHEMA)
