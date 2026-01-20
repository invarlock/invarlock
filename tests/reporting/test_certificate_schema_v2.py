import math
from typing import Any

from invarlock.reporting.certificate import (
    CERTIFICATE_JSON_SCHEMA,
    CERTIFICATE_SCHEMA_VERSION,
    make_certificate,
)


def _mock_report_with_primary_metric() -> dict[str, Any]:
    # Minimal report containing evaluation windows such that ppl points are computable
    # Two windows with token_counts and logloss to produce deterministic ppl
    preview = {
        "window_ids": [1, 2],
        "logloss": [1.0, 1.1],
        "token_counts": [100, 200],
    }
    final = {
        "window_ids": [3, 4],
        "logloss": [1.05, 1.15],
        "token_counts": [100, 200],
    }

    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 42,
            "seeds": {"python": 42, "numpy": 42, "torch": 42},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": math.exp((1.0 * 100 + 1.1 * 200) / 300),
                "final": math.exp((1.05 * 100 + 1.15 * 200) / 300),
                "ratio_vs_baseline": 1.0,
            },
            "bootstrap": {"replicates": 150, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {"preview": preview, "final": final},
        "edit": {"name": "structured"},
        "artifacts": {"events_path": "", "logs_path": ""},
        "guards": [],
    }
    return report


def _mock_baseline() -> dict[str, Any]:
    # Baseline with ppl_final used for ratio; evaluation windows for pairing CI
    preview = {
        "window_ids": [1, 2],
        "logloss": [1.0, 1.1],
        "token_counts": [100, 200],
    }
    final = {
        "window_ids": [3, 4],
        "logloss": [1.0, 1.1],
        "token_counts": [100, 200],
    }
    return {
        "run_id": "baseline123",
        "model_id": "m",
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": math.exp((1.0 * 100 + 1.1 * 200) / 300),
            },
            "bootstrap": {"replicates": 150, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {"preview": preview, "final": final},
    }


def test_schema_version_bumped_and_ci_arrays_guarded():
    # schema_version is a stable constant; just ensure it's a non-empty string
    assert isinstance(CERTIFICATE_SCHEMA_VERSION, str)
    assert len(CERTIFICATE_SCHEMA_VERSION) > 0

    pm_props = CERTIFICATE_JSON_SCHEMA["properties"]["primary_metric"]["properties"]
    # ci arrays are 2-length number arrays
    for key in ("ci", "display_ci"):
        sch = pm_props[key]
        assert sch["type"] == "array"
        assert sch.get("minItems") == 2
        assert sch.get("maxItems") == 2
        assert sch.get("items", {}).get("type") == "number"


def test_system_overhead_has_pattern_properties():
    sys_over = CERTIFICATE_JSON_SCHEMA["properties"]["system_overhead"]
    assert sys_over["type"] == "object"
    # guard that latency/throughput numeric keys are constrained
    assert "patternProperties" in sys_over
    pats: dict[str, Any] = sys_over["patternProperties"]
    assert any("latency_ms_" in p for p in pats)
    assert any("throughput_" in p for p in pats)
    for spec in pats.values():
        assert spec.get("type") == "number"


def test_primary_metric_analysis_basis_present_and_consistent():
    report = _mock_report_with_primary_metric()
    baseline = _mock_baseline()
    cert = make_certificate(report, baseline)

    pm = cert.get("primary_metric", {})
    assert isinstance(pm, dict)

    # analysis fields should exist for ppl_*
    if str(pm.get("kind")).startswith("ppl"):
        # Both preview/final analysis points are present and exp() ≈ display points
        ap_prev = pm.get("analysis_point_preview")
        ap_final = pm.get("analysis_point_final")
        assert isinstance(ap_prev, float)
        assert isinstance(ap_final, float)
        assert math.isclose(
            math.exp(ap_prev), float(pm.get("preview")), rel_tol=1e-9, abs_tol=1e-9
        )
        assert math.isclose(
            math.exp(ap_final), float(pm.get("final")), rel_tol=1e-9, abs_tol=1e-9
        )


def test_ci_instability_annotation_when_low_reps():
    report = _mock_report_with_primary_metric()
    # Force low replicate count to trigger unstable flags
    report["metrics"]["bootstrap"]["replicates"] = 50
    baseline = _mock_baseline()
    baseline["metrics"]["bootstrap"]["replicates"] = 50

    cert = make_certificate(report, baseline)
    # PM-only: unstable hint should be on primary_metric
    pm = cert.get("primary_metric", {})
    assert isinstance(pm, dict)
    assert pm.get("unstable") is True
