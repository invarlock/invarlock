from __future__ import annotations

import invarlock.reporting.report_builder as rb


def test_report_builder_wrapper_delegation(monkeypatch):
    monkeypatch.setattr(
        rb,
        "_prepare_guard_overhead_section_impl",
        lambda raw, validate_guard_overhead_fn: ({"evaluated": True}, True),
    )
    payload, passed = rb._prepare_guard_overhead_section({"k": "v"})
    assert payload["evaluated"] is True
    assert passed is True

    monkeypatch.setattr(
        rb,
        "_compute_quality_overhead_from_guard_impl",
        lambda raw_guard, pm_kind_hint, compute_primary_metric_from_report_fn, get_metric_fn: {
            "basis": "ratio",
            "value": 1.01,
        },
    )
    assert rb._compute_quality_overhead_from_guard(
        {"guarded_report": {}}, "ppl_causal"
    ) == {
        "basis": "ratio",
        "value": 1.01,
    }

    monkeypatch.setattr(
        rb,
        "_build_provenance_block_impl",
        lambda *args, **kwargs: {"policy": {"source": "test"}},
    )
    prov = rb._build_provenance_block(
        {},
        {},
        {"run_id": "base"},
        {"report_path": "runs/report.json"},
        {"source": "test"},
        "abc",
        {},
        "run123",
    )
    assert prov["policy"]["source"] == "test"

    monkeypatch.setattr(
        rb,
        "_resolve_pm_acceptance_range_from_report_impl",
        lambda report: {"min": 0.95, "max": 1.10},
    )
    assert rb._resolve_pm_acceptance_range_from_report({}) == {"min": 0.95, "max": 1.10}

    monkeypatch.setattr(
        rb,
        "_resolve_pm_drift_band_from_report_impl",
        lambda report, drift_band_default: {
            "min": drift_band_default[0],
            "max": drift_band_default[1],
        },
    )
    assert rb._resolve_pm_drift_band_from_report({}) == {
        "min": rb.PM_DRIFT_BAND_DEFAULT[0],
        "max": rb.PM_DRIFT_BAND_DEFAULT[1],
    }

    monkeypatch.setattr(rb, "_resolve_tiny_relax_from_report_impl", lambda report: True)
    assert rb._resolve_tiny_relax_from_report({}) is True

    seen: dict[str, object] = {}

    def _flags_stub(*args, **kwargs):
        seen["pm_drift_band_default"] = kwargs.get("pm_drift_band_default")
        seen["get_tier_policies_fn"] = kwargs.get("get_tier_policies_fn")
        return {"primary_metric_acceptable": True}

    monkeypatch.setattr(rb, "_compute_validation_flags_impl", _flags_stub)
    flags = rb._compute_validation_flags(
        ppl={},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={},
        target_ratio=1.0,
        guard_overhead={},
        primary_metric={},
        moe={},
        dataset_capacity={},
        pm_acceptance_range={"min": 0.95, "max": 1.1},
        pm_drift_band={"min": 0.95, "max": 1.05},
        pm_tail={},
        tiny_relax=False,
    )
    assert flags["primary_metric_acceptable"] is True
    assert seen["pm_drift_band_default"] == rb.PM_DRIFT_BAND_DEFAULT
    assert callable(seen["get_tier_policies_fn"])
