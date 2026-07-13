from __future__ import annotations

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _mk_pm_report(
    *, ratio: float, preview_tokens: int = 30000, final_tokens: int = 30000
) -> dict:
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "ts": "2024-01-01T00:00:00",
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "quant_rtn",
            "plan_digest": "deadbeef",
            "deltas": {
                "params_changed": 1,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 1,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 11.01,
                "ratio_vs_baseline": ratio,
            },
            "bootstrap": {
                "replicates": 10,
                "alpha": 0.05,
                "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
            },
            "preview_total_tokens": preview_tokens,
            "final_total_tokens": final_tokens,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_evaluation_report_policy_digest_changed_and_hysteresis_applied() -> None:
    # Subject is balanced; baseline conservative; ratio slightly above base limit but within hysteresis
    rep = _mk_pm_report(ratio=1.101)
    base = {**rep, "edit": {"name": "noop"}}
    base["meta"] = {**rep["meta"], "auto": {"tier": "conservative"}}
    base["metrics"] = {
        **rep["metrics"],
        "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0},
    }
    cert = make_report(rep, base)
    pd = cert.get("policy_digest", {})
    assert isinstance(pd, dict) and pd.get("changed") is True
    # Hysteresis fields present
    hyst = pd.get("hysteresis", {})
    assert isinstance(hyst, dict) and "ppl" in hyst
    # Validation flags should mark hysteresis applied and PM acceptable
    val = cert.get("validation", {})
    assert val.get("primary_metric_acceptable") is True
    assert val.get("hysteresis_applied") is True


def test_evaluation_report_guard_metric_impact_not_evaluated_fails_closed() -> None:
    rep = _mk_pm_report(ratio=1.0)
    # Provide guard_metric_impact payload without bare/guarded metrics → not evaluated branch
    rep["guard_metric_impact"] = {"source": "unit"}
    base = {**rep, "edit": {"name": "noop"}}
    cert = make_report(rep, base)
    go = cert.get("guard_metric_impact", {})
    assert go.get("evaluated") is False
    assert go.get("passed") is False
    assert any(
        "unavailable" in item.get("message", "").lower()
        for item in (go.get("diagnostics") or [])
    )
    assert "errors" not in go


def test_evaluation_report_guard_metric_impact_from_guard_degradation() -> None:
    # Build guard context with bare/guarded reports so guard_metric_impact can be computed
    bare = _mk_pm_report(ratio=1.0)
    guarded = _mk_pm_report(ratio=1.0)
    # Change final to introduce a small degradation.
    guarded["metrics"]["primary_metric"]["final"] = 10.1
    bare["metrics"]["primary_metric"]["final"] = 10.0
    # Provide windows so PM resolver can compute display-space points for guard metric impact
    bare.setdefault("evaluation_windows", {})["final"] = {
        "window_ids": [1],
        "logloss": [2.30],
        "token_counts": [100],
    }
    guarded.setdefault("evaluation_windows", {})["final"] = {
        "window_ids": [1],
        "logloss": [2.305],
        "token_counts": [100],
    }
    rep = _mk_pm_report(ratio=1.0)
    rep["guard_metric_impact"] = {"bare_report": bare, "guarded_report": guarded}
    base = {**rep, "edit": {"name": "noop"}}
    cert = make_report(rep, base)
    qo = cert.get("guard_metric_impact", {})
    assert qo.get("degradation_basis") == "relative_increase"
    assert isinstance(qo.get("degradation"), float)
