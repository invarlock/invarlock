from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


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


def test_certificate_policy_digest_changed_and_hysteresis_applied() -> None:
    # Subject is balanced; baseline conservative; ratio slightly above base limit but within hysteresis
    rep = _mk_pm_report(ratio=1.101)
    base = {
        "meta": {"auto": {"tier": "conservative"}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    cert = make_certificate(rep, base)
    pd = cert.get("policy_digest", {})
    assert isinstance(pd, dict) and pd.get("changed") is True
    # Hysteresis fields present
    hyst = pd.get("hysteresis", {})
    assert isinstance(hyst, dict) and "ppl" in hyst
    # Validation flags should mark hysteresis applied and PM acceptable
    val = cert.get("validation", {})
    assert val.get("primary_metric_acceptable") is True
    assert val.get("hysteresis_applied") is True


def test_certificate_guard_overhead_not_evaluated_soft_pass() -> None:
    rep = _mk_pm_report(ratio=1.0)
    # Provide guard_overhead payload without bare/guarded metrics → not evaluated branch
    rep["guard_overhead"] = {"messages": ["noop"]}
    base = {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}}}
    cert = make_certificate(rep, base)
    go = cert.get("guard_overhead", {})
    assert go.get("evaluated") is False
    assert go.get("passed") is True
    assert any("unavailable" in m.lower() for m in (go.get("errors") or []))


def test_certificate_quality_overhead_from_guard_ratio() -> None:
    # Build guard context with bare/guarded reports so quality_overhead can be computed
    bare = _mk_pm_report(ratio=1.0)
    guarded = _mk_pm_report(ratio=1.0)
    # Change final to introduce a small overhead
    guarded["metrics"]["primary_metric"]["final"] = 10.1
    bare["metrics"]["primary_metric"]["final"] = 10.0
    # Provide windows so PM resolver can compute display-space points for quality overhead
    bare.setdefault("evaluation_windows", {})["final"] = {
        "logloss": [2.30],
        "token_counts": [100],
    }
    guarded.setdefault("evaluation_windows", {})["final"] = {
        "logloss": [2.305],
        "token_counts": [100],
    }
    rep = _mk_pm_report(ratio=1.0)
    rep["guard_overhead"] = {"bare_report": bare, "guarded_report": guarded}
    base = {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}}}
    cert = make_certificate(rep, base)
    qo = cert.get("quality_overhead", {})
    assert qo.get("basis") == "ratio"
    assert isinstance(qo.get("value"), float)
