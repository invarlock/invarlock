from __future__ import annotations

from invarlock.reporting.certificate import (
    compute_console_validation_block,
    make_certificate,
)


def _mk_pm_report(*, ratio: float, pm_final: float = 10.0) -> dict:
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
                "preview": pm_final,
                "final": pm_final * ratio,
                "ratio_vs_baseline": ratio,
            },
            "bootstrap": {
                "replicates": 10,
                "alpha": 0.05,
                "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
            },
            "preview_total_tokens": 1000,
            "final_total_tokens": 1000,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_guard_overhead_structured_pass_and_fail() -> None:
    # Prepare bare/guarded reports to exercise structured validate_guard_overhead path
    bare = _mk_pm_report(ratio=1.0, pm_final=10.0)
    guarded = _mk_pm_report(ratio=1.01, pm_final=10.0)  # 1% overhead

    # PASS: threshold 2% > ratio-1%
    rep_pass = _mk_pm_report(ratio=1.0, pm_final=10.0)
    rep_pass["guard_overhead"] = {
        "overhead_threshold": 0.02,
        "bare_report": bare,
        "guarded_report": guarded,
    }
    base = {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}}}
    cert_pass = make_certificate(rep_pass, base)
    go_pass = cert_pass.get("guard_overhead", {})
    assert go_pass.get("evaluated") is True
    assert go_pass.get("passed") is True
    block_pass = compute_console_validation_block(cert_pass)
    labels = [r["label"] for r in block_pass["rows"]]
    assert "Guard Overhead Acceptable" in labels
    # Guard row status should be PASS
    row = next(
        r for r in block_pass["rows"] if r["label"] == "Guard Overhead Acceptable"
    )
    assert row["ok"] is True and row["status"].startswith("✅")

    # FAIL: threshold 0.5% < ratio-1%
    rep_fail = _mk_pm_report(ratio=1.0, pm_final=10.0)
    rep_fail["guard_overhead"] = {
        "overhead_threshold": 0.005,
        "bare_report": bare,
        "guarded_report": guarded,
    }
    cert_fail = make_certificate(rep_fail, base)
    go_fail = cert_fail.get("guard_overhead", {})
    assert go_fail.get("evaluated") is True
    assert go_fail.get("passed") is False
    block_fail = compute_console_validation_block(cert_fail)
    # Guard row status should be FAIL when evaluated
    row2 = next(
        r for r in block_fail["rows"] if r["label"] == "Guard Overhead Acceptable"
    )
    assert row2["ok"] is False and row2["status"].startswith("❌")
