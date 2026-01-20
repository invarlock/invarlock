from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def _mk_report_with_tokens(
    ratio: float, preview_tokens=30000, final_tokens=30000
) -> dict:
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 42,
            "ts": "now",
            "auto": {"tier": "balanced"},
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "noop",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0 * ratio,
                "ratio_vs_baseline": ratio,
                "display_ci": (ratio, ratio),
            },
            "preview_total_tokens": preview_tokens,
            "final_total_tokens": final_tokens,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_ppl_hysteresis_applied_near_threshold():
    # Balanced tier ratio limit=1.10; within +0.002 hysteresis → acceptable
    report = _mk_report_with_tokens(ratio=1.101)
    baseline = _mk_report_with_tokens(ratio=1.0)
    cert = make_certificate(report, baseline)
    assert cert["validation"]["primary_metric_acceptable"] is True
    assert cert["validation"].get("hysteresis_applied") is True


def test_ppl_min_tokens_floor_blocks_when_insufficient():
    # Tokens below 50k floor (balanced policy) → gate fails
    report = _mk_report_with_tokens(
        ratio=1.00, preview_tokens=10000, final_tokens=10000
    )
    baseline = _mk_report_with_tokens(ratio=1.0)
    cert = make_certificate(report, baseline)
    assert cert["validation"]["primary_metric_acceptable"] is False


def test_accuracy_hysteresis_and_min_examples():
    # Primary metric accuracy delta near threshold should pass with hysteresis
    report = _mk_report_with_tokens(ratio=1.0)
    baseline = _mk_report_with_tokens(ratio=1.0)
    report.setdefault("metrics", {})["primary_metric"] = {
        "kind": "accuracy",
        "preview": 0.80,
        "final": 0.85,
        # Slightly below threshold (-1.0) but within hysteresis (0.1)
        "ratio_vs_baseline": -1.05,
        "n_final": 250,
    }
    cert = make_certificate(report, baseline)
    assert cert["validation"]["primary_metric_acceptable"] is True
    assert cert["validation"].get("hysteresis_applied") is True

    # Now below min examples
    report2 = _mk_report_with_tokens(ratio=1.0)
    report2.setdefault("metrics", {})["primary_metric"] = {
        "kind": "accuracy",
        "preview": 0.80,
        "final": 0.85,
        "ratio_vs_baseline": -0.5,
        "n_final": 100,  # below 200 floor
    }
    cert2 = make_certificate(report2, baseline)
    assert cert2["validation"]["primary_metric_acceptable"] is False
