from __future__ import annotations

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _base_report() -> dict:
    return {
        "run_id": "run-1",
        "meta": {
            "model_id": "demo-model",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "demo-ds",
            "split": "eval",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
            "windows": {"preview": 2, "final": 2},
        },
        "artifacts": {"events_path": "", "logs_path": "", "generated_at": ""},
        "guards": [],
        "guard_metric_impact": {},
        "edit": {
            "name": "noop",
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [10.0, 10.0],
            },
            "preview_final_slice_delta_summary": {
                "mean": 0.0,
                "degenerate": False,
            },
            "logloss_delta_ci": (0.0, 0.0),
            "bootstrap": {"replicates": 400, "coverage": {"preview": {"used": 0}}},
            "window_plan": {"profile": "dev"},
            "spectral": {"caps_applied": 0, "max_caps": 5, "summary": {}},
            "rmt": {"stable": True},
            "variance": {"enabled": False},
            "window_overlap_fraction": 0.0,
            "window_match_fraction": 1.0,
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [0.1, 0.2],
                "token_counts": [10, 12],
            },
            "final": {
                "window_ids": [1, 2],
                "logloss": [0.15, 0.25],
                "token_counts": [10, 12],
            },
        },
    }


def _base_baseline() -> dict:
    base = _base_report()
    base["run_id"] = "base-1"
    base["metrics"]["primary_metric"]["final"] = 10.0
    return base


def test_make_evaluation_report_compares_invariants_against_baseline_report() -> None:
    report = _base_report()
    baseline = _base_baseline()

    baseline_checks = {
        "embedding_vocab_sizes": {"embed": 10},
        "parameter_count": 100,
        "structure_hash": "deadbeef",
    }
    current_checks = {
        **baseline_checks,
        "embedding_vocab_sizes": {"embed": 11},
    }

    baseline["guards"] = [
        {
            "name": "invariants",
            "passed": True,
            "metrics": {
                "checks_performed": 3,
                "violations_found": 0,
                "fatal_violations": 0,
                "warning_violations": 0,
            },
            "violations": [],
            "details": {
                "baseline_checks": baseline_checks,
                "current_checks": baseline_checks,
            },
        }
    ]
    report["guards"] = [
        {
            "name": "invariants",
            "passed": False,
            "metrics": {
                "checks_performed": 3,
                "violations_found": 0,
                "fatal_violations": 0,
                "warning_violations": 0,
            },
            "violations": [],
            "details": {
                "baseline_checks": current_checks,
                "current_checks": current_checks,
            },
        }
    ]

    evaluation_report = make_report(report, baseline)
    assert evaluation_report["invariants"]["status"] == "fail"
    assert evaluation_report["validation"]["invariants_pass"] is False
    assert any(
        entry.get("type") == "tokenizer_mismatch"
        for entry in (evaluation_report["invariants"].get("failures") or [])
    )
