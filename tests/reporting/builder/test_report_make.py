from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_schema import validate_report
from invarlock.reporting.report_types import create_empty_report


def test_make_evaluation_report_with_secondary_and_subgroups():
    report = create_empty_report()
    report["meta"].update(
        {
            "model_id": "model-x",
            "adapter": "hf_causal",
            "commit": "abc",
            "device": "cpu",
            "ts": "now",
            "seed": 1,
            "seeds": {"python": 1, "numpy": 1, "torch": 1},
            "model_profile": {"family": "gpt2"},
            "tokenizer_hash": "hash",
            "env_flags": {"foo": "bar"},
        }
    )
    report["data"].update(
        {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 2,
            "stride": 2,
            "preview_n": 180,
            "final_n": 180,
        }
    )
    report["metrics"].update(
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 1.1,
                "ratio_vs_baseline": 1.0,
                "analysis_basis": "mean_logloss",
                "analysis_point_final": 0.0,
            },
            "bootstrap": {
                "replicates": 1200,
                "coverage": {
                    "preview": {"used": 180},
                    "final": {"used": 180},
                    "replicates": {"used": 1200},
                },
            },
            "paired_windows": 2,
            "logloss_delta_ci": (-0.1, 0.0),
            "window_plan": {"profile": "ci", "preview_n": 180, "final_n": 180},
            "preview_total_tokens": 10,
            "final_total_tokens": 10,
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "secondary_metrics": [
                {
                    "kind": "accuracy",
                    "preview": 0.9,
                    "final": 0.91,
                    "ratio_vs_baseline": 1.0,
                    "unit": "acc",
                    "display_ci": (0.8, 1.0),
                    "ci": (0.8, 1.0),
                }
            ],
            "classification": {
                "subgroups": {
                    "preview": {"group_counts": {}},
                    "final": {"group_counts": {}},
                }
            },
        }
    )
    report["edit"]["name"] = "quant_rtn"
    report["edit"]["plan_digest"] = "noop"
    report["edit"]["deltas"]["params_changed"] = 1
    report["guards"] = [
        {
            "name": "variance",
            "policy": {},
            "metrics": {},
            "actions": [],
            "violations": [],
        }
    ]
    report["artifacts"] = {
        "events_path": "events",
        "logs_path": "logs",
        "checkpoint_path": None,
    }
    report["flags"] = {"guard_recovered": False, "rollback_reason": None}

    baseline = create_empty_report()
    baseline["meta"].update(
        {
            "model_id": "model-x",
            "adapter": "hf_causal",
            "commit": "abc",
            "device": "cpu",
            "ts": "now",
            "seed": 1,
            "seeds": {"python": 1, "numpy": 1, "torch": 1},
        }
    )
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 1.0,
        "final": 1.0,
        "ratio_vs_baseline": 1.0,
        "analysis_basis": "mean_logloss",
        "analysis_point_final": 0.0,
    }
    baseline["flags"] = {"guard_recovered": False, "rollback_reason": None}
    baseline["artifacts"] = {
        "events_path": "events",
        "logs_path": "logs",
        "checkpoint_path": None,
    }

    evaluation_report = make_report(report, baseline)

    assert evaluation_report["primary_metric"]["ratio_vs_baseline"] is not None
    assert evaluation_report["policy_digest"]["policy_version"] == "policy-v1"
    assert evaluation_report["validation"]["guard_overhead_acceptable"] in {True, False}
    assert evaluation_report["validation"]["primary_metric_acceptable"] in {True, False}

    # Evaluation Report should validate against schema/allowlist
    assert validate_report(evaluation_report)
