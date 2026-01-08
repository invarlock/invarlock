import math
from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.report_types import create_empty_report


def _mk_reports_with_tail_policy(*, mode: str) -> tuple[dict, dict]:
    baseline_ll = [1.0, 1.0, 1.0, 1.0]
    subject_ll = [1.0, 1.1, 1.1, 1.1]
    window_ids = [1, 2, 3, 4]
    token_counts = [10, 10, 10, 10]

    base = create_empty_report()
    base["meta"].update(
        {"model_id": "m", "adapter": "a", "commit": "cafebabe", "device": "cpu"}
    )
    base["metrics"]["primary_metric"] = {"kind": "ppl_causal", "final": math.exp(1.0)}
    base["evaluation_windows"] = {
        "final": {
            "window_ids": window_ids,
            "logloss": baseline_ll,
            "token_counts": token_counts,
        }
    }

    subj = create_empty_report()
    subj["meta"].update(
        {
            "model_id": "m",
            "adapter": "a",
            "commit": "deadbeef",
            "device": "cpu",
            "auto": {
                "enabled": True,
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
            "config": {
                "guards": {
                    "metrics": {
                        "pm_tail": {
                            "mode": mode,
                            "min_windows": 4,
                            "quantile": 0.95,
                            "quantile_max": 0.05,
                            "epsilon": 0.05,
                            "mass_max": 0.25,
                        }
                    }
                }
            },
        }
    )
    subj["data"].update(
        {"dataset": "dummy", "split": "validation", "seq_len": 8, "stride": 8}
    )
    ppl_sub = math.exp(sum(subject_ll) / len(subject_ll))
    subj["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": ppl_sub,
        "final": ppl_sub,
        "ratio_vs_baseline": ppl_sub / math.exp(1.0),
        "display_ci": [ppl_sub / math.exp(1.0), ppl_sub / math.exp(1.0)],
    }
    subj["evaluation_windows"] = {
        "final": {
            "window_ids": window_ids,
            "logloss": subject_ll,
            "token_counts": token_counts,
        }
    }
    return subj, base


def test_tail_gate_warn_does_not_fail_validation():
    report, baseline = _mk_reports_with_tail_policy(mode="warn")

    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)

    assert cert["validation"]["primary_metric_tail_acceptable"] is True
    pm_tail = cert.get("primary_metric_tail", {})
    assert pm_tail.get("mode") == "warn"
    assert pm_tail.get("evaluated") is True
    assert pm_tail.get("passed") is False
    assert pm_tail.get("warned") is True
    assert pm_tail.get("violations")


def test_tail_gate_fail_sets_validation_false():
    report, baseline = _mk_reports_with_tail_policy(mode="fail")

    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)

    assert cert["validation"]["primary_metric_tail_acceptable"] is False
    pm_tail = cert.get("primary_metric_tail", {})
    assert pm_tail.get("mode") == "fail"
    assert pm_tail.get("evaluated") is True
    assert pm_tail.get("passed") is False
    assert pm_tail.get("violations")
