from __future__ import annotations

import pytest

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.report_types import create_empty_report


def _base_ci_report_and_baseline():
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
            "auto": {"tier": "balanced", "probes": 0, "target_pm_ratio": 2.0},
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
            "paired_windows": 180,
            "logloss_delta_ci": (-0.1, 0.0),
            "window_plan": {"profile": "ci", "preview_n": 180, "final_n": 180},
            "preview_total_tokens": 10,
            "final_total_tokens": 10,
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
        }
    )
    report["edit"]["name"] = "quant_rtn"
    report["edit"]["plan_digest"] = "noop"
    report["edit"]["deltas"]["params_changed"] = 1

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
    return report, baseline


def test_make_certificate_ci_rejects_non_none_window_pairing_reason() -> None:
    report, baseline = _base_ci_report_and_baseline()
    report["metrics"]["window_pairing_reason"] = "no_baseline_reference"
    with pytest.raises(ValueError):
        make_certificate(report, baseline)


def test_make_certificate_ci_rejects_zero_paired_windows() -> None:
    report, baseline = _base_ci_report_and_baseline()
    report["metrics"]["paired_windows"] = 0
    report["metrics"]["window_pairing_reason"] = None
    with pytest.raises(ValueError):
        make_certificate(report, baseline)
