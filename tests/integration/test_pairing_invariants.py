import pytest

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.report_types import create_empty_report


def _build_report(match: float = 1.0, overlap: float = 0.0):
    report = create_empty_report()
    report["meta"]["model_id"] = "dummy-model"
    report["meta"]["adapter"] = "dummy"
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
        "ratio_vs_baseline": 1.0,
    }
    report["metrics"]["window_match_fraction"] = match
    report["metrics"]["window_overlap_fraction"] = overlap
    report["metrics"]["masked_tokens_total"] = 4
    report["metrics"]["masked_tokens_preview"] = 2
    report["metrics"]["masked_tokens_final"] = 2
    report["data"]["dataset"] = "synthetic"
    report["data"]["split"] = "validation"
    report["data"]["seq_len"] = 16
    report["data"]["stride"] = 8
    report["data"]["preview_n"] = 2
    report["data"]["final_n"] = 2
    report["evaluation_windows"] = {
        "preview": {
            "window_ids": [0, 1],
            "logloss": [3.0, 3.1],
            "input_ids": [[1, 2], [3, 4]],
            "attention_masks": [[1, 1], [1, 1]],
        },
        "final": {
            "window_ids": [2, 3],
            "logloss": [3.0, 3.1],
            "input_ids": [[5, 6], [7, 8]],
            "attention_masks": [[1, 1], [1, 1]],
        },
    }
    return report


@pytest.mark.integration
def test_pairing_invariants_ci_profile():
    baseline_report = _build_report()
    run_report = _build_report()

    certificate = make_certificate(run_report, baseline_report)
    stats = certificate.get("dataset", {}).get("windows", {}).get("stats", {})

    assert stats["window_match_fraction"] == pytest.approx(1.0)
    assert stats["window_overlap_fraction"] == pytest.approx(0.0)
