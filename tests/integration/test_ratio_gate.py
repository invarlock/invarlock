import pytest

from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import create_empty_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def _ratio_report(
    preview: float,
    final: float,
    tier: str,
    *,
    edit_name: str = "structured",
    target_pm_ratio: float | None = None,
) -> dict:
    report = create_empty_report()
    report["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "commit": "deadbeef",
            "device": "cpu",
            "auto": {
                "enabled": True,
                "tier": tier,
                "probes_used": 0,
                "target_pm_ratio": target_pm_ratio,
            },
        }
    )
    report["context"] = {"profile": "dev"}
    report["data"].update(
        {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 128,
            "stride": 128,
            "preview_n": 1,
            "final_n": 1,
        }
    )
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": preview,
        "final": final,
        "ratio_vs_baseline": final / preview if preview else 1.0,
    }
    report["edit"].update({"name": edit_name})
    return report


@pytest.mark.integration
def test_ratio_gate_respects_tier_limits():
    baseline = canonical_baseline(
        _ratio_report(40.0, 40.0, tier="balanced", edit_name="noop")
    )

    balanced_fail = canonical_run_report(
        _ratio_report(40.0, 46.0, tier="balanced")
    )  # 1.15x
    conservative_pass = canonical_run_report(
        _ratio_report(40.0, 42.0, tier="conservative")
    )  # 1.05x

    balanced_cert = make_report(balanced_fail, baseline)
    conservative_cert = make_report(conservative_pass, baseline)

    assert balanced_cert["validation"]["primary_metric_acceptable"] is False
    assert conservative_cert["validation"]["primary_metric_acceptable"] is True


@pytest.mark.integration
def test_ratio_gate_ignores_auto_target_pm_ratio_hint():
    baseline = canonical_baseline(
        _ratio_report(40.0, 40.0, tier="balanced", edit_name="noop")
    )

    subject = canonical_run_report(
        _ratio_report(40.0, 41.2, tier="balanced", target_pm_ratio=1.0)
    )  # 1.03x

    cert = make_report(subject, baseline)

    assert cert["validation"]["primary_metric_acceptable"] is True
