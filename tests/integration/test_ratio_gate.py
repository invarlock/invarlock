import pytest

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.report_types import create_empty_report


def _ratio_report(preview: float, final: float, tier: str) -> dict:
    report = create_empty_report()
    report["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
            "commit": "deadbeef",
            "device": "cpu",
            "auto": {
                "enabled": True,
                "tier": tier,
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        }
    )
    report["data"].update(
        {"dataset": "wikitext2", "split": "validation", "seq_len": 128, "stride": 128}
    )
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": preview,
        "final": final,
        "ratio_vs_baseline": final / preview if preview else 1.0,
    }
    report["edit"].update({"name": "structured"})
    return report


@pytest.mark.integration
def test_ratio_gate_respects_tier_limits():
    baseline = create_empty_report()
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 40.0,
        "final": 40.0,
        "ratio_vs_baseline": 1.0,
    }

    balanced_fail = _ratio_report(40.0, 46.0, tier="balanced")  # 1.15x
    conservative_pass = _ratio_report(40.0, 42.0, tier="conservative")  # 1.05x

    balanced_cert = make_certificate(balanced_fail, baseline)
    conservative_cert = make_certificate(conservative_pass, baseline)

    assert balanced_cert["validation"]["primary_metric_acceptable"] is False
    assert conservative_cert["validation"]["primary_metric_acceptable"] is True
