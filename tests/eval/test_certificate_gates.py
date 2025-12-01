from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.report_types import create_empty_report


def _make_report(preview: float, final: float, tier: str = "balanced"):
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
                "target_ppl_ratio": None,
            },
        }
    )
    report["data"].update(
        {"dataset": "wikitext2", "split": "validation", "seq_len": 128, "stride": 64}
    )
    report["edit"].update({"name": "structured"})
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": preview,
        "final": final,
        "ratio_vs_baseline": final / preview if preview > 0 else 1.0,
    }
    return report


def _make_baseline(ppl: float):
    base = create_empty_report()
    base["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
            "commit": "cafebabe",
            "device": "cpu",
        }
    )
    base["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": ppl,
        "final": ppl,
        "ratio_vs_baseline": 1.0,
    }
    return base


def test_certificate_enforces_tighter_ppl_ratio_gate():
    report = _make_report(preview=40.0, final=44.0, tier="balanced")  # 1.10 ratio
    baseline = _make_baseline(ppl=40.0)

    cert = make_certificate(report, baseline)
    assert cert["validation"]["primary_metric_acceptable"] is True

    failing_report = _make_report(
        preview=40.0, final=46.0, tier="balanced"
    )  # 1.15 ratio
    failing_cert = make_certificate(failing_report, baseline)
    assert failing_cert["validation"]["primary_metric_acceptable"] is False


def test_certificate_uses_tier_specific_thresholds():
    balanced_report = _make_report(preview=40.0, final=46.0, tier="balanced")  # 1.15
    conservative_report = _make_report(
        preview=40.0, final=42.0, tier="conservative"
    )  # 1.05
    aggressive_report = _make_report(
        preview=40.0, final=48.0, tier="aggressive"
    )  # 1.20

    baseline = _make_baseline(ppl=40.0)

    balanced_cert = make_certificate(balanced_report, baseline)
    conservative_cert = make_certificate(conservative_report, baseline)
    aggressive_cert = make_certificate(aggressive_report, baseline)

    assert balanced_cert["validation"]["primary_metric_acceptable"] is False
    assert conservative_cert["validation"]["primary_metric_acceptable"] is True
    assert aggressive_cert["validation"]["primary_metric_acceptable"] is True
