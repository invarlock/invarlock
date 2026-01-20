from __future__ import annotations

import re

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _mk_min_report(tier: str = "balanced") -> dict:
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "commit": "deadbeef",
            "device": "cpu",
            "seed": 42,
            "ts": "2024-01-01T00:00:00",
            "auto": {"tier": tier, "probes_used": 0, "target_pm_ratio": None},
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "structured",
            "plan_digest": "abc123",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 49.0,
                "ratio_vs_baseline": 0.98,
                "display_ci": (0.97, 0.99),
            },
            "preview_total_tokens": 30000,
            "final_total_tokens": 30000,
            "spectral": {"caps_applied": 0},
            "rmt": {"stable": True},
            "invariants": {"status": "pass", "summary": {}},
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_certificate_includes_policy_digest_fields():
    report = _mk_min_report(tier="balanced")
    baseline = _mk_min_report(tier="balanced")
    cert = make_certificate(report, baseline)

    assert "policy_digest" in cert, "certificate should include policy_digest"
    pd = cert["policy_digest"]
    assert isinstance(pd.get("policy_version"), str) and pd[
        "policy_version"
    ].startswith("policy-")
    assert pd.get("tier_policy_name") == "balanced"
    th = pd.get("thresholds_hash")
    assert isinstance(th, str) and re.fullmatch(r"[0-9a-f]{8,64}", th)
    # Hysteresis must include ppl and accuracy keys
    hys = pd.get("hysteresis")
    assert isinstance(hys, dict) and "ppl" in hys and "accuracy_delta_pp" in hys
    # min_effective populated (variance min_effect_lognll)
    assert isinstance(pd.get("min_effective"), float)


def test_policy_digest_changes_across_tiers_and_markdown_note():
    # Same metrics; only tier differs → thresholds hash must differ
    report_bal = _mk_min_report(tier="balanced")
    report_cons = _mk_min_report(tier="conservative")
    baseline = _mk_min_report(tier="balanced")

    cert_bal = make_certificate(report_bal, baseline)
    cert_cons = make_certificate(report_cons, baseline)

    hash_bal = cert_bal["policy_digest"]["thresholds_hash"]
    hash_cons = cert_cons["policy_digest"]["thresholds_hash"]
    assert hash_bal != hash_cons, "thresholds hash should change with tier"

    # Markdown should surface policy version + short hash and note when changed vs baseline
    md_bal = render_certificate_markdown(cert_bal)
    md_cons = render_certificate_markdown(cert_cons)
    assert "Policy Version:" in md_bal
    assert "Thresholds Digest:" in md_bal
    # Conservative vs balanced baseline should note a policy change
    assert "policy changed" in md_cons.lower()
