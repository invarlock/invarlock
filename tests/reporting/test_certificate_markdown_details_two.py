from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _mk_base() -> tuple[dict, dict]:
    report = {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            }
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "logloss": [2.302585093],
                "token_counts": [1],
            },
            "final": {"window_ids": [2], "logloss": [2.302585093], "token_counts": [1]},
        },
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    return report, baseline


def test_markdown_policy_provenance_and_resolved_policy_blocks() -> None:
    rep, base = _mk_base()
    cert = make_certificate(rep, base)
    cert.setdefault("policy_provenance", {})
    cert["policy_provenance"].update(
        {
            "tier": "balanced",
            "overrides": ["metrics.pm_ratio.hysteresis_ratio=0.002"],
            "policy_digest": cert.get("policy_digest", {}).get("thresholds_hash")
            or "deadbeefcafe0123",
            "resolved_at": cert.get("artifacts", {}).get("generated_at"),
        }
    )
    cert["resolved_policy"] = {"metrics": {"pm_ratio": {"min_tokens": 50000}}}

    md = render_certificate_markdown(cert)
    assert "Policy Configuration" in md and "Overrides:" in md and "Digest:" in md
    assert "Resolved Policy YAML" in md and "```yaml" in md
