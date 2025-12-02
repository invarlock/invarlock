from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _base_report_and_baseline():
    report = {
        "run_id": "r2",
        "meta": {
            "model_id": "m",
            "adapter": "a",
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
            "commit": "dead",
            "seed": 1,
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "structured",
            "plan_digest": "abcd",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (0.9, 1.1),
            }
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [1.0]}},
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    baseline = {
        "run_id": "b2",
        "model_id": "m",
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [1.0]}},
    }
    return report, baseline


def test_render_markdown_plugins_overhead_and_rmt_variants():
    report, baseline = _base_report_and_baseline()
    with patch(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        return_value=(-0.1, 0.1),
    ):
        cert = make_certificate(report, baseline)

    # Add plugins and render
    cert["plugins"] = {
        "adapter": {"name": "ad", "version": "1", "module": "x.y", "entry_point": "z"},
        "edit": {"name": "ed", "version": "1", "module": "x.e"},
        "guards": [
            {"name": "g1", "version": "1", "module": "x.g"},
            {"name": "g2", "version": "2", "module": "x.h"},
        ],
    }
    _ = render_certificate_markdown(cert)

    # Empty plugins path (still validates)
    cert["plugins"] = {}
    _ = render_certificate_markdown(cert)

    # Guard overhead present with percent
    cert["guard_overhead"] = {"overhead_percent": 0.5, "threshold_percent": 1.0}
    _ = render_certificate_markdown(cert)

    # Guard overhead absent path
    cert.pop("guard_overhead")
    _ = render_certificate_markdown(cert)

    # RMT with nonzero baseline outliers
    cert["rmt"] = {
        "stable": True,
        "outliers_guarded": 1,
        "outliers_bare": 2,
        "epsilon": 0.1,
    }
    _ = render_certificate_markdown(cert)

    # RMT with zero baseline outliers branch
    cert["rmt"] = {
        "stable": True,
        "outliers_guarded": 0,
        "outliers_bare": 0,
        "epsilon": 0.1,
    }
    _ = render_certificate_markdown(cert)

    # Policy provenance toggles
    cert["policy_provenance"] = {
        "tier": "balanced",
        "overrides": [],
        "policy_digest": "",
        "resolved_at": None,
    }
    _ = render_certificate_markdown(cert)
    cert["policy_provenance"] = {
        "tier": "balanced",
        "overrides": ["p1"],
        "policy_digest": "abcd",
    }
    _ = render_certificate_markdown(cert)
