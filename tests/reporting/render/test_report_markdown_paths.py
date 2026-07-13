from unittest.mock import patch

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


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
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
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
    baseline = {**report, "run_id": "b2", "edit": {"name": "noop"}}
    return report, baseline


def test_render_markdown_plugins_overhead_and_rmt_variants():
    report, baseline = _base_report_and_baseline()
    with patch(
        "invarlock.core.bootstrap.compute_paired_delta_log_ci",
        return_value=(-0.1, 0.1),
    ):
        cert = make_report(report, baseline)

    # Add plugins and render
    cert["plugins"] = {
        "adapter": {"name": "ad", "version": "1", "module": "x.y", "entry_point": "z"},
        "edit": {"name": "ed", "version": "1", "module": "x.e"},
        "guards": [
            {"name": "g1", "version": "1", "module": "x.g"},
            {"name": "g2", "version": "2", "module": "x.h"},
        ],
    }
    _ = render_report_markdown(cert)

    # Empty plugins path (still validates)
    cert["plugins"] = {}
    _ = render_report_markdown(cert)

    # Guard metric impact present with percent
    cert["guard_metric_impact"] = {"display_value": 0.5, "display_limit": 1.0}
    _ = render_report_markdown(cert)

    # Guard metric impact absent path
    cert.pop("guard_metric_impact")
    _ = render_report_markdown(cert)

    # RMT with nonzero baseline outliers
    cert["rmt"] = {
        "stable": True,
        "outliers_guarded": 1,
        "outliers_bare": 2,
        "epsilon": 0.1,
    }
    _ = render_report_markdown(cert)

    # RMT with zero baseline outliers branch
    cert["rmt"] = {
        "stable": True,
        "outliers_guarded": 0,
        "outliers_bare": 0,
        "epsilon": 0.1,
    }
    _ = render_report_markdown(cert)

    # Policy provenance toggles
    cert["policy_provenance"] = {
        "tier": "balanced",
        "overrides": [],
        "policy_digest": "",
        "resolved_at": None,
    }
    _ = render_report_markdown(cert)
    cert["policy_provenance"] = {
        "tier": "balanced",
        "overrides": ["p1"],
        "policy_digest": "abcd",
    }
    markdown = render_report_markdown(cert)
    assert "abcd" in markdown
