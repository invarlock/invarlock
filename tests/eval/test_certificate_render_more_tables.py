from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _base_minimal():
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


def test_render_spectral_tables_and_plugins_missing_fields():
    report, baseline = _base_minimal()
    with patch(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        return_value=(-0.1, 0.1),
    ):
        cert = make_certificate(report, baseline)

    # Inject spectral tables
    cert["spectral"] = {
        "caps_applied": 1,
        "max_caps": 5,
        "summary": {"status": "capped", "max_caps": 5, "caps_exceeded": False},
        "caps_applied_by_family": {"ffn": 1, "attn": 0},
        "family_caps": {"ffn": {"kappa": 2.5}, "attn": {"kappa": 3.0}},
        "family_z_quantiles": {"ffn": {"q95": 2.1, "q99": 2.5, "max": 2.6, "count": 5}},
    }

    # Plugins with missing version/entry_point should not crash rendering
    cert["plugins"] = {
        "adapter": {"name": "ad", "module": "x.y"},  # missing version/entry_point
        "edit": {"name": "ed"},  # minimal
        "guards": [
            {"name": "g1", "module": "x.g"},  # missing version
            {"name": "g2"},
        ],
    }

    # RMT families table
    cert["rmt"] = {
        "stable": True,
        "families": {
            "ffn": {"epsilon": 0.1, "bare": 2, "guarded": 1},
            "attn": {"epsilon": 0.12, "bare": 0, "guarded": 0},
        },
    }

    md = render_certificate_markdown(cert)
    assert isinstance(md, str) and "Spectral Guard" in md and "RMT Guard" in md
