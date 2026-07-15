from unittest.mock import patch

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _mk_cert(edit_name):
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
            "name": edit_name,
            "plan_digest": "abcd",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
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
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [1.0]}},
    }
    baseline = {**report, "run_id": "b2", "edit": {"name": "noop"}}
    with (
        patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ),
        patch(
            "invarlock.core.bootstrap.compute_paired_delta_log_ci",
            return_value=(-0.1, 0.1),
        ),
    ):
        cert = make_report(report, baseline)
    return cert


def test_render_edit_name_variants():
    for name in ("quant_rtn", "magnitude_prune", "structured", "custom_unknown"):
        cert = _mk_cert(name)
        md = render_report_markdown(cert)
        assert isinstance(md, str) and "Guard Observability" in md
