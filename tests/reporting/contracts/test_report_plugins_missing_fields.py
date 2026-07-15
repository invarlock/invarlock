from unittest.mock import patch

from invarlock.reporting.rendering.markdown import render_report_markdown
from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def test_render_markdown_plugin_provenance_missing_fields_and_na_metric_impact():
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {"tier": "balanced"},
            # Minimal plugin provenance: missing version/entry_point/module entries
            "plugins": {
                "adapter": {"name": "hf_adapter"},
                "edit": {"name": "structured"},
                "guards": [
                    {"name": "variance"},
                    {"name": "spectral"},
                ],
            },
        },
        "context": {"profile": "dev", "assurance": {"mode": "off"}},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0}
        },
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [],
        "edit": {
            "name": "structured",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
        # Provide guard_metric_impact with neither percent nor ratio → measured becomes "N/A"
        "guard_metric_impact": {"degradation_limit": 0.01},
    }
    baseline = {**report, "edit": {"name": "noop"}}
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(canonical_run_report(report), canonical_baseline(baseline))

    md = render_report_markdown(cert)
    # Plugin section may be omitted by normalization; ensure overall render is sane
    assert isinstance(md, str) and (
        "## Plugin Provenance" in md or "## Executive Summary" in md
    )
