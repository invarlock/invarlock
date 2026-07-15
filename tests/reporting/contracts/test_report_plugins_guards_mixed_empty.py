from unittest.mock import patch

from invarlock.reporting.rendering.markdown import (
    render_report_markdown,
)
from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def test_plugins_guards_list_mixed_non_dict_entries_skips_render():
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {"tier": "balanced"},
            "plugins": {
                "adapter": {"name": "hf"},
                "edit": {"name": "structured"},
                "guards": ["x", 123, None],
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
    }
    baseline = {**report, "edit": {"name": "noop"}}
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(canonical_run_report(report), canonical_baseline(baseline))
    md = render_report_markdown(cert)
    # No guards bullet should render because filtered list is empty
    assert "- Guards:" not in md
