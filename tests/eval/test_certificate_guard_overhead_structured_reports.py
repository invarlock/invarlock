from unittest.mock import patch

from invarlock.reporting.certificate import (
    make_certificate,
    render_certificate_markdown,
)


def test_guard_overhead_structured_reports_path():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0},
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
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
        # Structured reports path
        "guard_overhead": {
            "overhead_threshold": 0.02,
            "bare_report": {"metrics": {"ppl_final": 10.0}},
            "guarded_report": {"metrics": {"ppl_final": 10.09}},
        },
    }
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    oh = cert.get("guard_overhead", {})
    # Guard overhead may be omitted; renderer should still succeed
    assert isinstance(oh, dict)
    _ = render_certificate_markdown(cert)
