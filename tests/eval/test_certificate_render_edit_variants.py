from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


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
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "ppl_ratio": 1.0,
            "ppl_preview_ci": (9.5, 10.5),
            "ppl_final_ci": (9.5, 10.5),
            "ppl_ratio_ci": (0.9, 1.1),
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [1.0]}},
    }
    baseline = {
        "run_id": "b2",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [1.0]}},
    }
    with (
        patch("invarlock.reporting.certificate.validate_report", return_value=True),
        patch(
            "invarlock.reporting.certificate.compute_paired_delta_log_ci",
            return_value=(-0.1, 0.1),
        ),
    ):
        cert = make_certificate(report, baseline)
    return cert


def test_render_edit_name_variants():
    for name in ("quant_rtn", "lowrank_svd", "structured", "custom_unknown"):
        cert = _mk_cert(name)
        md = render_certificate_markdown(cert)
        assert isinstance(md, str) and "Guard Observability" in md
