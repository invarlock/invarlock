from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_guard_overhead_direct_values_and_unavailable_ratio_path(monkeypatch):
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": float("nan"),
            }
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
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
        # Inject a guard_overhead block with bare/guarded values but invalid (bare_ppl <= 0) → ratio unavailable path
        "guard_overhead": {
            "bare_ppl": 0.0,
            "guarded_ppl": 10.0,
            "overhead_threshold": 0.01,
        },
    }
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 10.0,
                "preview": 10.0,
            }
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    monkeypatch.setattr(
        "invarlock.reporting.certificate.validate_report", lambda _: True
    )
    cert = make_certificate(report, baseline)
    # Guard overhead section may be omitted; renderer should handle it gracefully
    oh = cert.get("guard_overhead", {})
    assert isinstance(oh, dict)
    _ = render_certificate_markdown(cert)
