from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_certificate_includes_provider_digest_in_provenance():
    report = {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 42,
            "ts": "now",
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
            "name": "noop",
            "plan_digest": "",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (1.0, 1.0),
            },
            "latency_ms_per_tok": 0.0,
            "memory_mb_peak": 0.0,
            "spectral": {},
            "rmt": {},
            "invariants": {},
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "provenance": {
            "provider_digest": {
                "ids_sha256": "id123",
                "tokenizer_sha256": "tok456",
                "masking_sha256": "mask789",
            }
        },
        "evaluation_windows": {
            "preview": {"window_ids": [1], "labels": [[-100]]},
            "final": {"window_ids": [2], "labels": [[-100]]},
        },
    }
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }

    cert = make_certificate(report, baseline)
    # Provider digest may be omitted after normalization; rendering should still succeed
    md = render_certificate_markdown(cert)
    assert "# InvarLock Evaluation Certificate" in md
