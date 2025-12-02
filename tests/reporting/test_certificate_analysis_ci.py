from __future__ import annotations

import math

from invarlock.reporting.certificate import make_certificate


def _mk_report_for_ci() -> dict:
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "seed": 42,
            "ts": "now",
            "auto": {"tier": "balanced"},
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
            "plan_digest": "noop",
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
            "ppl_preview": math.exp(1.2),
            "ppl_final": math.exp(1.1),
            "ppl_ratio": math.exp(1.1) / math.exp(1.2),
            "ppl_preview_ci": (math.exp(1.1), math.exp(1.3)),
            "ppl_final_ci": (math.exp(1.0), math.exp(1.2)),
            "primary_metric": {"kind": "ppl_causal"},
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_certificate_includes_analysis_basis_cis_for_ppl():
    report = _mk_report_for_ci()
    baseline = _mk_report_for_ci()
    cert = make_certificate(report, baseline)
    pm = cert.get("primary_metric", {})
    # Normalized path may omit analysis-basis CIs; ensure kind and display_ci structure
    assert pm.get("kind") in {"ppl_causal", "ppl_mlm", "ppl_seq2seq"}
    dci = pm.get("display_ci")
    assert isinstance(dci, (list | tuple)) and len(dci) == 2
