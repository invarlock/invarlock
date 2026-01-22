from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.commands.verify import _validate_certificate_payload
from invarlock.reporting.certificate import make_certificate


def _mk_report() -> dict:
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
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
                "layers_modified": 0,
                "sparsity": None,
                "bitwidth_map": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 49.0,
                "display_ci": (48.0, 50.0),
                "ratio_vs_baseline": 49.0 / 50.0,
            },
            # Pairing stats (PM-only location is dataset.windows.stats, but tests seed via metrics)
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "bootstrap": {
                "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                # Deprecated copies (kept for context)
                "window_match_fraction": 1.0,
                "window_overlap_fraction": 0.0,
            },
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_verify_smoke_recompute_and_consistency(tmp_path: Path):
    report = _mk_report()
    baseline = _mk_report()
    cert = make_certificate(report, baseline)
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    errors = _validate_certificate_payload(p)
    assert errors == []
