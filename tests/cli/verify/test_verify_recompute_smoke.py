from __future__ import annotations

import json
import math
from pathlib import Path

from invarlock.reporting.report_make import make_report
from invarlock.reporting.verify_contract import _validate_evaluation_report_payload
from tests.cli._support_runtime_policy import bind_runtime_policy


def _mk_report() -> dict:
    report = {
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
                "final": 50.0,
                "display_ci": (1.0, 1.0),
                "ratio_vs_baseline": 1.0,
            },
            "bootstrap": {
                "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
            },
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [0],
                "logloss": [math.log(50.0)],
                "token_counts": [1],
            },
            "final": {
                "window_ids": [1],
                "logloss": [math.log(50.0)],
                "token_counts": [1],
            },
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    return bind_runtime_policy(report)


def test_verify_smoke_recompute_and_consistency(tmp_path: Path):
    report = _mk_report()
    baseline = _mk_report()
    cert = make_report(report, baseline)
    p = tmp_path / "cert.json"
    p.write_text(json.dumps(cert))
    errors = _validate_evaluation_report_payload(p)
    assert errors == []
