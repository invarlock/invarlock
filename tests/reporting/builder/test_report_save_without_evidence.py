from __future__ import annotations

import copy
from pathlib import Path

from invarlock.reporting.report_bundle import save_evaluation_bundle
from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import canonical_run_report


def _report_and_base():
    rep = {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "abc",
            "seed": 1,
            "device": "cpu",
            "ts": "2024-01-01T00:00:00",
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "deadbeef",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 100.0,
                "preview": 100.0,
                "ratio_vs_baseline": 1.0,
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "evaluation_windows": {
            "final": {"window_ids": [1], "logloss": [4.0], "token_counts": [100]}
        },
    }
    base = copy.deepcopy(rep)
    base["edit"]["name"] = "noop"
    return canonical_run_report(rep), canonical_run_report(base)


def test_save_report_without_evidence(tmp_path: Path):
    rep, base = _report_and_base()
    out = save_evaluation_bundle(
        run_report=rep,
        output_dir=tmp_path,
        evaluation_report=make_report(rep, base),
    )
    assert out["report"].exists() and out["report_md"].exists()
    # No debug env flag → evidence file is optional and typically absent
    assert not (tmp_path / "guards_evidence.json").exists()
