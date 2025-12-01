from __future__ import annotations

from pathlib import Path

from invarlock.reporting.report import save_report


def _report_and_base():
    rep = {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "abc",
            "seed": 1,
            "device": "cpu",
            "ts": "2024-01-01T00:00:00",
            "auto": None,
        },
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
    base = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 100.0}},
        "evaluation_windows": {
            "final": {"window_ids": [1], "logloss": [4.0], "token_counts": [100]}
        },
    }
    return rep, base


def test_save_report_cert_without_evidence(tmp_path: Path):
    rep, base = _report_and_base()
    out = save_report(
        rep, tmp_path, formats=["cert"], baseline=base, filename_prefix="r"
    )
    assert out["cert"].exists() and out["cert_md"].exists()
    # No debug env flag → evidence file is optional and typically absent
    assert not (tmp_path / "guards_evidence.json").exists()
