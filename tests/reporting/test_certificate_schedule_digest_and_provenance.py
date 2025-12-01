from __future__ import annotations

from typing import Any

from invarlock.reporting.certificate import make_certificate


def _mk_report_with_final_windows() -> dict[str, Any]:
    return {
        "meta": {"model_id": "stub", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 0,
            "final_n": 3,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            }
        },
        "evaluation_windows": {
            "final": {
                "window_ids": [10, 20, 30],
                "logloss": [2.0, 2.0, 2.0],
                "token_counts": [10, 10, 10],
            }
        },
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def _mk_baseline_like(report: dict[str, Any]) -> dict[str, Any]:
    # Reuse final windows to ensure pairing logic is valid
    return {
        "run_id": "base",
        "model_id": report["meta"]["model_id"],
        "evaluation_windows": {"final": report["evaluation_windows"]["final"]},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }


def test_window_ids_digest_and_guard_schedule_digest_present() -> None:
    rep = _mk_report_with_final_windows()
    base = _mk_baseline_like(rep)
    cert = make_certificate(rep, base)

    # Provenance should include a stable digest of the final window IDs
    prov = cert.get("provenance", {})
    assert isinstance(prov, dict)
    digest = prov.get("window_ids_digest")
    assert isinstance(digest, str) and len(digest) >= 8

    # Guard overhead section mirrors schedule digest for auditability
    guard = cert.get("guard_overhead", {})
    assert isinstance(guard, dict)
    assert guard.get("schedule_digest") == digest
