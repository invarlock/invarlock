from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def _mk_minimal_report() -> dict:
    return {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
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
            "preview": {
                "window_ids": [1],
                "logloss": [2.302585093],
                "token_counts": [1],
            },
            "final": {"window_ids": [2], "logloss": [2.302585093], "token_counts": [1]},
        },
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_certificate_embeds_telemetry_summary_line(monkeypatch):
    # Ensure telemetry emission path runs (printing is optional; we inspect certificate payload)
    monkeypatch.setenv("INVARLOCK_TELEMETRY", "1")
    rep = _mk_minimal_report()
    base = _mk_minimal_report()

    cert = make_certificate(rep, base)
    tel = cert.get("telemetry", {})
    assert isinstance(tel, dict)
    line = tel.get("summary_line")
    assert isinstance(line, str) and line.startswith("INVARLOCK_TELEMETRY ")
