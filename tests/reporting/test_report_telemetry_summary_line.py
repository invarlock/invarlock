from __future__ import annotations

from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_telemetry import (
    telemetry_output_enabled,
    telemetry_summary_line,
)


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


def test_evaluation_report_embeds_telemetry_summary_line(monkeypatch):
    # Ensure telemetry emission path runs (printing is optional; we inspect evaluation_report payload)
    monkeypatch.setenv("INVARLOCK_TELEMETRY", "1")
    rep = _mk_minimal_report()
    base = _mk_minimal_report()

    cert = make_report(rep, base)
    tel = cert.get("telemetry", {})
    assert isinstance(tel, dict)
    line = tel.get("summary_line")
    assert isinstance(line, str) and line.startswith("INVARLOCK_TELEMETRY ")


def test_telemetry_helpers_resolve_summary_line_and_env(monkeypatch):
    monkeypatch.setenv("INVARLOCK_TELEMETRY", "yes")
    assert telemetry_output_enabled() is True
    assert (
        telemetry_summary_line(
            {"telemetry": {"summary_line": "INVARLOCK_TELEMETRY run_id=demo"}}
        )
        == "INVARLOCK_TELEMETRY run_id=demo"
    )

    monkeypatch.delenv("INVARLOCK_TELEMETRY", raising=False)
    assert telemetry_output_enabled() is False
    assert telemetry_summary_line({"telemetry": {}}) is None


def test_telemetry_summary_line_strips_control_characters() -> None:
    rep = _mk_minimal_report()
    rep.setdefault("provenance", {})["dataset_split"] = "val\nforged"
    rep.setdefault("metrics", {})["primary_metric"]["kind"] = "ppl\tcausal"
    rep.setdefault("meta", {})["run_id"] = "demo\r\nrun"
    base = _mk_minimal_report()

    cert = make_report(rep, base)

    line = telemetry_summary_line(cert)
    assert line is not None
    assert "\n" not in line and "\r" not in line and "\t" not in line
    assert "metric=ppl causal" in line
    assert "split=val forged" in line
