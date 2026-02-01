from __future__ import annotations

import invarlock.reporting.report_schema as schema_mod


def test_evaluation_report_schema_requires_primary_metric_and_window_stats() -> None:
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-1234",
        "artifacts": {},
        "plugins": {"adapters": [], "edits": [], "guards": []},
        "meta": {"model_id": "m", "adapter": "hf_causal", "seed": 1, "device": "cpu"},
        "dataset": {
            "provider": "synthetic",
            "seq_len": 8,
            "windows": {"preview": 1, "final": 1, "seed": 42},
        },
        # Intentionally omit primary_metric + dataset.windows.stats
    }

    assert schema_mod.validate_report(cert) is False
