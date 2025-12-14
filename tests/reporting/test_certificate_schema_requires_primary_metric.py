from __future__ import annotations

import invarlock.reporting.certificate_schema as schema_mod


def test_certificate_schema_requires_primary_metric_and_window_stats() -> None:
    cert = {
        "schema_version": schema_mod.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-1234",
        "artifacts": {},
        "plugins": {"adapters": [], "edits": [], "guards": []},
        "meta": {"model_id": "m", "adapter": "hf_gpt2", "seed": 1, "device": "cpu"},
        "dataset": {
            "provider": "synthetic",
            "seq_len": 8,
            "windows": {"preview": 1, "final": 1, "seed": 42},
        },
        # Intentionally omit primary_metric + dataset.windows.stats (legacy)
    }

    assert schema_mod.validate_certificate(cert) is False
