from __future__ import annotations

import json

from invarlock.reporting.certificate_schema import (
    CERTIFICATE_SCHEMA_VERSION,
    validate_certificate,
)
from invarlock.reporting.dataset_hashing import (
    _compute_actual_window_hashes,
    _extract_dataset_info,
)


def test_certificate_schema_valid_and_fallback(monkeypatch):
    # Valid minimal certificate (JSON schema path)
    cert = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "RID1234",
        "artifacts": {},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "ds",
            "seq_len": 4,
            "windows": {"preview": 1, "final": 1},
        },
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        "validation": {},
    }
    assert validate_certificate(cert) is True

    # Force JSON schema failure by adding an unknown validation key; should fall back to minimal
    cert2 = json.loads(json.dumps(cert))
    cert2["validation"] = {"not_allowed_key": True}
    # Ensure our schema has been tightened by the module code
    assert validate_certificate(cert2) is True


def test_dataset_hashing_helpers():
    # Config-based fallback – missing explicit window IDs
    report = {
        "meta": {"seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "evaluation_windows": {},
    }
    hashed = _compute_actual_window_hashes(report)
    assert isinstance(hashed, dict) and hashed
    info = _extract_dataset_info(report)
    assert info["provider"] == "ds" and "hash" in info
    # Explicit input_ids path
    report2 = {
        "data": {
            "dataset": "ds2",
            "split": "val",
            "seq_len": 2,
            "preview_n": 1,
            "final_n": 1,
        },
        "evaluation_windows": {
            "preview": {"input_ids": [[1, 2]], "window_ids": [1]},
            "final": {"input_ids": [[3, 4]], "window_ids": [2]},
        },
    }
    info2 = _extract_dataset_info(report2)
    assert info2["hash"]["preview"].startswith("sha256:") and info2["hash"][
        "final"
    ].startswith("sha256:")
    # Explicit data hashes path
    report3 = {
        "data": {
            "dataset": "ds3",
            "split": "val",
            "seq_len": 2,
            "preview_n": 1,
            "final_n": 1,
            "preview_hash": "abc",
            "final_hash": "def",
            "preview_total_tokens": 2,
            "final_total_tokens": 2,
        }
    }
    info3 = _extract_dataset_info(report3)
    assert info3["hash"]["preview"].startswith("blake2s:") and info3["hash"][
        "final"
    ].startswith("blake2s:")
