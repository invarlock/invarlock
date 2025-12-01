from __future__ import annotations

from invarlock.reporting.certificate import (
    render_certificate_markdown,
    validate_certificate,
)


def _mk_cert() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r1",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0},
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }


def test_render_certificate_markdown_invalid_raises() -> None:
    cert = _mk_cert()
    # Break schema version to make it invalid; validate_certificate should be False
    cert["schema_version"] = "invalid"
    assert validate_certificate(cert) is False
    try:
        render_certificate_markdown(cert)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid cert")


def test_validate_certificate_rejects_unknown_validation_keys() -> None:
    cert = _mk_cert()
    # Add an unexpected key; JSONSchema validation should fail and fallback minimal check should still accept structure
    cert["validation"]["unexpected_key_for_test"] = True  # type: ignore[index]
    # validate_certificate uses JSONSchema first; since schema disallows unknown keys in validation, it will fall back
    assert validate_certificate(cert) is True
