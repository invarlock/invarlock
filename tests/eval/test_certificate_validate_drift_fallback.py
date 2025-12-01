from invarlock.reporting.certificate import (
    CERTIFICATE_SCHEMA_VERSION,
    validate_certificate,
)


def test_validate_certificate_accepts_pm_only_without_legacy_drift():
    cert = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "r1",
        "meta": {},
        "auto": {},
        "dataset": {"windows": {"preview": 1, "final": 1, "seed": 42}},
        "baseline_ref": {},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 11.0,
            "ratio_vs_baseline": 1.1,
            "display_ci": [1.0, 1.2],
        },
        "invariants": {},
        "spectral": {},
        "rmt": {},
        "variance": {},
        "structure": {},
        "policies": {},
        "plugins": {},
        "artifacts": {},
        "validation": {
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_overhead_acceptable": True,
        },
    }
    assert validate_certificate(cert) is True
