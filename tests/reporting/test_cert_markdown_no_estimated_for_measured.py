from invarlock.reporting.certificate import CERTIFICATE_SCHEMA_VERSION
from invarlock.reporting.render import render_certificate_markdown


def test_markdown_excludes_estimated_suffix_for_measured_accuracy():
    cert = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "abc",
        "artifacts": {"generated_at": "now"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "wikitext2",
            "seq_len": 16,
            "windows": {"preview": 1, "final": 1, "seed": 1},
            "hash": {"preview_tokens": None, "final_tokens": None, "total_tokens": 0},
        },
        "invariants": {"status": "pass"},
        "spectral": {"caps_applied": 0, "max_caps": 5},
        "rmt": {"stable": True},
        "guard_overhead": {},
        "structure": {"layers_modified": 0, "params_changed": 0},
        "variance": {"enabled": False},
        "validation": {
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_overhead_acceptable": True,
        },
        "primary_metric": {
            "kind": "accuracy",
            "preview": 0.8,
            "final": 0.9,
            "ratio_vs_baseline": +0.1,
            "estimated": False,
            "counts_source": "measured",
        },
        "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        "policies": {},
        "resolved_policy": {},
        "policy_provenance": {},
    }
    md = render_certificate_markdown(cert)
    assert "(estimated)" not in md
