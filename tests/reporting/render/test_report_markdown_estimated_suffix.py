from invarlock.reporting.rendering.markdown import render_report_markdown
from invarlock.reporting.report_make import REPORT_SCHEMA_VERSION


def test_markdown_includes_estimated_suffix_and_note_for_accuracy():
    evaluation_report = {
        "schema_version": REPORT_SCHEMA_VERSION,
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
        "guard_metric_impact": {},
        "structure": {"layers_modified": 0, "params_changed": 0},
        "variance": {"enabled": False},
        "validation": {
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_metric_impact_acceptable": True,
        },
        "primary_metric": {
            "kind": "accuracy",
            "preview": 1.0,
            "final": 1.0,
            "delta_vs_baseline_pp": 0.0,
            "estimated": True,
        },
        "auto": {"tier": "balanced", "probes_used": 0, "target_ppl_ratio": None},
        "policies": {},
        "resolved_policy": {},
        "policy_provenance": {},
    }
    md = render_report_markdown(evaluation_report)
    assert "(estimated)" in md
    assert "ESTIMATED / PSEUDO ACCURACY" in md
    assert "NOT MEASURED LABEL ACCURACY" in md
    assert "NON-ASSURANCE REPORT" in md
    assert "Accuracy derived from pseudo counts" in md
