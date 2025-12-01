import pytest

from invarlock.reporting.render import render_certificate_markdown


def test_render_certificate_markdown_rejects_invalid_schema():
    # Minimal malformed certificate to exercise validate_certificate(False) path
    cert = {
        "schema_version": "bad-version",
        "run_id": "r1",
        "meta": {},
        "auto": {},
        "dataset": {"windows": {"preview": 1, "final": 1, "seed": 42}},
        "baseline_ref": {},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "invariants": {},
        "spectral": {},
        "rmt": {},
        "variance": {},
        "structure": {},
        "policies": {},
        "plugins": {},
        "artifacts": {},
        "validation": {"primary_metric_acceptable": True},
    }
    with pytest.raises(ValueError):
        render_certificate_markdown(cert)
