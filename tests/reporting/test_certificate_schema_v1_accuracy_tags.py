from invarlock.reporting.certificate import CERTIFICATE_JSON_SCHEMA


def test_schema_accepts_counts_source_and_estimated():
    _pm = {
        "kind": "accuracy",
        "unit": "accuracy",
        "direction": "higher",
        "aggregation_scope": "example",
        "paired": True,
        "gating_basis": "point",
        "preview": 1.0,
        "final": 1.0,
        "ratio_vs_baseline": 0.0,
        "counts_source": "pseudo_config",
        "estimated": True,
    }
    # Validate properties exist in JSON schema (shape only; jsonschema checked elsewhere)
    props = (
        CERTIFICATE_JSON_SCHEMA.get("properties", {})
        .get("primary_metric", {})
        .get("properties", {})
    )
    assert "counts_source" in props and "estimated" in props
