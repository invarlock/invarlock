from __future__ import annotations

from pathlib import Path

import jsonschema

from invarlock.public_contracts import load_verify_output_schema
from invarlock.reporting.verify_output import (
    build_verify_error_payload,
    build_verify_json_payload,
)


def test_verify_output_success_payload_matches_public_schema(tmp_path: Path) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report = {
        "schema_version": "v1",
        "primary_metric": {
            "kind": "ppl_causal",
            "ratio_vs_baseline": 1.0,
            "display_ci": [0.99, 1.01],
        },
    }

    payload = build_verify_json_payload(
        [report_path],
        ok=True,
        reason="ok",
        tolerance=1e-9,
        load_report_fn=lambda _path: report,
    )

    jsonschema.validate(instance=payload, schema=load_verify_output_schema())


def test_verify_output_error_payload_matches_public_schema(tmp_path: Path) -> None:
    payload = build_verify_error_payload(
        tmp_path / "bad.report.json",
        reason="malformed",
        encoded_error={"code": "E601", "message": "schema validation failed"},
    )

    jsonschema.validate(instance=payload, schema=load_verify_output_schema())
