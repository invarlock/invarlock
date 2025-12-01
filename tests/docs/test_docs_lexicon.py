from __future__ import annotations

import json
from pathlib import Path


def test_reference_uses_canonical_labels_only():
    # Load the v1 schema reference JSON example and validate its validation keys
    p = Path("docs/reference/certificate-schema.md")
    text = p.read_text("utf-8")
    # Extract the JSON block delimited by ```json ... ```
    start = text.find("```json")
    assert start != -1, "v1 reference must include a JSON example"
    end = text.find("```", start + 7)
    assert end != -1
    snippet = text[start + 7 : end].strip()
    obj = json.loads(snippet)
    validation = obj.get("validation", {}) or {}
    allowed = {
        "primary_metric_acceptable",
        "preview_final_drift_acceptable",
        "guard_overhead_acceptable",
        # optional non-gating rows may be omitted from examples
    }
    assert set(validation.keys()).issubset(allowed)
