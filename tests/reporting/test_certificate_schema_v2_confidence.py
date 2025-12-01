from __future__ import annotations

import math
from typing import Any

from invarlock.reporting.certificate import (
    CERTIFICATE_JSON_SCHEMA,
    make_certificate,
)


def _mk_report() -> dict[str, Any]:
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "seed": 42,
            "seeds": {"python": 42, "numpy": 42, "torch": 42},
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {"name": "noop", "plan_digest": "noop"},
        "metrics": {
            # Ensure paired path with finite CI width
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 50.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (0.98, 1.02),
            },
            "bootstrap": {"replicates": 200, "alpha": 0.05},
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [math.log(50.0), math.log(50.0)],
                "token_counts": [100, 200],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [math.log(50.0), math.log(50.0)],
                "token_counts": [100, 200],
            },
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_certificate_schema_includes_confidence_block():
    # Schema should expose a confidence object with the documented fields
    props = CERTIFICATE_JSON_SCHEMA["properties"]
    assert "confidence" in props, "schema must include confidence block"
    conf = props["confidence"]
    assert conf["type"] == "object"
    cprops = conf["properties"]
    assert {"label", "basis", "width", "threshold", "unstable"}.issubset(
        set(cprops.keys())
    )
    # label is a bounded enum; label and basis are required
    assert cprops["label"]["enum"] == ["High", "Medium", "Low"]
    assert set(conf["required"]) == {"label", "basis"}


def test_generated_certificate_populates_confidence_fields():
    report = _mk_report()
    baseline = _mk_report()
    cert = make_certificate(report, baseline)
    conf = cert.get("confidence", {})
    assert isinstance(conf, dict) and conf, (
        "confidence should be present on certificate"
    )
    assert conf.get("label") in {"High", "Medium", "Low"}
    assert conf.get("basis") in {"ppl_ratio", "accuracy", "vqa_accuracy"}
    # width/threshold should be numeric when CI is present
    assert isinstance(conf.get("width"), float)
    assert isinstance(conf.get("threshold"), float)
