from __future__ import annotations

import json
from pathlib import Path

from scripts.evidence_packs.python.verdict.records import _build_record


def test_evidence_pack_record_surfaces_guard_warnings(tmp_path: Path) -> None:
    cert = {
        "validation": {
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
        },
        "invariants": {"status": "pass"},
        "spectral": {"caps_applied": 1},
        "guard_warnings": {
            "present": True,
            "warning_count": 1,
            "warnings": [
                {
                    "guard": "spectral",
                    "kind": "new_capped_module",
                    "module": "layers.31.mlp.up_proj",
                    "policy_gate": "pass",
                }
            ],
        },
    }

    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(cert), encoding="utf-8")

    record = _build_record(
        cert=cert,
        cert_path=cert_path,
        model_name="model",
        category="stress",
        scenario_id="spectral_moderate_scale_stress",
        run_num=1,
        scenario_index={},
        baseline_cert=None,
    )

    assert record["guard_warnings"]["present"] is True
    assert record["guard_warnings"]["warning_count"] == 1
    assert record["guard_warnings"]["warnings"][0]["kind"] == "new_capped_module"
