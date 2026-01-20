from __future__ import annotations

import json
import os
from pathlib import Path

from invarlock.reporting.report import save_report


def _minimal_report() -> dict:
    return {
        "meta": {
            "model_id": "stub",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 7,
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "x",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            }
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_manifest_matches_schema(tmp_path: Path):
    primary = _minimal_report()
    baseline = _minimal_report()
    out_dir = tmp_path / "cert"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enable evidence pointer
    os.environ["INVARLOCK_EVIDENCE_DEBUG"] = "1"
    try:
        save_report(
            primary,
            out_dir,
            formats=["cert"],
            baseline=baseline,
            filename_prefix="evaluation",
        )
    finally:
        os.environ.pop("INVARLOCK_EVIDENCE_DEBUG", None)

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text("utf-8"))

    import jsonschema  # type: ignore

    schema = json.loads(
        (Path.cwd() / "tests/schemas/manifest_v1.schema.json").read_text("utf-8")
    )
    jsonschema.validate(instance=manifest, schema=schema)
    # Evidence pointer is present when env flag was set
    ev = manifest.get("evidence", {})
    assert isinstance(ev, dict) and ev.get("guards_evidence"), (
        "evidence.guards_evidence missing"
    )
    assert (out_dir / Path(ev["guards_evidence"]).name).exists(), (
        "guards_evidence.json file missing"
    )
