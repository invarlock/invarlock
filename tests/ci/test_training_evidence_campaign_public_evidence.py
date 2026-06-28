from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = REPO_ROOT / "public_evidence" / "model_editing_evidence_bundle_v0"


def test_training_evidence_campaign_summary_is_public_safe() -> None:
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
    summary_path = REPO_ROOT / manifest["training_evidence_campaign_summary"]
    inventory_path = REPO_ROOT / manifest["training_evidence_campaign_hash_inventory"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    assert summary["schema"] == "invarlock.training_evidence_campaign.summary.v1"
    assert summary["status"] == "completed"
    assert summary["claim_boundary"] == (
        "empirical training evidence only; no new assurance claim"
    )
    assert summary["weights_vendored"] is False
    assert (
        inventory["schema"] == "invarlock.training_evidence_campaign.hash_inventory.v1"
    )
    assert inventory["status"] == "completed"
    assert inventory["weights_vendored"] is False
    assert {lane["target"] for lane in summary["lanes"]} == {
        "peft_lora",
        "fine_tune",
    }
    assert {artifact["target"] for artifact in inventory["artifacts"]} == {
        "peft_lora",
        "fine_tune",
    }
    assert {lane["verification"]["verify_status"] for lane in summary["lanes"]} == {
        "ok"
    }
    assert {
        lane["verification"]["runtime_provenance_status"] for lane in summary["lanes"]
    } == {"verified"}
    assert "CUDA-capable validation host" in json.dumps(summary)
    for public_text in [
        summary_path.read_text(encoding="utf-8"),
        inventory_path.read_text(encoding="utf-8"),
    ]:
        assert "/private/tmp" not in public_text
        assert "/Users/" not in public_text
        assert "/root" not in public_text
        assert "root@" not in public_text
