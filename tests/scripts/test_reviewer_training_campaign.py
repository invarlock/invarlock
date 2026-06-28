from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "smoke" / "run_reviewer_training_campaign.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_reviewer_training_campaign", SCRIPT_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _public_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reviewer_training_campaign_dry_run_writes_public_safe_summary(
    tmp_path: Path,
) -> None:
    module = _load_module()
    work_root = tmp_path / "campaign"
    publish_root = tmp_path / "published"

    result = module.main(
        [
            "--dry-run",
            "--campaign-id",
            "reviewer-training-test",
            "--work-root",
            str(work_root),
            "--publish-summary",
            str(publish_root),
        ]
    )

    assert result == 0
    summary_path = work_root / "campaign_summary.json"
    inventory_path = work_root / "hash_inventory.json"
    published_summary_path = publish_root / "campaign_summary.json"
    published_inventory_path = publish_root / "hash_inventory.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    assert summary["schema"] == module.SUMMARY_SCHEMA
    assert summary["campaign_id"] == "reviewer-training-test"
    assert summary["status"] == "planned"
    assert summary["claim_boundary"] == module.CLAIM_BOUNDARY
    assert summary["weights_vendored"] is False
    assert {lane["target"] for lane in summary["lanes"]} == {
        "peft_lora",
        "fine_tune",
    }
    assert {lane["display_name"] for lane in summary["lanes"]} == {
        "PEFT LoRA train-and-merge subject",
        "Full fine-tune subject",
    }
    for lane in summary["lanes"]:
        assert lane["status"] == "planned"
        assert lane["weights_vendored"] is False
        assert "checkpoint_refs.json" in lane["publishable_artifact_names"]
        assert "evaluation.report.json" in lane["publishable_artifact_names"]
        assert not any(
            name.endswith((".safetensors", ".bin", ".pt"))
            for name in lane["publishable_artifact_names"]
        )
        assert lane["campaign_relative_report_dir"].startswith(
            f"{lane['target']}/reports/"
        )

    assert inventory == {
        "schema": module.HASH_INVENTORY_SCHEMA,
        "campaign_id": "reviewer-training-test",
        "status": "planned",
        "claim_boundary": module.CLAIM_BOUNDARY,
        "weights_vendored": False,
        "artifacts": [],
    }
    assert published_summary_path.read_text(encoding="utf-8") == _public_text(
        summary_path
    )
    assert published_inventory_path.read_text(encoding="utf-8") == _public_text(
        inventory_path
    )

    public_text = published_summary_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in public_text
    assert "/Users/" not in public_text
    assert "/private/tmp" not in public_text
    assert "/root" not in public_text
    assert "root@" not in public_text


def test_reviewer_training_campaign_completed_lane_hashes_artifacts(
    tmp_path: Path,
) -> None:
    module = _load_module()
    work_root = tmp_path / "campaign"
    report_dir = work_root / "fine_tune" / "reports" / "tiny-fine-tune"
    report_dir.mkdir(parents=True)

    for filename in [
        "evaluation.report.json",
        "runtime.manifest.json",
        "verify.json",
        "checkpoint_refs.json",
        "external_edit_summary.json",
        "fixture_summary.json",
    ]:
        _write_json(report_dir / filename, {"name": filename})
    _write_json(
        report_dir / "lane_artifact.json",
        {"lane_artifact_label": "cpu-host-off"},
    )
    (report_dir / "run_summary.txt").write_text(
        "\n".join(
            [
                "status: success",
                "lane_artifact_label: cpu-host-off",
                "execution_mode: host",
                "assurance: off",
                "runtime_provenance: host",
                "device: cpu",
                "verify_status: ok",
                "verify_runtime_provenance_status: verified",
                "verify_runtime_provenance_verified: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    args = Namespace(
        allow_network=False,
        device="cpu",
        execution_lane="host",
        profile="release",
        render_html=False,
        tier="balanced",
    )
    lane, inventory = module._completed_lane_payload(
        config=module.TARGETS["fine_tune"],
        paths={"report_dir": report_dir},
        args=args,
        work_root=work_root,
    )

    assert lane["status"] == "completed"
    assert lane["target"] == "fine_tune"
    assert lane["weights_vendored"] is False
    assert lane["lane_artifact_label"] == "cpu-host-off"
    assert lane["verification"] == {
        "assurance": "off",
        "profile": "release",
        "runtime_provenance": "host",
        "runtime_provenance_status": "verified",
        "runtime_provenance_verified": "true",
        "verify_status": "ok",
    }
    assert set(lane["artifacts"]) == set(module.PUBLISHABLE_ARTIFACTS)
    assert len(inventory) == len(module.PUBLISHABLE_ARTIFACTS)
    for artifact in inventory:
        assert artifact["target"] == "fine_tune"
        assert artifact["campaign_relative_path"].startswith("fine_tune/reports/")
        assert artifact["sha256"].startswith("sha256:")
        assert artifact["bytes"] > 0
        assert "repo_relative_path" not in artifact


def test_reviewer_training_campaign_rejects_unknown_target() -> None:
    module = _load_module()

    try:
        module._selected_targets(["peft_lora,missing"])
    except ValueError as exc:
        assert "unknown target" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown target was accepted")
