from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "public_evidence" / "larger_model_smoke_findings"

PRIVATE_TEXT_PATTERNS = (
    "/private/tmp",
    "/Users/",
    "/root",
    "root@",
    "private/remote host",
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_larger_model_smoke_findings_summary_is_public_safe_and_complete() -> None:
    summary = _load_json(EVIDENCE_DIR / "findings_summary.json")

    assert summary["schema"] == "invarlock.larger_model_smoke_findings.summary.v1"
    assert summary["status"] == "completed"
    assert summary["validation_environment"] == "CUDA-capable validation host"
    assert summary["raw_logs_published"] is False
    assert summary["weights_vendored"] is False
    assert summary["support_matrix_change_claimed"] is False
    assert summary["suite"] == "model-catalog-gpu"
    assert summary["execution_mode"] == "container"

    counts = summary["counts"]
    assert counts == {
        "completed_runs": 21,
        "clean_runs": 19,
        "failed_runs": 2,
        "unique_clean_lanes": 17,
        "unique_failed_lanes": 1,
        "pre_verification_failures": 2,
        "report_materialized_clean": 19,
        "verify_materialized_clean": 19,
    }

    clean_lanes = summary["clean_lanes"]
    assert isinstance(clean_lanes, list)
    assert len(clean_lanes) == counts["unique_clean_lanes"]
    for lane in clean_lanes:
        assert isinstance(lane, dict)
        assert lane["rc"] == 0
        assert lane["evaluate_exit"] == 0
        assert lane["verify_exit"] == 0
        assert lane["report_materialized"] is True
        assert lane["verify_materialized"] is True
        assert lane["status"] == "ok"
        preset = REPO_ROOT / str(lane["preset"])
        assert preset.is_file()

    duplicate_runs = summary["duplicate_clean_runs"]
    assert isinstance(duplicate_runs, list)
    assert {entry["slug"] for entry in duplicate_runs if isinstance(entry, dict)} == {
        "google_flan_t5_base",
        "tinyllama_tinyllama_1_1b_chat_v1_0",
    }

    failed_findings = summary["failed_findings"]
    assert isinstance(failed_findings, list)
    assert failed_findings == [
        {
            "slug": "microsoft_phi_4_mini_instruct",
            "model_id": "microsoft/Phi-4-mini-instruct",
            "preset": "configs/presets/causal_lm/phi4_mini_512.yaml",
            "attempts": 2,
            "status": "evaluate_failed_before_report",
            "rc": 1,
            "evaluate_exit": 1,
            "verify_exit": None,
            "report_materialized": False,
            "verify_materialized": False,
            "classification": "pre_verification_evaluate_failure",
            "public_note": (
                "Evaluation exited nonzero before report or verifier artifacts were "
                "materialized. This finding is not counted as clean evidence."
            ),
        }
    ]

    serialized = json.dumps(summary, sort_keys=True)
    for pattern in PRIVATE_TEXT_PATTERNS:
        assert pattern not in serialized
    assert "snapshot_restore_failed" not in serialized


def test_larger_model_smoke_findings_hash_inventory_matches_public_files() -> None:
    inventory = _load_json(EVIDENCE_DIR / "hash_inventory.json")

    assert inventory["schema"] == (
        "invarlock.larger_model_smoke_findings.hash_inventory.v1"
    )
    assert inventory["status"] == "completed"
    artifacts = inventory["artifacts"]
    assert isinstance(artifacts, list)

    by_path = {artifact["path"]: artifact for artifact in artifacts}
    assert set(by_path) == {
        "README.md",
        "findings_summary.json",
        "evidence.meta.json",
    }
    for rel_path, artifact in by_path.items():
        path = EVIDENCE_DIR / rel_path
        assert path.is_file()
        assert artifact["sha256"] == _sha256(path)
        assert artifact["bytes"] == path.stat().st_size


def test_larger_model_smoke_findings_metadata_declares_summary_only_findings() -> None:
    metadata = _load_json(EVIDENCE_DIR / "evidence.meta.json")

    assert metadata["schema"] == "invarlock.public_evidence.meta.v1"
    assert metadata["evidence_class"] == "larger_model_smoke_findings"
    assert metadata["artifact_paths"] == {
        "findings_summary": "findings_summary.json",
        "hash_inventory": "hash_inventory.json",
    }
    assert "invarlock evaluate" not in str(metadata["generated_by"])
