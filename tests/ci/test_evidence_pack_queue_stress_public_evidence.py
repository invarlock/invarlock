from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "public_evidence" / "evidence_pack_queue_stress_resume"

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


def test_queue_stress_summary_is_public_safe_and_complete() -> None:
    summary = _load_json(EVIDENCE_DIR / "stress_summary.json")

    assert summary["schema"] == "invarlock.evidence_pack_queue_stress_resume.summary.v1"
    assert summary["status"] == "completed"
    assert summary["evidence_scope"] == (
        "evidence-pack queue stress/resume validation only; "
        "no model-quality or assurance claim"
    )
    assert summary["validation_environment"] == "CUDA-capable validation host"
    assert summary["raw_logs_published"] is False
    assert summary["weights_vendored"] is False

    suites = summary["suites"]
    assert isinstance(suites, list)
    by_name = {suite["name"]: suite for suite in suites if isinstance(suite, dict)}
    assert set(by_name) == {"queue_manager_shell", "queue_state_python"}

    queue_manager = by_name["queue_manager_shell"]
    assert queue_manager["command"] == (
        "scripts/evidence_packs/tests/run.sh --filter test_queue_manager"
    )
    assert queue_manager["rc"] == 0
    assert queue_manager["tests_passed"] == 74
    assert queue_manager["tests_failed"] == 0
    assert "orphaned_running_task_reclamation" in queue_manager["coverage_surface"]

    queue_state = by_name["queue_state_python"]
    assert queue_state["command"] == (
        "python -m pytest tests/scripts/test_evidence_pack_queue_state.py -q"
    )
    assert queue_state["rc"] == 0
    assert queue_state["tests_passed"] == 3
    assert queue_state["tests_failed"] == 0

    serialized = json.dumps(summary, sort_keys=True)
    for pattern in PRIVATE_TEXT_PATTERNS:
        assert pattern not in serialized


def test_queue_stress_hash_inventory_matches_public_files() -> None:
    inventory = _load_json(EVIDENCE_DIR / "hash_inventory.json")

    assert inventory["schema"] == (
        "invarlock.evidence_pack_queue_stress_resume.hash_inventory.v1"
    )
    assert inventory["status"] == "completed"
    artifacts = inventory["artifacts"]
    assert isinstance(artifacts, list)

    by_path = {artifact["path"]: artifact for artifact in artifacts}
    assert set(by_path) == {
        "README.md",
        "stress_summary.json",
        "evidence.meta.json",
    }
    for rel_path, artifact in by_path.items():
        path = EVIDENCE_DIR / rel_path
        assert path.is_file()
        assert artifact["sha256"] == _sha256(path)
        assert artifact["bytes"] == path.stat().st_size


def test_queue_stress_metadata_declares_summary_only_scope() -> None:
    metadata = _load_json(EVIDENCE_DIR / "evidence.meta.json")

    assert metadata["schema"] == "invarlock.public_evidence.meta.v1"
    assert metadata["evidence_class"] == "evidence_pack_queue_stress_resume"
    assert metadata["artifact_paths"] == {
        "stress_summary": "stress_summary.json",
        "hash_inventory": "hash_inventory.json",
    }
    assert "invarlock evaluate" not in str(metadata["generated_by"])
