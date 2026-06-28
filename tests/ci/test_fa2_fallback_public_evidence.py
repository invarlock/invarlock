from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "public_evidence" / "fa2_fallback_compatibility"

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


def test_fa2_fallback_summary_is_public_safe_and_complete() -> None:
    summary = _load_json(EVIDENCE_DIR / "compatibility_summary.json")

    assert summary["schema"] == "invarlock.fa2_fallback_compatibility.summary.v1"
    assert summary["status"] == "completed"
    assert summary["evidence_scope"] == (
        "FA2 unavailable fallback behavior only; "
        "no FA2 success, model-quality, or assurance claim"
    )
    assert summary["validation_environment"] == "CUDA-capable validation host"
    assert summary["raw_logs_published"] is False
    assert summary["weights_vendored"] is False
    assert summary["fa2_success_claimed"] is False

    probe = summary["cuda_probe"]
    assert isinstance(probe, dict)
    assert probe["rc"] == 0
    assert probe["torch_cuda_available"] is True
    assert probe["torch_cuda_device_count"] == 2
    assert probe["flash_attn_importable"] is False
    assert probe["transformers_flash_attn_2_available"] is False

    checks = summary["checks"]
    assert isinstance(checks, list)
    by_name = {check["name"]: check for check in checks if isinstance(check, dict)}
    assert set(by_name) == {
        "flash_attn_dependency_fallbacks",
        "flash_attention_config_fallback",
    }

    dependency = by_name["flash_attn_dependency_fallbacks"]
    assert (
        dependency["command"]
        == "scripts/evidence_packs/tests/run.sh --filter flash_attn"
    )
    assert dependency["rc"] == 0
    assert dependency["tests_passed"] == 3
    assert dependency["tests_failed"] == 0
    assert "install_import_failure_fallback" in dependency["coverage_surface"]
    assert "killed_build_fallback" in dependency["coverage_surface"]

    config = by_name["flash_attention_config_fallback"]
    assert (
        config["command"] == "scripts/evidence_packs/tests/run.sh --filter strict_flash"
    )
    assert config["rc"] == 0
    assert config["tests_passed"] == 1
    assert config["tests_failed"] == 0
    assert (
        "eager_attention_config_comment_when_unavailable" in config["coverage_surface"]
    )

    serialized = json.dumps(summary, sort_keys=True)
    for pattern in PRIVATE_TEXT_PATTERNS:
        assert pattern not in serialized


def test_fa2_fallback_hash_inventory_matches_public_files() -> None:
    inventory = _load_json(EVIDENCE_DIR / "hash_inventory.json")

    assert (
        inventory["schema"] == "invarlock.fa2_fallback_compatibility.hash_inventory.v1"
    )
    assert inventory["status"] == "completed"
    artifacts = inventory["artifacts"]
    assert isinstance(artifacts, list)

    by_path = {artifact["path"]: artifact for artifact in artifacts}
    assert set(by_path) == {
        "README.md",
        "compatibility_summary.json",
        "evidence.meta.json",
    }
    for rel_path, artifact in by_path.items():
        path = EVIDENCE_DIR / rel_path
        assert path.is_file()
        assert artifact["sha256"] == _sha256(path)
        assert artifact["bytes"] == path.stat().st_size


def test_fa2_fallback_metadata_declares_summary_only_scope() -> None:
    metadata = _load_json(EVIDENCE_DIR / "evidence.meta.json")

    assert metadata["schema"] == "invarlock.public_evidence.meta.v1"
    assert metadata["evidence_class"] == "fa2_fallback_compatibility"
    assert metadata["artifact_paths"] == {
        "compatibility_summary": "compatibility_summary.json",
        "hash_inventory": "hash_inventory.json",
    }
    assert "flash-attn unavailable fallback" not in str(metadata["summary"]).lower()
