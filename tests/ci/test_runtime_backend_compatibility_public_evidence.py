from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "public_evidence" / "runtime_backend_compatibility_cuda128"

EXPECTED_FAMILIES = {
    "cuda-bnb": ["hf_bnb"],
    "cuda-compressed-tensors": ["hf_ct"],
    "cuda-gptqmodel": ["hf_awq", "hf_gptq"],
    "cuda-hqq": ["hf_hqq"],
    "cuda-quanto": ["hf_quanto"],
    "cuda-torchao": ["hf_torchao"],
}

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


def test_cuda128_backend_compat_summary_is_public_safe_and_complete() -> None:
    summary = _load_json(EVIDENCE_DIR / "compatibility_summary.json")

    assert (
        summary["schema"]
        == "invarlock.runtime_backend_compatibility.cuda128.summary.v1"
    )
    assert summary["status"] == "completed"
    assert summary["evidence_scope"] == (
        "runtime backend build/import smoke compatibility only; "
        "no model-quality or assurance claim"
    )
    assert summary["validation_environment"] == "CUDA-capable validation host"
    assert summary["raw_logs_published"] is False
    assert summary["weights_vendored"] is False

    families = summary["families"]
    assert isinstance(families, list)
    assert {entry["family"] for entry in families if isinstance(entry, dict)} == set(
        EXPECTED_FAMILIES
    )

    for entry in families:
        assert isinstance(entry, dict)
        family = entry["family"]
        assert entry["adapter_smoke"] == EXPECTED_FAMILIES[family]
        assert entry["build_rc"] == 0
        assert entry["smoke_rc"] == 0
        assert entry["gpu_required"] is True
        assert str(entry["smoke_result"]).startswith("quant runtime image imports ok:")
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", str(entry["image_id"]))
        assert isinstance(entry["image_size_bytes"], int)
        assert entry["image_size_bytes"] > 0

        lock = REPO_ROOT / str(entry["requirements_lock"])
        assert lock.is_file()
        assert str(entry["build_command"]).startswith(
            "examples/integrations/_runtime_images/build_example_runtime_image.sh "
        )
        assert str(entry["smoke_command"]).startswith(
            "examples/integrations/_runtime_images/smoke_example_runtime_image.sh "
        )

    serialized = json.dumps(summary, sort_keys=True)
    for pattern in PRIVATE_TEXT_PATTERNS:
        assert pattern not in serialized


def test_cuda128_backend_compat_hash_inventory_matches_public_files() -> None:
    inventory = _load_json(EVIDENCE_DIR / "hash_inventory.json")

    assert inventory["schema"] == (
        "invarlock.runtime_backend_compatibility.cuda128.hash_inventory.v1"
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


def test_cuda128_backend_compat_metadata_declares_narrow_scope() -> None:
    metadata = _load_json(EVIDENCE_DIR / "evidence.meta.json")

    assert metadata["schema"] == "invarlock.public_evidence.meta.v1"
    assert metadata["evidence_class"] == "runtime_backend_compatibility"
    assert metadata["artifact_paths"] == {
        "compatibility_summary": "compatibility_summary.json",
        "hash_inventory": "hash_inventory.json",
    }
    assert "invarlock evaluate" not in str(metadata["generated_by"])
