from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from invarlock.evidence_pack import verify_comparison_evidence
from invarlock.evidence_pack_contract import canonical_json_bytes, sha256_digest
from tests.evidence_packs.test_evidence_pack import (
    _rebind_and_resign_pack,
    _signing_key,
)

_REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("fixture", "field", "replacement", "expected_error"),
    [
        (
            "mistral-7b-weight-scale-hf",
            "checkpoint_tree_sha256",
            "0" * 64,
            "hf_transformers provider receipt does not match request setting "
            "'checkpoint_tree_sha256'",
        ),
        (
            "qwen2-vl-2b-7b-multimodal",
            "checkpoint_tree_sha256",
            "0" * 64,
            "hf_vision_text provider receipt does not match request setting "
            "'checkpoint_tree_sha256'",
        ),
        (
            "tinyllama-tensorrt-llm-checkpoints",
            "runner_binary_sha256",
            "0" * 64,
            "tensorrt_llm provider receipt does not match request setting "
            "'runner_binary_sha256'",
        ),
    ],
)
def test_strict_pack_verifier_rejects_resigned_provider_request_substitution(
    tmp_path: Path,
    fixture: str,
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    source = _REPOSITORY / "public_evidence" / "evidence" / fixture / "evidence"
    pack = tmp_path / "evidence"
    shutil.copytree(source, pack)
    request_path = pack / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    reference = json.loads(
        (
            _REPOSITORY
            / "scripts/release/reference_evidence"
            / f"{fixture}-anchors.json"
        ).read_bytes()
    )
    policy_bytes = (_REPOSITORY / reference["policy"]["path"]).read_bytes()
    assert sha256_digest(policy_bytes) == reference["policy"]["digest"]

    signing_key, signer_fingerprint = _signing_key(tmp_path)
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["signing_key_fingerprint"] = signer_fingerprint
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _rebind_and_resign_pack(pack, signing_key)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest["inputs"]

    def verify():
        return verify_comparison_evidence(
            pack,
            policy_path=None,
            policy_bytes=policy_bytes,
            expected_artifact_digests={
                "baseline": inputs["baseline"]["material_digest"],
                "subject": inputs["subject"]["material_digest"],
            },
            expected_schedule_digest=inputs["dataset"]["material_digest"],
            expected_runtime_digests={
                "baseline": inputs["baseline_runtime"]["material_digest"],
                "subject": inputs["subject_runtime"]["material_digest"],
            },
            expected_signer_fingerprint=signer_fingerprint,
            expected_request_digest=sha256_digest(request_path.read_bytes()),
        )

    control = verify()
    assert control.payload["ok"] is True
    assert control.payload["errors"] == []
    request["comparison"]["baseline"]["runtime"]["settings"][field] = replacement
    request_path.chmod(0o600)
    request_path.write_bytes(canonical_json_bytes(request))
    _rebind_and_resign_pack(pack, signing_key)
    result = verify()

    assert result.payload["ok"] is False
    assert expected_error in "\n".join(result.payload["errors"])
