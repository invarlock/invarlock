"""The requested task, schedule and provider capabilities must describe one task."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from invarlock.evidence_pack import (
    EvidencePackError,
    publish_comparison_evidence,
    verify_comparison_evidence,
)
from invarlock.evidence_pack_contract import canonical_json_bytes, sha256_digest
from tests.evidence_packs.test_evidence_pack import (
    _publish,
    _rebind_and_resign_pack,
    _verification_anchors,
)


@pytest.mark.parametrize("mutation", ["request", "baseline", "subject"])
def test_resigned_task_contradiction_is_rejected_with_exact_request_anchor(
    tmp_path: Path, mutation: str
) -> None:
    pack, policy, fingerprint, runtimes, key, arguments = _publish(tmp_path)
    anchors = _verification_anchors(arguments)

    def verify():
        return verify_comparison_evidence(
            pack,
            policy_path=policy,
            **anchors,
            expected_runtime_digests=runtimes,
            expected_signer_fingerprint=fingerprint,
        )

    control = verify()
    assert control.payload["ok"] is True
    assert control.payload["errors"] == []
    if mutation == "request":
        path = pack / "request.json"
        request = json.loads(path.read_bytes())
        request["comparison"]["task"] = "vision_text"
        payload = canonical_json_bytes(request)
        path.chmod(0o600)
        path.write_bytes(payload)
        anchors["expected_request_digest"] = sha256_digest(payload)
    else:
        path = pack / f"providers/{mutation}/runtime-provider.receipt.json"
        receipt = json.loads(path.read_bytes())
        receipt["capabilities"]["tasks"] = ["vision_text"]
        payload = canonical_json_bytes(receipt, newline=False)
        path.chmod(0o600)
        path.write_bytes(payload)
        manifest_path = pack / f"providers/{mutation}/runtime.manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["runtime_provider"]["receipt"]["sha256"] = sha256_digest(
            payload
        ).removeprefix("sha256:")
        manifest_path.chmod(0o600)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
    _rebind_and_resign_pack(pack, key)

    result = verify()
    assert result.payload["ok"] is False
    assert result.payload["integrity_ok"] is False
    assert result.payload["policy_verdict"] is None
    expected = (
        "request task does not match the canonical schedule"
        if mutation == "request"
        else f"{mutation} runtime provider evidence does not declare task 'text_causal'"
    )
    assert expected in " ".join(result.payload["errors"])


@pytest.mark.parametrize("mutation", ["request", "baseline", "subject"])
def test_publication_rejects_task_contradiction_before_writing(
    tmp_path: Path, mutation: str
) -> None:
    _pack, _policy, _fingerprint, _runtimes, _key, arguments = _publish(tmp_path)
    if mutation == "request":
        arguments["normalized_request"]["comparison"]["task"] = "vision_text"
    else:
        field = f"{mutation}_evidence"
        side = arguments[field]
        receipt = json.loads(side.provider_receipt)
        receipt["capabilities"]["tasks"] = ["vision_text"]
        receipt_payload = canonical_json_bytes(receipt, newline=False)
        manifest = json.loads(side.runtime_manifest)
        manifest["runtime_provider"]["receipt"]["sha256"] = sha256_digest(
            receipt_payload
        ).removeprefix("sha256:")
        arguments[field] = replace(
            side,
            provider_receipt=receipt_payload,
            runtime_manifest=canonical_json_bytes(manifest),
        )
    destination = tmp_path / "contradictory-evidence"
    with pytest.raises(EvidencePackError, match="task"):
        publish_comparison_evidence(destination, **arguments)
    assert not destination.exists()
