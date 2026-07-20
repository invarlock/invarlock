from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from invarlock import evidence_pack_verification as verification
from invarlock.evidence_pack_contract import (
    EVIDENCE_INPUT_IDENTITY_FORMAT,
    EVIDENCE_PATHS,
    EvidencePackError,
    canonical_json_bytes,
    sha256_digest,
)
from invarlock.evidence_pack_support import EvidencePackStatus
from tests.reporting.schema.test_evidence_pack_contract import _manifest


def test_manifest_validator_reports_closed_field_and_binding_failures() -> None:
    manifest = _manifest()
    assert verification._validate_manifest(manifest) == []

    mutations = [
        (lambda value: value.update(extra=True), "manifest fields"),
        (lambda value: value.update(format="other"), "manifest format"),
        (lambda value: value.update(comparison_id="bad value"), "comparison_id"),
        (lambda value: value.update(checksums_sha256="other"), "checksums_sha256"),
        (
            lambda value: value.update(checksums_sha256_digest="bad"),
            "checksums_sha256_digest",
        ),
        (lambda value: value.update(signing_key_fingerprint="bad"), "fingerprint"),
        (lambda value: value.update(inputs=[]), "manifest inputs"),
        (
            lambda value: value["inputs"].update(baseline=[]),
            "input baseline fields",
        ),
        (
            lambda value: value["inputs"]["baseline"].update(
                path="inputs/subject.json"
            ),
            "input baseline path",
        ),
        (
            lambda value: value["inputs"]["baseline"].update(digest="bad"),
            "input baseline digest",
        ),
        (lambda value: value.update(evidence=[]), "evidence roles"),
        (
            lambda value: value["evidence"].update(request=[]),
            "evidence request fields",
        ),
        (
            lambda value: value["evidence"]["request"].update(path="other"),
            "evidence request path",
        ),
        (
            lambda value: value["evidence"]["request"].update(digest="bad"),
            "evidence request digest",
        ),
        (lambda value: value.update(paired_records=[]), "paired_records fields"),
        (
            lambda value: value["paired_records"].update(path="other"),
            "paired_records path",
        ),
        (
            lambda value: value["paired_records"].update(digest="bad"),
            "paired_records digest",
        ),
        (
            lambda value: value["paired_records"].update(count=True),
            "paired_records count",
        ),
    ]
    for index, (mutate, message) in enumerate(mutations):
        candidate = copy.deepcopy(manifest)
        mutate(candidate)
        assert message in " ".join(verification._validate_manifest(candidate)), index


def test_manifest_validator_rejects_historical_unnamespaced_format_clearly() -> None:
    manifest = _manifest()
    manifest["format"] = "evidence-pack-v1"

    assert verification._validate_manifest(manifest) == [
        "unsupported manifest format 'evidence-pack-v1'; expected "
        "'invarlock/evidence-pack-v1'"
    ]


def test_json_object_loader_rejects_nonobject_oversized_and_symlink(
    tmp_path: Path,
) -> None:
    path = tmp_path / "value.json"
    path.write_text("[]\n", encoding="utf-8")
    payload, error, raw = verification._load_json_object(
        path, label="value", max_bytes=16
    )
    assert payload is None and "JSON object" in str(error) and raw == b"[]\n"

    path.write_bytes(b"x" * 17)
    payload, error, raw = verification._load_json_object(
        path, label="value", max_bytes=16
    )
    assert payload is None and "size limit" in str(error) and raw is None

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    path.unlink()
    path.symlink_to(target)
    payload, error, raw = verification._load_json_object(
        path, label="value", max_bytes=16
    )
    assert payload is None and "symlink" in str(error) and raw is None


def test_verifier_rejects_unsafe_pack_paths_and_missing_bound_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    with pytest.raises(EvidencePackError, match="unsafe"):
        verification._safe_pack_path(root, "../outside", label="payload")

    evidence = {
        role: {"path": relative, "digest": "sha256:" + "0" * 64}
        for role, relative in EVIDENCE_PATHS.items()
    }
    evidence["request"] = []
    loaded, errors = verification._load_bound_evidence(root, evidence)

    assert loaded == {}
    joined = "\n".join(errors)
    assert "manifest evidence request is missing" in joined
    assert "unavailable" in joined


def test_identity_verification_enforces_closed_canonical_manifest_binding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pack"
    identity_path = root / "inputs/baseline.json"
    identity_path.parent.mkdir(parents=True)
    material = "sha256:" + "a" * 64
    payload = {
        "format": EVIDENCE_INPUT_IDENTITY_FORMAT,
        "role": "baseline",
        "digest": material,
    }
    raw = canonical_json_bytes(payload)
    reference = {
        "path": "inputs/baseline.json",
        "digest": sha256_digest(raw),
        "material_digest": material,
    }
    identity_path.write_bytes(raw)

    observed, errors = verification._verify_identity(
        root, role="baseline", reference=reference
    )
    assert observed == payload
    assert errors == []

    mutations = [
        ({**payload, "extra": True}, reference, "fields are invalid", True),
        ({**payload, "format": "other"}, reference, "format is invalid", True),
        ({**payload, "role": "subject"}, reference, "role is invalid", True),
        ({**payload, "digest": "bad"}, reference, "digest is invalid", True),
        (
            payload,
            {**reference, "digest": "sha256:" + "b" * 64},
            "digest does not match manifest",
            True,
        ),
        (
            payload,
            {**reference, "material_digest": "sha256:" + "b" * 64},
            "material digest does not match manifest",
            True,
        ),
        (payload, reference, "not canonical JSON", False),
    ]
    for candidate, candidate_reference, message, canonical in mutations:
        candidate_raw = (
            canonical_json_bytes(candidate)
            if canonical
            else (json.dumps(candidate, indent=2) + "\n").encode()
        )
        identity_path.write_bytes(candidate_raw)
        _observed, errors = verification._verify_identity(
            root, role="baseline", reference=candidate_reference
        )
        assert message in " ".join(errors)

    identity_path.write_text("[]\n", encoding="utf-8")
    observed, errors = verification._verify_identity(
        root, role="baseline", reference=reference
    )
    assert observed is None
    assert "JSON object" in " ".join(errors)


def test_request_binding_rejects_invalid_comparison_artifacts_and_provider_identity() -> (
    None
):
    assert verification._request_input_binding_errors({}, {}, {}) == [
        "normalized request comparison is invalid"
    ]
    request = {
        "comparison": {
            "baseline": {"artifact": []},
            "subject": {"artifact": {"model_id": "subject"}},
        }
    }
    errors = verification._request_input_binding_errors(
        request,
        {},
        {"subject_provider_identity": b"{}"},
    )
    joined = "\n".join(errors)
    assert "baseline artifact is invalid" in joined
    assert "subject artifact identity could not be decoded" in joined


def test_bound_evidence_detects_digest_mismatch_while_loading_other_roles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pack"
    request_path = root / EVIDENCE_PATHS["request"]
    request_path.parent.mkdir(parents=True)
    request_path.write_text("{}\n", encoding="utf-8")
    evidence = {
        role: {"path": relative, "digest": "sha256:" + "0" * 64}
        for role, relative in EVIDENCE_PATHS.items()
    }

    loaded, errors = verification._load_bound_evidence(root, evidence)

    assert loaded["request"] == b"{}\n"
    assert "request digest does not match manifest" in errors


def test_top_level_verifier_collects_independent_anchor_and_manifest_errors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pack"
    root.mkdir()

    result = verification.verify_comparison_evidence(
        root,
        policy_path=None,
        expected_artifact_digests={"baseline": "bad"},
        expected_schedule_digest="bad",
        expected_runtime_digests={"baseline": "bad"},
        expected_signer_fingerprint="bad",
    )

    assert result.status is EvidencePackStatus.FORMAT
    assert result.payload["ok"] is False
    joined = "\n".join(result.payload["errors"])
    assert "policy_path anchor is required" in joined
    assert "exactly baseline and subject" in joined
    assert "schedule anchor must be a sha256" in joined
    assert "signer anchor" in joined
    assert "manifest.json is unavailable" in joined
    assert str(tmp_path) not in json.dumps(result.payload, sort_keys=True)


def test_top_level_verifier_rejects_invalid_policy_runtime_and_noncanonical_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    manifest = _manifest()
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    policy = tmp_path / "policy.json"
    policy.write_text("[]\n", encoding="utf-8")

    result = verification.verify_comparison_evidence(
        root,
        policy_path=policy,
        expected_artifact_digests={
            "baseline": "sha256:" + "a" * 64,
            "subject": "sha256:" + "b" * 64,
        },
        expected_schedule_digest="sha256:" + "d" * 64,
        expected_runtime_digests={
            "baseline": "bad",
            "subject": "sha256:" + "b" * 64,
        },
        expected_signer_fingerprint="sha256:" + "c" * 64,
    )

    joined = "\n".join(result.payload["errors"])
    assert "policy anchor must be a JSON object" in joined
    assert "baseline runtime anchor must be a sha256" in joined
    assert "manifest.json is not canonical JSON" in joined


def test_result_authenticity_and_policy_status_are_independent(tmp_path: Path) -> None:
    fingerprint = "sha256:" + "a" * 64
    anchors = {"signer_fingerprint": fingerprint}

    passing = verification._result(
        tmp_path,
        errors=[],
        signer_fingerprint=fingerprint,
        comparison_id="comparison",
        request_digest=None,
        anchors=anchors,
        status=EvidencePackStatus.INTEGRITY,
        policy_verdict="pass",
    )
    policy_fail = verification._result(
        tmp_path,
        errors=[],
        signer_fingerprint=fingerprint,
        comparison_id="comparison",
        request_digest=None,
        anchors=anchors,
        status=EvidencePackStatus.INTEGRITY,
        policy_verdict="fail",
    )
    mismatch = verification._result(
        tmp_path,
        errors=["tampered"],
        signer_fingerprint="sha256:" + "b" * 64,
        comparison_id=None,
        request_digest=None,
        anchors=anchors,
        status=EvidencePackStatus.SIGNATURE,
    )

    assert passing.status is EvidencePackStatus.OK
    assert passing.payload["authenticity"] == "pinned"
    assert policy_fail.status is EvidencePackStatus.REPORTS
    assert policy_fail.payload["integrity_ok"] is True
    assert mismatch.status is EvidencePackStatus.SIGNATURE
    assert mismatch.payload["authenticity"] == "mismatch"
    assert mismatch.payload["verification_scope"] == "not_verified"


def test_runtime_side_parser_rejects_nonobject_manifest() -> None:
    errors = verification._verify_runtime_side(
        Path("/unused"),
        side="baseline",
        loaded={"baseline_runtime_manifest": b"[]"},
        expected_runtime_digest="sha256:" + "a" * 64,
    )

    assert errors == ["baseline runtime manifest must be a JSON object"]


def test_manifest_observation_references_fail_closed() -> None:
    manifest = _manifest()
    invalid_sets = [
        ([], "between 1 and 64"),
        ({}, "between 1 and 64"),
        ({"bad id": {}}, "identifier is invalid"),
        ({"valid": []}, "fields are invalid"),
        (
            {
                "valid": {
                    "path": "wrong",
                    "digest": "bad",
                    "kind": "diagnostic",
                    "scope": "comparison",
                }
            },
            "path is invalid",
        ),
    ]
    for observations, message in invalid_sets:
        candidate = copy.deepcopy(manifest)
        candidate["observations"] = observations
        assert message in " ".join(verification._validate_manifest(candidate))


def test_observation_loader_rejects_invalid_request_and_manifest_entries(
    tmp_path: Path,
) -> None:
    arguments = {
        "pack_dir": tmp_path,
        "comparison_id": "comparison-1",
        "schedule_digest": "sha256:" + "a" * 64,
        "policy_digest": "sha256:" + "b" * 64,
        "artifact_digests": {
            "baseline": "sha256:" + "c" * 64,
            "subject": "sha256:" + "d" * 64,
        },
    }
    assert verification._verify_observations(
        references=None, requested={}, **arguments
    )[1] == ["normalized request observations are invalid"]
    assert verification._verify_observations(
        references=None, requested=[{}], **arguments
    )[1] == ["normalized request observation entry is invalid"]
    assert verification._verify_observations(
        references=None,
        requested=[{"id": "one"}, {"id": "one"}],
        **arguments,
    )[1] == ["normalized request observation entry is invalid"]
    assert verification._verify_observations(
        references=None, requested=[{"id": "one"}], **arguments
    )[1] == ["normalized request observations are missing from manifest"]
    assert verification._verify_observations(references=[], requested=[], **arguments)[
        1
    ] == ["manifest observations are invalid"]
    assert verification._verify_observations(
        references={"other": {}}, requested=[{"id": "one"}], **arguments
    )[1] == ["manifest observations do not match normalized request"]

    _verified, errors = verification._verify_observations(
        references={"one": []}, requested=[{"id": "one"}], **arguments
    )
    assert errors == ["manifest observation entry is invalid"]
    _verified, errors = verification._verify_observations(
        references={"one": {}}, requested=[{"id": "one"}], **arguments
    )
    assert errors == ["observation 'one' path is invalid"]
    _verified, errors = verification._verify_observations(
        references={"one": {"path": "../outside"}},
        requested=[{"id": "one"}],
        **arguments,
    )
    assert "path is unsafe" in errors[0]


def test_snapshot_failure_and_stability_results_preserve_anchors(
    tmp_path: Path,
) -> None:
    artifacts = {
        "baseline": "sha256:" + "a" * 64,
        "subject": "sha256:" + "b" * 64,
    }
    runtimes = {
        "baseline": "sha256:" + "c" * 64,
        "subject": "sha256:" + "d" * 64,
    }
    failure = verification._snapshot_failure_result(
        tmp_path / "pack",
        errors=["capture failed"],
        policy_path=None,
        expected_artifact_digests=artifacts,
        expected_schedule_digest="sha256:" + "e" * 64,
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint="sha256:" + "f" * 64,
        manifest_digest="sha256:" + "0" * 64,
    )
    assert failure.status is EvidencePackStatus.INTEGRITY
    assert failure.manifest_digest == "sha256:" + "0" * 64
    assert failure.payload["anchors"]["artifact_digests"] == artifacts  # type: ignore[index]

    augmented = verification._with_snapshot_errors(
        verification._result(
            tmp_path / "materialized",
            errors=[],
            signer_fingerprint=None,
            comparison_id="comparison",
            request_digest=None,
            anchors={},
            status=EvidencePackStatus.OK,
        ),
        pack_dir=tmp_path / "source-pack",
        errors=["changed", "changed"],
        manifest_digest="sha256:" + "1" * 64,
    )
    assert augmented.status is EvidencePackStatus.INTEGRITY
    assert augmented.payload["pack"] == "source-pack"
    assert augmented.payload["errors"] == ["changed"]
    assert augmented.payload["integrity_ok"] is False
