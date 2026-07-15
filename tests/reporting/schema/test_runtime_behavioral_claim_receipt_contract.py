from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import jsonschema
import pytest

from invarlock import public_contracts
from invarlock.core.runtime_provider.claims import RUNTIME_BEHAVIORAL_CLAIM_SET
from invarlock.reporting.validation.runtime_behavioral_claim import (
    RuntimeBehavioralClaimVerificationResult,
)
from invarlock.runtime_behavioral_claim_receipt import (
    RUNTIME_BEHAVIORAL_CLAIM_RECEIPT_FORMAT,
    RuntimeBehavioralClaimReceipt,
    RuntimeBehavioralClaimReceiptError,
    RuntimeBehavioralEvidenceBindings,
    build_runtime_behavioral_claim_receipt,
    canonical_runtime_behavioral_claim_receipt_json,
    runtime_behavioral_claim_receipt_sha256,
    verify_runtime_behavioral_claim_receipt,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bindings(role: str) -> RuntimeBehavioralEvidenceBindings:
    return RuntimeBehavioralEvidenceBindings(
        runtime_manifest_sha256=_sha256(f"{role}-manifest"),
        evaluation_report_sha256=_sha256(f"{role}-report"),
        provider_receipt_sidecar_sha256=_sha256(f"{role}-provider-receipt"),
        scoring_observation_sidecar_sha256=_sha256(f"{role}-observation"),
        artifact_identity_sidecar_sha256=_sha256(f"{role}-artifact"),
    )


def _verification() -> RuntimeBehavioralClaimVerificationResult:
    return RuntimeBehavioralClaimVerificationResult(
        ok=True,
        errors=(),
        claim_set=RUNTIME_BEHAVIORAL_CLAIM_SET,
        metric="exact_match",
        baseline_score=1.0,
        subject_score=0.75,
        regression=0.25,
        schedule_sha256=_sha256("schedule"),
        policy_digest="sha256:" + _sha256("policy"),
    )


def _receipt() -> RuntimeBehavioralClaimReceipt:
    return build_runtime_behavioral_claim_receipt(
        baseline=_bindings("baseline"),
        subject=_bindings("subject"),
        verification=_verification(),
    )


def _verify(payload: dict[str, Any]) -> RuntimeBehavioralClaimReceipt:
    return verify_runtime_behavioral_claim_receipt(
        payload,
        expected_baseline=_bindings("baseline"),
        expected_subject=_bindings("subject"),
        expected_verification=_verification(),
    )


def test_claim_receipt_round_trip_binds_portable_digest_only_evidence() -> None:
    receipt = _receipt()
    payload = receipt.to_payload()

    jsonschema.validate(
        payload,
        public_contracts.load_runtime_behavioral_claim_receipt_schema(),
    )
    assert _verify(payload) == receipt
    assert receipt.format_version == RUNTIME_BEHAVIORAL_CLAIM_RECEIPT_FORMAT
    assert receipt.claim_set == RUNTIME_BEHAVIORAL_CLAIM_SET
    assert receipt.metric == "exact_match"
    assert receipt.verdict == "pass"
    assert receipt.regression == 0.25

    encoded = canonical_runtime_behavioral_claim_receipt_json(receipt)
    assert encoded == json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert (
        runtime_behavioral_claim_receipt_sha256(receipt)
        == hashlib.sha256(encoded).hexdigest()
    )
    roles = cast(
        tuple[dict[str, object], dict[str, object]],
        (
            payload["baseline"],
            payload["subject"],
        ),
    )
    assert not any("path" in key or "host" in key for role in roles for key in role)


def test_claim_receipt_schema_is_closed_and_packaged_byte_identically() -> None:
    schema = public_contracts.load_runtime_behavioral_claim_receipt_schema()
    jsonschema.Draft202012Validator.check_schema(schema)

    repository = Path("contracts/runtime_behavioral_claim_receipt.schema.json")
    packaged = public_contracts.PACKAGE_CONTRACTS_ROOT.joinpath(repository.name)
    assert packaged.is_file()
    assert packaged.read_bytes() == repository.read_bytes()

    catalog = public_contracts.contract_catalog()
    assert catalog["runtime_behavioral_claim_receipt"]["path"] == (
        "contracts/runtime_behavioral_claim_receipt.schema.json"
    )
    assert public_contracts.public_subcontract_catalog()[
        "runtime_behavioral_claim_receipt"
    ] == {
        "version": RUNTIME_BEHAVIORAL_CLAIM_RECEIPT_FORMAT,
        "source": "contracts/runtime_behavioral_claim_receipt.schema.json",
        "compatibility": "closed_versioned_receipt",
    }
    assert (
        public_contracts.RUNTIME_BEHAVIORAL_CLAIM_RECEIPT_FORMAT_VERSION
        == RUNTIME_BEHAVIORAL_CLAIM_RECEIPT_FORMAT
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload["baseline"].update({"unexpected": True}),
        lambda payload: payload["baseline"].update(
            {"runtime_manifest_sha256": "non-digest-path-material"}
        ),
        lambda payload: payload.update({"baseline": payload["subject"]}),
        lambda payload: payload.update({"schedule_sha256": _sha256("other")}),
        lambda payload: payload.update({"policy_digest": "sha256:" + _sha256("other")}),
        lambda payload: payload.update({"metric": "multiple_choice_accuracy"}),
        lambda payload: payload.update({"subject_score": 0.5}),
        lambda payload: payload.update({"regression": 0.0}),
        lambda payload: payload.update({"verdict": "fail"}),
    ],
)
def test_claim_receipt_rejects_schema_and_cross_binding_tampering(mutate) -> None:
    payload = copy.deepcopy(_receipt().to_payload())
    mutate(payload)

    with pytest.raises(RuntimeBehavioralClaimReceiptError):
        _verify(payload)


@pytest.mark.parametrize(
    "verification",
    [
        replace(
            _verification(),
            ok=False,
            errors=("subject score is below the policy minimum",),
        ),
        replace(_verification(), metric="multiple_choice_accuracy"),
        replace(_verification(), regression=0.5),
        replace(_verification(), baseline_score=float("nan")),
        replace(_verification(), schedule_sha256="not-a-digest"),
        replace(_verification(), policy_digest=_sha256("bare-policy")),
    ],
)
def test_claim_receipt_builder_rejects_unverified_or_incoherent_replay(
    verification: RuntimeBehavioralClaimVerificationResult,
) -> None:
    with pytest.raises(RuntimeBehavioralClaimReceiptError):
        build_runtime_behavioral_claim_receipt(
            baseline=_bindings("baseline"),
            subject=_bindings("subject"),
            verification=verification,
        )


def test_claim_receipt_binding_type_rejects_host_or_path_material() -> None:
    with pytest.raises(RuntimeBehavioralClaimReceiptError, match="sha256"):
        RuntimeBehavioralEvidenceBindings(
            runtime_manifest_sha256="non-digest-host-material",
            evaluation_report_sha256=_sha256("report"),
            provider_receipt_sidecar_sha256=_sha256("receipt"),
            scoring_observation_sidecar_sha256=_sha256("observation"),
            artifact_identity_sidecar_sha256=_sha256("artifact"),
        )
