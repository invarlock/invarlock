"""Independent verification and verifier-signed captured rejection statements."""

from __future__ import annotations

import base64
import re
import unicodedata
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.captured_contracts import (
    CAPTURED_PACK_FORMAT,
    CAPTURED_RECEIPT_FORMAT,
    CONTROL_LIMIT,
    DETECTOR_LIMIT,
    PAYLOAD_LIMIT,
    RECEIPT_LIMIT,
    CapturedContractError,
    CapturedIntegrityError,
    atomic_write,
    captured_snapshot,
    detect_manifest,
    json_object,
    load_payloads,
    read_file,
    require_outside,
    sha,
    validate_contract,
)
from invarlock.captured_normalization import (
    captured_comparison_id,
    comparison_policy_digest,
)
from invarlock.core.scoring import (
    InvalidScoringReference,
    MetricError,
    UnsupportedScoringEnvironment,
    require_scoring_environment,
)
from invarlock.evaluation_comparison.capacity import DEFAULT_MAX_BOOTSTRAP_DRAWS
from invarlock.evaluation_comparison.comparison import (
    LocalWorkBudgetExceeded,
    check_bootstrap_budget,
    compare_runs,
    metric_summaries,
)
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_receipt import ReceiptVerification


class CapturedVerificationIncomplete(ValueError):
    """Local operation did not complete; no integrity finding or receipt."""

    exit_code = 2


class CapturedVerificationError(ValueError):
    """Submitted evidence failed a check, optionally after a signed rejection."""

    def __init__(self, message: str, *, exit_code: int = 6) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _sha(payload: bytes) -> str:
    return sha(payload)


def _json(
    path: Path,
    label: str,
    limit: int = PAYLOAD_LIMIT,
    *,
    canonical: bool = True,
) -> tuple[dict[str, Any], bytes]:
    raw = read_file(path, limit)
    return json_object(raw, label, canonical=canonical), raw


def _key(path: Path, *, key_bytes: bytes | None = None) -> ed25519.Ed25519PrivateKey:
    try:
        raw = key_bytes if key_bytes is not None else read_file(path, CONTROL_LIMIT)
        if not isinstance(raw, bytes) or len(raw) > CONTROL_LIMIT:
            raise ValueError("verifier key exceeds byte limit")
        value = serialization.load_pem_private_key(raw, password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise CapturedVerificationIncomplete(
            "verifier key could not be loaded"
        ) from exc
    if not isinstance(value, ed25519.Ed25519PrivateKey):
        raise CapturedVerificationIncomplete("verifier key must be Ed25519")
    return value


def _assurance(policy: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "name": metric["name"],
            "slice": scope,
            "kind": metric["kind"],
            "scoring_assurance": "recorded"
            if metric["kind"] == "recorded"
            else "recomputed",
        }
        for scope in ["overall", *(subset["name"] for subset in policy["slices"])]
        for metric in policy["metrics"]
    ]


def _validate_receipt(receipt: dict[str, Any], policy: dict[str, Any]) -> None:
    validate_contract(receipt, receipt=True)
    statement = receipt["statement"]
    if statement["subject"]["run_digest"] != statement["anchors"]["subject_run_digest"]:
        raise CapturedContractError(
            "captured receipt subject differs from expected anchor"
        )
    if statement["verdict"]["integrity_ok"] and statement[
        "scoring_assurance"
    ] != _assurance(policy):
        raise CapturedContractError(
            "captured receipt assurance is incomplete or contradicts policy"
        )


def _write_receipt(
    path: Path,
    *,
    manifest_digest: str,
    expected_baseline_run: str,
    expected_subject_run: str,
    expected_request: str,
    expected_policy: str,
    expected_signer: str,
    verifier_identity: str,
    key: ed25519.Ed25519PrivateKey,
    profile_digest: str | None,
    replay_status: str,
    scoring_assurance: object,
    decision: str | None,
    policy_verdict: str | None,
    integrity_ok: bool | None,
    ok: bool,
    verification_status: int | None = None,
) -> None:
    public = key.public_key()
    statement = {
        "format": CAPTURED_RECEIPT_FORMAT,
        "kind": "captured",
        "pack_format": CAPTURED_PACK_FORMAT,
        "pack_manifest_digest": manifest_digest,
        "anchors": {
            "baseline_run_digest": expected_baseline_run,
            "subject_run_digest": expected_subject_run,
            "policy_digest": expected_policy,
            "request_digest": expected_request,
            "signer_fingerprint": expected_signer,
        },
        "subject": {"kind": "captured_run", "run_digest": expected_subject_run},
        "verifier": {
            "identity": verifier_identity,
            "signing_key_fingerprint": public_key_fingerprint(public),
            "trust_profile_digest": profile_digest,
        },
        "verification_scope": "captured_comparison",
        "replay_status": replay_status,
        "scoring_assurance": scoring_assurance,
        "verdict": {
            "ok": ok,
            "integrity_ok": integrity_ok,
            "decision": decision,
            "policy_verdict": policy_verdict,
            "verification_status": verification_status
            if verification_status is not None
            else (0 if ok else 7 if integrity_ok else 6),
        },
    }
    signature = {
        "format": "invarlock/evidence-verification-receipt-signature-v1",
        "algorithm": "ed25519",
        "public_key": {
            "encoding": "pem",
            "value": public.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii"),
        },
        "value": base64.b64encode(key.sign(canonical_json_bytes(statement))).decode(
            "ascii"
        ),
    }
    receipt = {"statement": statement, "signature": signature}
    validate_contract(receipt, receipt=True)
    raw = canonical_json_bytes(receipt)
    if len(raw) > RECEIPT_LIMIT:
        raise CapturedVerificationIncomplete("receipt exceeds byte limit")
    try:
        atomic_write(path, raw)
    except (OSError, ValueError) as exc:
        raise CapturedVerificationIncomplete("receipt_publication_failed") from exc


def verify_captured_evidence(
    evidence_path: Path,
    *,
    policy_path: Path,
    expected_baseline_run: str | None,
    expected_subject_run: str | None,
    expected_request_digest: str | None,
    expected_signer: str | None,
    receipt_path: Path | None,
    verifier_signing_key_path: Path | None,
    verifier_identity: str | None,
    trust_profile_digest: str | None = None,
    max_bootstrap_draws: int | None = DEFAULT_MAX_BOOTSTRAP_DRAWS,
    policy_bytes: bytes | None = None,
    verifier_signing_key_bytes: bytes | None = None,
) -> dict[str, Any]:
    evidence = Path(evidence_path)
    anchors = (
        expected_baseline_run,
        expected_subject_run,
        expected_request_digest,
        expected_signer,
    )
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"sha256:[a-f0-9]{64}", value) is None
        for value in anchors
    ):
        raise CapturedVerificationIncomplete(
            "captured run, request, and signer anchors are required"
        )
    assert isinstance(expected_baseline_run, str)
    assert isinstance(expected_subject_run, str)
    assert isinstance(expected_request_digest, str)
    assert isinstance(expected_signer, str)
    if (
        not isinstance(verifier_identity, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", verifier_identity)
        is None
    ):
        raise CapturedVerificationIncomplete("valid verifier identity is required")
    if trust_profile_digest is not None and (
        not isinstance(trust_profile_digest, str)
        or re.fullmatch(r"sha256:[a-f0-9]{64}", trust_profile_digest) is None
    ):
        raise CapturedVerificationIncomplete("trust profile digest is invalid")
    if receipt_path is None or (
        verifier_signing_key_path is None and verifier_signing_key_bytes is None
    ):
        raise CapturedVerificationIncomplete(
            "receipt and verifier signing key are required"
        )
    if max_bootstrap_draws is not None and (
        type(max_bootstrap_draws) is not int or max_bootstrap_draws < 0
    ):
        raise CapturedVerificationIncomplete(
            "max_bootstrap_draws must be a non-negative integer"
        )
    try:
        require_outside(Path(receipt_path), evidence)
        policy_raw = (
            policy_bytes
            if policy_bytes is not None
            else read_file(Path(policy_path), PAYLOAD_LIMIT)
        )
        if not isinstance(policy_raw, bytes) or len(policy_raw) > PAYLOAD_LIMIT:
            raise ValueError("independent policy exceeds byte limit")
        policy = json_object(policy_raw, "independent policy", canonical=False)
        policy_digest = comparison_policy_digest(policy)
        key = _key(
            Path(verifier_signing_key_path) if verifier_signing_key_path else Path("."),
            key_bytes=verifier_signing_key_bytes,
        )
    except (OSError, ValueError, TypeError) as exc:
        raise CapturedVerificationIncomplete(
            f"invalid verification inputs: {str(exc)[:240]}"
        ) from exc
    replay_status = "not_started"
    manifest_raw: bytes | None = None
    failure: CapturedContractError | None = None
    try:
        with captured_snapshot(evidence) as snapshot:
            manifest_raw = snapshot.manifest_bytes
            manifest, payloads, signer = load_payloads(snapshot)
            if manifest["authentication"] != "signed":
                raise CapturedIntegrityError(
                    "captured verification requires signed evidence"
                )
            if signer != expected_signer:
                raise CapturedIntegrityError(
                    "captured evidence signer does not match expected anchor"
                )
            for role, expected in (
                ("baseline", expected_baseline_run),
                ("subject", expected_subject_run),
                ("request", expected_request_digest),
                ("policy", policy_digest),
            ):
                if manifest["files"][role]["digest"] != expected:
                    raise CapturedIntegrityError(
                        f"captured {role} differs from expected anchor"
                    )
            if canonical_json_bytes(policy) != snapshot.files["inputs/policy.json"]:
                raise CapturedIntegrityError(
                    "captured policy differs from independent policy"
                )
            for metric in policy["metrics"]:
                require_scoring_environment(
                    metric["kind"],
                    metric["configuration"],
                    unicode_version=unicodedata.unidata_version,
                )
            # Charge all planned rows/scopes before invoking the scorer. Submitted
            # evidence cannot increase this caller-owned allowance.
            rows = payloads["baseline"]["records"]
            scope_count = len(rows) + sum(
                sum(
                    all(row["metadata"].get(k) == v for k, v in subset["where"].items())
                    for row in rows
                )
                for subset in policy["slices"]
            )
            check_bootstrap_budget(policy, scope_count, max_bootstrap_draws)
            replay_status = "failed"
            try:
                comparison = compare_runs(
                    payloads["baseline"],
                    payloads["subject"],
                    policy,
                    max_bootstrap_draws=max_bootstrap_draws,
                )
            except (EvaluationRecordsError, MetricError) as exc:
                if isinstance(
                    exc, (LocalWorkBudgetExceeded, UnsupportedScoringEnvironment)
                ):
                    raise
                if isinstance(
                    exc.__cause__,
                    (UnsupportedScoringEnvironment, LocalWorkBudgetExceeded),
                ):
                    raise CapturedVerificationIncomplete(exc.__cause__.reason) from exc
                if isinstance(exc.__cause__, OSError):
                    raise CapturedVerificationIncomplete(
                        "verification_io_error"
                    ) from exc
                if (
                    not isinstance(exc, InvalidScoringReference)
                    and exc.__cause__ is not None
                    and not isinstance(
                        exc.__cause__, (EvaluationRecordsError, MetricError)
                    )
                ):
                    raise CapturedVerificationIncomplete(
                        "verification_internal_error"
                    ) from exc
                raise CapturedIntegrityError(
                    f"captured replay failed: {str(exc)[:240]}"
                ) from exc
            replay_status = "completed"
            if (
                canonical_json_bytes(comparison)
                != snapshot.files["reports/evaluation.report.json"]
            ):
                raise CapturedIntegrityError(
                    "captured replay disagrees with stored comparison"
                )
            assurance = [
                {
                    field: metric[field]
                    for field in ("name", "slice", "kind", "scoring_assurance")
                }
                for metric in comparison["metrics"]
            ]
            if assurance != _assurance(policy):
                raise CapturedVerificationIncomplete("verification_internal_error")
            decision = comparison["decision"]
            summaries = metric_summaries(comparison)
    except CapturedContractError as exc:
        failure = exc
        manifest_raw = exc.manifest_bytes
    except CapturedVerificationIncomplete:
        raise
    except (UnsupportedScoringEnvironment, LocalWorkBudgetExceeded) as exc:
        raise CapturedVerificationIncomplete(exc.reason) from exc
    except OSError as exc:
        raise CapturedVerificationIncomplete("verification_io_error") from exc
    except Exception as exc:
        raise CapturedVerificationIncomplete("verification_internal_error") from exc

    # The snapshot's final stability check must finish before any receipt appears.
    if manifest_raw is not None:
        try:
            _write_receipt(
                Path(receipt_path),
                manifest_digest=sha(manifest_raw),
                expected_baseline_run=expected_baseline_run,
                expected_subject_run=expected_subject_run,
                expected_request=expected_request_digest,
                expected_policy=policy_digest,
                expected_signer=expected_signer,
                verifier_identity=verifier_identity,
                key=key,
                profile_digest=trust_profile_digest,
                replay_status=replay_status,
                scoring_assurance=None if failure else assurance,
                decision=None if failure else decision,
                policy_verdict=None
                if failure
                else "pass"
                if decision == "pass"
                else "fail",
                integrity_ok=failure is None,
                ok=failure is None and decision == "pass",
                verification_status=failure.exit_code
                if failure
                else 0
                if decision == "pass"
                else 7,
            )
        except Exception as exc:
            raise CapturedVerificationIncomplete("receipt_publication_failed") from exc
    if failure is not None:
        raise CapturedVerificationError(
            str(failure), exit_code=failure.exit_code
        ) from failure
    if manifest_raw is None:
        raise CapturedVerificationIncomplete("verification_internal_error")
    return {
        "format_version": "invarlock/evidence-pack-verify-v2",
        "kind": "captured",
        "ok": decision == "pass",
        "integrity_ok": True,
        "policy_verdict": "pass" if decision == "pass" else "fail",
        "decision": decision,
        "replay_status": "completed",
        "signed_receipt": Path(receipt_path).name,
        "pack_manifest_digest": sha(manifest_raw),
        "metric_summaries": summaries,
    }


def verify_captured_receipt(
    receipt_path: Path,
    pack_dir: Path,
    *,
    policy_path: Path,
    expected_baseline_run: str,
    expected_subject_run: str,
    expected_request_digest: str,
    expected_signer: str,
    expected_verifier_identity: str,
    expected_verifier_fingerprint: str,
    expected_profile_digest: str | None = None,
) -> ReceiptVerification:
    """Authenticate expectations separately from whether the examined pack met them."""
    errors: list[str] = []
    statement = None
    derived = None
    try:
        receipt = json_object(
            read_file(Path(receipt_path), RECEIPT_LIMIT),
            "verification receipt",
            canonical=False,
        )
        policy = _json(Path(policy_path), "independent policy", canonical=False)[0]
        policy_digest = comparison_policy_digest(policy)
        _validate_receipt(receipt, policy)
        statement = receipt["statement"]
        signature = receipt["signature"]
        manifest_raw = read_file(Path(pack_dir) / "manifest.json", DETECTOR_LIMIT)
        manifest = detect_manifest(manifest_raw)
        if statement["pack_manifest_digest"] != sha(manifest_raw):
            raise ValueError("receipt does not bind the examined manifest")
        expected_anchors = {
            "baseline_run_digest": expected_baseline_run,
            "subject_run_digest": expected_subject_run,
            "policy_digest": policy_digest,
            "request_digest": expected_request_digest,
            "signer_fingerprint": expected_signer,
        }
        if statement["anchors"] != expected_anchors:
            raise ValueError(
                "captured receipt anchors do not match caller expectations"
            )
        verifier = statement["verifier"]
        if (
            verifier["identity"] != expected_verifier_identity
            or verifier["trust_profile_digest"] != expected_profile_digest
        ):
            raise ValueError(
                "captured receipt verifier does not match caller expectations"
            )
        public = serialization.load_pem_public_key(
            signature["public_key"]["value"].encode("ascii")
        )
        if not isinstance(public, ed25519.Ed25519PublicKey):
            raise ValueError("captured receipt verifier key must be Ed25519")
        derived = public_key_fingerprint(public)
        if (
            derived != expected_verifier_fingerprint
            or verifier["signing_key_fingerprint"] != derived
        ):
            raise ValueError("captured receipt verifier key is unauthorized")
        public.verify(
            base64.b64decode(signature["value"], validate=True),
            canonical_json_bytes(statement),
        )
        if statement["verdict"]["integrity_ok"]:
            validate_contract(manifest)
            if (
                canonical_json_bytes(manifest) != manifest_raw
                or manifest["authentication"] != "signed"
                or manifest["signing_key_fingerprint"] != expected_signer
            ):
                raise ValueError(
                    "completed captured receipt contradicts manifest signer"
                )
            for role, anchor in (
                ("baseline", "baseline_run_digest"),
                ("subject", "subject_run_digest"),
                ("policy", "policy_digest"),
                ("request", "request_digest"),
            ):
                if manifest["files"][role]["digest"] != expected_anchors[anchor]:
                    raise ValueError(
                        "completed captured receipt contradicts manifest references"
                    )
            if manifest["request_digest"] != expected_request_digest or manifest[
                "comparison_id"
            ] != captured_comparison_id(
                request_digest=expected_request_digest,
                baseline_run_digest=expected_baseline_run,
                subject_run_digest=expected_subject_run,
                policy_digest=policy_digest,
            ):
                raise ValueError(
                    "completed captured receipt contradicts manifest identity"
                )
        # Rejected manifests are never used to select paths or derive expectations.
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        InvalidSignature,
        RecursionError,
    ) as exc:
        errors.append(str(exc)[:240] or "captured receipt signature is invalid")
    return ReceiptVerification(
        not errors,
        True,
        statement if not errors else None,
        derived if not errors else None,
        tuple(errors),
    )


__all__ = [
    "CAPTURED_PACK_FORMAT",
    "CAPTURED_RECEIPT_FORMAT",
    "CapturedVerificationError",
    "CapturedVerificationIncomplete",
    "verify_captured_evidence",
    "verify_captured_receipt",
]
