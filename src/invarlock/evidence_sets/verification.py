"""Fresh recipient verification of two independently authorized components."""

from __future__ import annotations

import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.captured_contracts import atomic_write, read_file, secure_directory, sha
from invarlock.captured_reporting import _load as load_captured_snapshot
from invarlock.captured_verification import verify_captured_evidence
from invarlock.evaluation_comparison.capacity import DEFAULT_MAX_BOOTSTRAP_DRAWS
from invarlock.evaluation_comparison.comparison import _check_policy
from invarlock.evaluation_records.cases import case_set_digest
from invarlock.evaluation_records.io import run_digest
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.evidence_sets.contracts import (
    CONTROL_LIMIT,
    DETERMINISTIC_KINDS,
    INDEX_FILE,
    MEMBERS,
    SCOPE,
    STATISTICAL_SCOPE,
    EvidenceSetError,
    check_statements,
    load_index,
    load_policy,
    member_path,
    read_object,
    require_external,
    validate,
)
from invarlock.judge_measurements.acceptance import (
    verify_judge_evidence,
    write_signed_judge_verification_receipt,
)
from invarlock.public_contracts import (
    load_evidence_set_verification_schema,
    load_judge_measurement_recipient_policy_schema,
)
from invarlock.trust_inputs import CapturedTrustInputs, load_trust_inputs

RESULT_LIMIT = 1024 * 1024


@dataclass(frozen=True)
class EvidenceSetVerification:
    payload: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return bool(self.payload["accepted"])

    @property
    def exit_code(self) -> int:
        return 0 if self.accepted else 7 if self.payload["verified"] else 4

    def as_json(self) -> str:
        return canonical_json_bytes(self.payload).decode()


def _empty_member(kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "verified": False,
        "authenticated": False,
        "replayed": False,
        "accepted": False,
        "decision": None,
        "statement_sha256": None,
        "trust_profile_sha256": None,
        "receipt": None,
        "errors": [],
    }


def _empty_result() -> dict[str, Any]:
    return {
        "format_version": "invarlock/evidence-set-verification-v1",
        "kind": "evidence_set",
        "scope": SCOPE,
        "statistical_scope": STATISTICAL_SCOPE,
        "ok": False,
        "verified": False,
        "authenticated": False,
        "replayed": False,
        "accepted": False,
        "decision": None,
        "index_sha256": None,
        "recipient_policy_sha256": None,
        "shared_inputs": None,
        "shared_inputs_verified": False,
        "members": {
            "deterministic": _empty_member("captured"),
            "judge": _empty_member("judge"),
        },
        "errors": [],
    }


def shared_captured_inputs(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Derive complete input identity from a validated captured snapshot."""
    baseline, subject = payloads["baseline"], payloads["subject"]
    case_sets = [
        case_set_digest(
            {
                "format": "invarlock/evaluation-case-set-v1",
                "cases": [
                    {key: row[key] for key in ("id", "input", "expected", "metadata")}
                    for row in run["records"]
                ],
            }
        )
        for run in (baseline, subject)
    ]
    if case_sets[0] != case_sets[1]:
        raise EvidenceSetError("component baseline and subject case sets differ")
    return {
        "baseline_run_sha256": run_digest(baseline),
        "subject_run_sha256": run_digest(subject),
        "case_set_sha256": case_sets[0],
        "subject_artifact_sha256": subject["artifact_digest"],
    }


def require_deterministic_policy(policy: dict[str, Any]) -> None:
    _check_policy(policy)
    if any(metric["kind"] not in DETERMINISTIC_KINDS for metric in policy["metrics"]):
        raise EvidenceSetError(
            "evidence set requires recomputed deterministic captured metrics; recorded scores are unsupported"
        )


def _verify(
    root: Path, policy_path: Path, result: dict[str, Any], maximum: int | None
) -> None:
    index, index_raw = load_index(root)
    policy, policy_raw = load_policy(policy_path, root)
    result.update(
        index_sha256=sha(index_raw),
        recipient_policy_sha256=sha(policy_raw),
        shared_inputs=policy["shared_inputs"],
    )
    if sha(index_raw) != policy["index_sha256"]:
        raise EvidenceSetError("evidence set index differs from recipient pin")
    check_statements(root, index)
    paths = {
        name: member_path(root, index["members"][name]["path"]) for name in MEMBERS
    }
    profiles = {
        name: member_path(policy_path.parent, policy["members"][name]["trust_profile"])
        for name in MEMBERS
    }
    profile_values, profile_raws = {}, {}
    for name in MEMBERS:
        require_external(profiles[name], root)
        profile_values[name], profile_raws[name] = read_object(profiles[name])
        expected = policy["members"][name]["trust_profile_sha256"]
        if sha(profile_raws[name]) != expected:
            raise EvidenceSetError(f"{name} trust profile differs from recipient pin")
        result["members"][name].update(
            statement_sha256=index["members"][name]["statement_sha256"],
            trust_profile_sha256=expected,
        )
    captured_profile = load_trust_inputs(profiles["deterministic"])
    if not isinstance(captured_profile, CapturedTrustInputs):
        raise EvidenceSetError("deterministic member requires a captured trust profile")
    if captured_profile.profile_digest != sha(
        canonical_json_bytes(profile_values["deterministic"])
    ):
        raise EvidenceSetError("captured trust profile changed while loading")
    for path in (
        captured_profile.policy_path,
        captured_profile.verifier_signing_key_path,
    ):
        require_external(path, root)
    deterministic_policy = parse_json_bytes(
        captured_profile.policy_bytes, label="deterministic policy"
    )
    if not isinstance(deterministic_policy, dict):
        raise EvidenceSetError("deterministic policy must be an object")
    require_deterministic_policy(deterministic_policy)
    shared = policy["shared_inputs"]
    for role in ("baseline", "subject"):
        if captured_profile.expected_run_digests[role] != shared[f"{role}_run_sha256"]:
            raise EvidenceSetError(
                f"captured {role} differs from the shared recipient pin"
            )
    manifest, payloads, _, _ = load_captured_snapshot(paths["deterministic"])
    if (
        sha(canonical_json_bytes(manifest))
        != index["members"]["deterministic"]["statement_sha256"]
    ):
        raise EvidenceSetError("captured statement changed while loading")
    if shared_captured_inputs(payloads) != shared:
        raise EvidenceSetError(
            "captured runs, cases, or subject artifact differ from shared recipient pins"
        )
    expected_envelope, expected_envelope_raw = read_object(
        paths["judge"] / "envelope.json"
    )
    if sha(expected_envelope_raw) != index["members"]["judge"]["statement_sha256"]:
        raise EvidenceSetError("judge statement changed before verification")
    judge_policy = profile_values["judge"]
    validate(
        judge_policy,
        load_judge_measurement_recipient_policy_schema(),
        label="judge recipient policy",
    )
    if judge_policy.get("intended_subject") != shared["subject_artifact_sha256"]:
        raise EvidenceSetError(
            "judge intended subject differs from shared recipient pin"
        )
    for field in ("baseline_run_sha256", "subject_run_sha256", "case_set_sha256"):
        if judge_policy.get("bindings", {}).get(field) != shared[field]:
            raise EvidenceSetError(f"judge {field} differs from shared recipient pin")
    # Child verification is authoritative. Earlier reads only reject mismatches cheaply.
    with tempfile.TemporaryDirectory(
        prefix="invarlock-evidence-set-", dir=Path(tempfile.gettempdir()).resolve()
    ) as temporary:
        receipt_path = Path(temporary) / "captured.json"
        try:
            captured_result = verify_captured_evidence(
                paths["deterministic"],
                policy_path=captured_profile.policy_path,
                expected_baseline_run=captured_profile.expected_run_digests["baseline"],
                expected_subject_run=captured_profile.expected_run_digests["subject"],
                expected_request_digest=captured_profile.expected_request_digest,
                expected_signer=captured_profile.expected_signer_fingerprint,
                receipt_path=receipt_path,
                verifier_signing_key_path=captured_profile.verifier_signing_key_path,
                verifier_identity=captured_profile.verifier_identity,
                trust_profile_digest=captured_profile.profile_digest,
                max_bootstrap_draws=maximum,
                policy_bytes=captured_profile.policy_bytes,
                verifier_signing_key_bytes=captured_profile.verifier_signing_key_bytes,
            )
        except (OSError, ValueError) as exc:
            result["members"]["deterministic"]["errors"] = [
                str(exc)[:1024] or "captured verification failed"
            ]
            if receipt_path.exists():
                result["members"]["deterministic"]["receipt"] = read_file(
                    receipt_path, CONTROL_LIMIT
                ).decode("utf-8")
            raise
        captured_receipt = read_file(receipt_path, CONTROL_LIMIT).decode("utf-8")
    if (
        captured_result["pack_manifest_digest"]
        != index["members"]["deterministic"]["statement_sha256"]
    ):
        raise EvidenceSetError("verified captured statement differs from index pin")
    result["members"]["deterministic"].update(
        verified=True,
        authenticated=True,
        replayed=True,
        accepted=captured_result["ok"],
        decision=captured_result["decision"],
        receipt=captured_receipt,
    )
    judge = verify_judge_evidence(
        paths["judge"], recipient_policy_path=profiles["judge"]
    )
    with tempfile.TemporaryDirectory(
        prefix="invarlock-evidence-set-judge-",
        dir=Path(tempfile.gettempdir()).resolve(),
    ) as temporary:
        judge_receipt_path = Path(temporary) / "judge.json"
        write_signed_judge_verification_receipt(
            paths["judge"],
            judge,
            judge_receipt_path,
            recipient_policy_path=profiles["judge"],
            verifier_identity=captured_profile.verifier_identity,
            verifier_signing_key_path=captured_profile.verifier_signing_key_path,
            verifier_signing_key_bytes=captured_profile.verifier_signing_key_bytes,
        )
        judge_receipt = read_file(judge_receipt_path, CONTROL_LIMIT).decode("utf-8")
    if (
        judge.recipient_policy_sha256
        != sha(canonical_json_bytes(judge_policy, newline=False))[7:]
    ):
        raise EvidenceSetError("judge trust profile changed during verification")
    if (
        judge.envelope_sha256 is not None
        and judge.envelope_sha256
        != sha(canonical_json_bytes(expected_envelope, newline=False))[7:]
    ):
        raise EvidenceSetError("verified judge statement differs from index pin")
    result["members"]["judge"].update(
        verified=judge.verified,
        authenticated=judge.authenticated,
        replayed=judge.replayed,
        accepted=judge.accepted,
        decision=judge.decision,
        receipt=judge_receipt,
        errors=list(judge.errors),
    )
    if not judge.verified:
        raise EvidenceSetError(
            "judge component did not authenticate and replay: "
            + "; ".join(judge.errors)
        )
    if judge.bindings is None or any(
        getattr(judge.bindings, field) != shared[field]
        for field in ("baseline_run_sha256", "subject_run_sha256", "case_set_sha256")
    ):
        raise EvidenceSetError(
            "verified judge shared inputs differ from recipient pins"
        )
    if judge.intended_subject != shared["subject_artifact_sha256"]:
        raise EvidenceSetError(
            "verified judge subject artifact differs from recipient pin"
        )
    # An advisory-only judge cannot become a required accepted component.
    analysis_policy, _ = read_object(paths["judge"] / "analysis_policy.json")
    if (
        sha(canonical_json_bytes(analysis_policy, newline=False))[7:]
        != judge.bindings.analysis_policy_sha256
    ):
        raise EvidenceSetError("judge analysis policy changed during composition")
    if analysis_policy.get("decision_role") != "required":
        raise EvidenceSetError(
            "evidence set requires a required judge policy; advisory promotion is forbidden"
        )
    check_statements(root, index)
    if (
        read_file(root / INDEX_FILE, CONTROL_LIMIT) != index_raw
        or read_file(policy_path, CONTROL_LIMIT) != policy_raw
    ):
        raise EvidenceSetError("composition inputs changed during verification")
    for name in MEMBERS:
        if read_file(profiles[name], CONTROL_LIMIT) != profile_raws[name]:
            raise EvidenceSetError(
                "component trust profile changed during verification"
            )
    decisions = {member["decision"] for member in result["members"].values()}
    decision = (
        "regression"
        if "regression" in decisions
        else "insufficient_evidence"
        if "insufficient_evidence" in decisions
        else "pass"
    )
    accepted = all(member["accepted"] for member in result["members"].values())
    result.update(
        verified=True,
        authenticated=True,
        replayed=True,
        shared_inputs_verified=True,
        decision=decision,
        accepted=accepted,
        ok=accepted,
    )


def verify_evidence_set(
    evidence: Path,
    *,
    recipient_policy: Path,
    receipt: Path | None = None,
    max_bootstrap_draws: int | None = DEFAULT_MAX_BOOTSTRAP_DRAWS,
) -> EvidenceSetVerification:
    """Verify both components freshly and optionally retain their complete receipts.

    The result is a local computation, not a new signed attestation. Embedded
    component receipts preserve their distinct existing scopes.
    """
    result = _empty_result()
    root, policy_path = Path(evidence).absolute(), Path(recipient_policy).absolute()
    try:
        if max_bootstrap_draws is not None and (
            type(max_bootstrap_draws) is not int or max_bootstrap_draws < 0
        ):
            raise EvidenceSetError("max_bootstrap_draws must be a non-negative integer")
        if receipt is not None:
            require_external(receipt, root)
            if Path(receipt).exists() or Path(receipt).is_symlink():
                raise EvidenceSetError("composition receipt destination already exists")
        with ExitStack() as stack:
            stack.enter_context(secure_directory(root))
            stack.enter_context(secure_directory(policy_path.parent))
            _verify(root, policy_path, result, max_bootstrap_draws)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.update(
            ok=False,
            verified=False,
            accepted=False,
            decision=None,
            shared_inputs_verified=False,
        )
        result["errors"] = [str(exc)[:1024] or "evidence set verification failed"]
    validate(
        result,
        load_evidence_set_verification_schema(),
        label="evidence set verification",
    )
    raw = canonical_json_bytes(result)
    if len(raw) > RESULT_LIMIT:
        raise EvidenceSetError("composition verification result exceeds byte limit")
    if receipt is not None:
        require_external(receipt, root)
        atomic_write(Path(receipt), raw)
    return EvidenceSetVerification(result)


def verify_stored_evidence_set_result(
    stored_result: Path,
    evidence: Path,
    *,
    recipient_policy: Path,
    max_bootstrap_draws: int | None = DEFAULT_MAX_BOOTSTRAP_DRAWS,
) -> EvidenceSetVerification:
    """Stored success is never authority; require exact fresh recomputation."""
    actual = verify_evidence_set(
        evidence,
        recipient_policy=recipient_policy,
        max_bootstrap_draws=max_bootstrap_draws,
    )
    try:
        claimed, _ = read_object(Path(stored_result), maximum=RESULT_LIMIT)
        validate(
            claimed,
            load_evidence_set_verification_schema(),
            label="stored composition result",
        )
        if canonical_json_bytes(claimed) != canonical_json_bytes(actual.payload):
            raise EvidenceSetError(
                "stored composition result differs from fresh verification"
            )
    except (OSError, ValueError) as exc:
        value = dict(actual.payload)
        value.update(ok=False, verified=False, accepted=False, errors=[str(exc)[:1024]])
        return EvidenceSetVerification(value)
    return actual
