"""Explicit coarse Rust transactions and the authorized Python scorer boundary.

Selecting this backend requires the native package. There is no implicit
fallback and no extension discovery here: the caller supplies its registry.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

if TYPE_CHECKING:
    from invarlock.captured_evaluation import (
        CapturedEvaluationPreflightResult,
        CapturedEvaluationTransactionResult,
    )
    from invarlock.evaluator_qualification import EvaluatorQualificationResult
    from invarlock.evidence_pack_publication import EvidencePublication
    from invarlock.evidence_sets.verification import EvidenceSetVerification
    from invarlock.evidence_verification import EvidenceVerification
    from invarlock.judge_measurements.acceptance import JudgeVerificationResult
    from invarlock.trust_inputs import CapturedTrustInputs, TrustInputs


class NativeBackendUnavailable(RuntimeError):
    """The explicitly selected native backend cannot run in this environment."""


def require_native() -> Any:
    """Load the optional extension only after explicit native selection."""
    try:
        native = importlib.import_module("invarlock_native")
    except ImportError as exc:
        raise NativeBackendUnavailable(
            "The Rust backend requires the matching invarlock-native-experiment "
            "companion wheel for this Python and platform. Install that wheel "
            "before selecting backend='rust'."
        ) from exc
    if not callable(getattr(native, "dispatch_json", None)):
        raise NativeBackendUnavailable(
            "The installed Rust extension does not provide the transaction API. "
            "Install the matching invarlock-native-experiment companion wheel."
        )
    return native


def _bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def scorer_callback(registry: Any) -> Callable[[bytes], bytes]:
    """Adapt one explicitly authorized registry; native code owns replay checks."""
    from invarlock.core.scorer_extension import (
        AuthenticatedScorerRecord,
        ScorerReplayRequest,
        decode_scorer_binding,
        scorer_result_payload,
    )

    def replay(raw: bytes) -> bytes:
        request = json.loads(raw)
        result = registry.replay(
            ScorerReplayRequest(
                binding=decode_scorer_binding(request["binding"]),
                task=request["task"],
                input_kinds=tuple(request["input_kinds"]),
                output_kind=request["output_kind"],
                schedule_sha256=request["schedule_sha256"],
                records=tuple(
                    AuthenticatedScorerRecord(**record) for record in request["records"]
                ),
            )
        )
        return _bytes(scorer_result_payload(result))

    return replay


def transact(
    operation: str,
    config: dict[str, Any],
    *,
    scorer_registry: Any = None,
    policy_bytes: bytes | None = None,
    signing_key_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Dispatch one owned request, releasing the GIL except for explicit scorers."""
    invarlock_native = require_native()

    for label, raw in (
        ("independent policy", policy_bytes),
        ("verifier signing key", signing_key_bytes),
    ):
        if raw is not None and type(raw) is not bytes:
            raise TypeError(f"{label} bytes must be exact bytes")
    normalized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    return cast(
        dict[str, Any],
        json.loads(
            invarlock_native.dispatch_json(
                operation,
                _bytes(normalized),
                scorer_callback=scorer_callback(scorer_registry)
                if scorer_registry is not None
                else None,
                policy_bytes=policy_bytes,
                signing_key_bytes=signing_key_bytes,
            )
        ),
    )


def verify_captured_evidence(evidence_path: Path, **options: Any) -> dict[str, Any]:
    invarlock_native = require_native()

    from invarlock.captured_verification import (
        CapturedVerificationError,
        CapturedVerificationIncomplete,
    )

    policy_bytes = options.pop("policy_bytes", None)
    key_bytes = options.pop("verifier_signing_key_bytes", None)
    try:
        payload = cast(
            dict[str, Any],
            transact(
                "verify-captured",
                {"evidence_path": evidence_path, **options},
                policy_bytes=policy_bytes,
                signing_key_bytes=key_bytes,
            )["payload"],
        )
        # Restore the Python SDK collection type after decoding the JSON wire value.
        payload["metric_summaries"] = tuple(payload["metric_summaries"])
        return payload
    except invarlock_native.NativeVerificationError as exc:
        if exc.exit_code == 2:
            raise CapturedVerificationIncomplete(str(exc)) from exc
        error = CapturedVerificationError(str(exc), exit_code=exc.exit_code)
        error.receipt_written = exc.receipt_written
        raise error from exc


def verify_native_evidence(evidence_path: Path, **options: Any) -> EvidenceVerification:
    invarlock_native = require_native()

    from invarlock.evidence_verification import (
        EvidenceVerification,
        EvidenceVerificationError,
    )

    policy_bytes = options.pop("policy_bytes", None)
    key_bytes = options.pop("verifier_signing_key_bytes", None)
    registry = options.pop("scorer_registry", None)
    receipt = Path(options["receipt_path"]).resolve()
    try:
        outcome = transact(
            "verify-native",
            {"evidence_path": evidence_path, **options},
            policy_bytes=policy_bytes,
            signing_key_bytes=key_bytes,
            scorer_registry=registry,
        )
    except invarlock_native.NativeVerificationError as exc:
        raise EvidenceVerificationError(
            str(exc),
            exit_code=exc.exit_code,
            receipt_path=receipt if exc.receipt_written else None,
        ) from exc
    payload = outcome["payload"]
    if not payload["ok"]:
        errors = payload.get("errors")
        message = (
            "; ".join(str(item) for item in errors)
            if errors
            else (
                "authenticated evidence does not satisfy the approved policy"
                if payload.get("integrity_ok") is True
                and payload.get("policy_verdict") == "fail"
                else "evidence verification failed"
            )
        )
        details: tuple[str, ...] = ()
        if outcome.get("verified_report") is not None:
            from invarlock.evidence_explanation import core_policy_checks

            details = tuple(
                f"{check['name']}: {check['observed']}; required {check['required']}. {check['explanation']}"
                for check in core_policy_checks(outcome["verified_report"])
                if not check["passed"]
            )
        raise EvidenceVerificationError(
            message,
            exit_code=outcome["status"] or 1,
            payload=payload,
            details=details,
            receipt_path=receipt,
        )
    return EvidenceVerification(Path(evidence_path).resolve(), payload, receipt)


def verify_evidence_set(
    evidence_path: Path,
    *,
    recipient_policy: Path,
    receipt: Path | None = None,
    max_bootstrap_draws: int | None = 102_400_000,
) -> EvidenceSetVerification:
    from invarlock.evidence_sets.verification import EvidenceSetVerification

    outcome = transact(
        "verify-evidence-set",
        {
            "evidence_path": evidence_path,
            "recipient_policy_path": recipient_policy,
            "receipt_path": receipt,
            "max_bootstrap_draws": max_bootstrap_draws,
        },
    )
    return EvidenceSetVerification(outcome["payload"])


def verify_judge_evidence_with_policy(
    evidence_path: Path, recipient_policy_path: Path
) -> JudgeVerificationResult:
    from invarlock.judge_measurements.acceptance import _result_from_dict

    return _result_from_dict(
        transact(
            "verify-judge",
            {
                "evidence_path": evidence_path,
                "recipient_policy_path": recipient_policy_path,
            },
        )["payload"]
    )


def write_signed_judge_verification_receipt(
    evidence_path: Path, result: Any, receipt_path: Path, **options: Any
) -> str:
    return cast(
        str,
        transact(
            "write-judge-receipt",
            {
                "evidence_path": evidence_path,
                "result": result.to_dict(),
                "receipt_path": receipt_path,
                **options,
            },
        )["payload"]["verifier_fingerprint"],
    )


def load_trust_inputs(
    path: Path, *, verifier_key_bytes_override: bytes | None = None
) -> CapturedTrustInputs | TrustInputs:
    from types import MappingProxyType

    invarlock_native = require_native()

    from invarlock.trust_inputs import (
        CapturedTrustInputs,
        TrustInputs,
        TrustInputsError,
    )

    try:
        raw, policy, key, captured = invarlock_native.load_trust_snapshot(
            str(path), verifier_key_bytes_override
        )
    except invarlock_native.NativeVerificationError as exc:
        raise TrustInputsError(str(exc)) from exc
    config = json.loads(raw)
    values = {
        "policy_path": Path(config["policy_path"]),
        "policy_bytes": policy,
        "expected_request_digest": config["expected_request_digest"],
        "expected_signer_fingerprint": config["expected_signer"],
        "verifier_identity": config["verifier_identity"],
        "verifier_signing_key_path": Path(config["verifier_signing_key_path"]),
        "verifier_signing_key_bytes": key,
        "profile_digest": config["trust_profile_digest"],
    }
    if captured:
        return CapturedTrustInputs(
            **values,
            expected_run_digests=MappingProxyType(
                {
                    side: config[f"expected_{side}_run"]
                    for side in ("baseline", "subject")
                }
            ),
        )
    return TrustInputs(
        **values,
        expected_artifact_digests=MappingProxyType(
            {
                side: config[f"expected_{side}_artifact"]
                for side in ("baseline", "subject")
            }
        ),
        expected_runtime_digests=MappingProxyType(
            {
                side: config[f"expected_{side}_runtime"]
                for side in ("baseline", "subject")
            }
        ),
        expected_schedule_digest=config["expected_schedule"],
        allow_installed_scorers=config["allow_installed_scorers"],
    )


def qualify_evaluator_export(**paths: Any) -> EvaluatorQualificationResult:
    from invarlock.core.runtime_provider import RuntimeScoringRecord
    from invarlock.evaluator_qualification import EvaluatorQualificationResult

    outcome = transact("qualify-export", paths)["payload"]
    result = outcome["result"]
    return EvaluatorQualificationResult(
        profile_id=result["profile_id"],
        outcome=result["outcome"],
        authority=result["authority"],
        reason_codes=tuple(result["reason_codes"]),
        record_count=result["record_count"],
        scores=tuple(result["scores"]),
        mean_score=result["mean_score"],
        records_sha256=result["records_sha256"],
        **result["bindings"],
        _records=tuple(
            RuntimeScoringRecord(**record) for record in outcome["runtime_records"]
        ),
        _metric=outcome["metric"] or {"kind": "exact_match"},
    )


def _encoded(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode("ascii")


def runtime_material_config(
    *,
    normalized_request: Any,
    schedule: Any,
    policy_bytes: bytes,
    baseline_evidence: Any,
    subject_evidence: Any,
) -> dict[str, Any]:
    from invarlock.evidence_pack_contract import schedule_bytes

    result = {
        "normalized_request": normalized_request,
        "schedule_base64": _encoded(schedule_bytes(schedule)),
        "policy_base64": _encoded(policy_bytes),
    }
    fields = {
        "run_report": "run_report",
        "runtime_manifest": "runtime_manifest",
        "runtime_config": "runtime_config",
        "provider_identity": "artifact_identity",
        "provider_receipt": "provider_receipt",
        "scoring_observation": "scoring_observation",
    }
    for role, evidence in (
        ("baseline", baseline_evidence),
        ("subject", subject_evidence),
    ):
        result[role + "_material"] = {
            name: _encoded(getattr(evidence, field)) for name, field in fields.items()
        }
    return result


def derive_runtime_transaction(
    *, scorer_registry: Any = None, **options: Any
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        transact(
            "derive-runtime-transaction",
            runtime_material_config(**options),
            scorer_registry=scorer_registry,
        )["payload"],
    )


def publish_comparison_evidence(
    destination: Path, **options: Any
) -> EvidencePublication:
    from dataclasses import asdict

    from cryptography.hazmat.primitives import serialization

    from invarlock.evidence_pack_publication import EvidencePublication

    config = runtime_material_config(
        **{
            name: options[name]
            for name in (
                "normalized_request",
                "schedule",
                "policy_bytes",
                "baseline_evidence",
                "subject_evidence",
            )
        }
    )
    config.update(
        destination=str(destination),
        comparison_id=options["comparison_id"],
        identities={
            role: asdict(options[role])
            for role in (
                "baseline",
                "subject",
                "dataset",
                "baseline_runtime",
                "subject_runtime",
                "policy",
            )
        },
        expected_paired_records=options.get("expected_paired_records"),
    )
    config["observations"] = [
        {
            "id": value.observation_id,
            "kind": value.kind,
            "scope": value.scope,
            "payload_base64": _encoded(value.payload),
        }
        for value in options.get("observations", ())
    ]
    key = options["signing_key_path"]
    key_bytes = None
    if isinstance(key, Path):
        config["signing_key_path"] = str(key)
    else:
        key_bytes = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    payload = transact(
        "publish-native",
        config,
        scorer_registry=options.get("scorer_registry"),
        signing_key_bytes=key_bytes,
    )["payload"]
    return EvidencePublication(
        Path(payload["evidence_path"]),
        payload["pack_manifest_digest"],
        payload["policy_verdict"],
    )


def _captured_request_config(request: Any, **options: Any) -> dict[str, Any]:
    from collections.abc import Mapping
    from dataclasses import fields

    authored = {
        "format_version": request.format_version,
        "execution": {"mode": "captured"},
        "comparison": {"policy": request.policy.relative_to(request.root).as_posix()},
        "output": {"evidence": request.evidence.relative_to(request.root).as_posix()},
    }
    config = {"root": str(request.root), "authored_request": authored, **options}
    for side in ("baseline", "subject"):
        source = getattr(request, side)
        spec = {}
        for field in fields(source):
            value = getattr(source, field.name)
            if value is not None or (
                field.name == "artifact_digest" and source.service_identity is not None
            ):
                spec[field.name] = (
                    value.relative_to(request.root).as_posix()
                    if isinstance(value, Path)
                    else dict(value)
                    if isinstance(value, Mapping)
                    else value
                )
        authored["comparison"][side] = spec
        if source.adapter != "invarlock":
            from invarlock.captured_evaluation import _run

            config.setdefault("imported_runs", {})[side] = _run(request, side)
    if request.metric is not None:
        authored["comparison"]["metric"] = request.metric
    return config


@overload
def captured_request(
    request: Any, *, preflight: Literal[True], **options: Any
) -> CapturedEvaluationPreflightResult: ...


@overload
def captured_request(
    request: Any, *, preflight: Literal[False], **options: Any
) -> CapturedEvaluationTransactionResult: ...


def captured_request(
    request: Any, *, preflight: bool, **options: Any
) -> CapturedEvaluationPreflightResult | CapturedEvaluationTransactionResult:
    invarlock_native = require_native()

    from invarlock.captured_evaluation import (
        CapturedEvaluationError,
        CapturedEvaluationPreflightResult,
        CapturedEvaluationTransactionResult,
    )
    from invarlock.evaluation_comparison.comparison import metric_summaries

    try:
        prepared = transact(
            "prepare-captured", _captured_request_config(request, **options)
        )["payload"]
        values = dict(prepared["preflight"])
        for name in ("format_version", "kind", "ok"):
            values.pop(name, None)
        values["checks"] = tuple(values["checks"])
        result = CapturedEvaluationPreflightResult(**values)
        if preflight:
            return result
        published = transact(
            "publish-captured",
            {
                name: prepared[name]
                for name in ("baseline", "subject", "policy", "normalized_request")
            }
            | {"destination": str(request.evidence), **options},
        )["payload"]
    except invarlock_native.NativeVerificationError as exc:
        raise CapturedEvaluationError(str(exc)) from exc
    manifest = published["manifest"]
    return CapturedEvaluationTransactionResult(
        evidence_path=request.evidence,
        comparison_id=manifest["comparison_id"],
        baseline_run_digest=result.baseline_run_digest,
        subject_run_digest=result.subject_run_digest,
        policy_digest=result.policy_digest,
        authentication=manifest["authentication"],
        policy_verdict=published["comparison"]["decision"],
        pack_manifest_digest=published["pack_manifest_digest"],
        request_digest=result.request_digest,
        metric_summaries=metric_summaries(published["comparison"]),
    )


def runtime_authored_request(request: Any) -> dict[str, Any]:
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass

    def value(item):
        if isinstance(item, Path):
            return item.relative_to(request.root).as_posix()
        if isinstance(item, Mapping):
            return {key: value(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [value(child) for child in item]
        if is_dataclass(item):
            return {
                field.name: value(getattr(item, field.name))
                for field in fields(item)
                if getattr(item, field.name) is not None
            }
        return item

    result = {
        "format_version": request.format_version,
        "comparison": value(request.comparison),
        "execution": value(request.execution),
        "output": value(request.output),
    }
    if request.observations:
        result["observations"] = [
            {
                "id": row.observation_id,
                "kind": row.kind,
                "scope": row.scope,
                "path": value(row.path),
            }
            for row in request.observations
        ]
    return result


def prepare_runtime_inputs(request: Any) -> dict[str, Any]:
    import base64

    from invarlock.core.schedule_preparation import _schedule_from_native
    from invarlock.evidence_pack import EvidenceObservation

    outcome = cast(
        dict[str, Any],
        transact(
            "prepare-runtime-inputs",
            {
                "root": str(request.root),
                "authored_request": runtime_authored_request(request),
            },
        )["payload"],
    )
    outcome["schedule_object"] = _schedule_from_native(
        {"value": outcome["schedule"], "digest": outcome["schedule_digest"]}
    )
    outcome["policy_bytes"] = base64.b64decode(outcome["policy_base64"], validate=True)
    outcome["observations_objects"] = tuple(
        EvidenceObservation(
            row["id"],
            row["scope"],
            row["kind"],
            base64.b64decode(row["payload_base64"], validate=True),
        )
        for row in outcome["observations"]
    )
    return outcome


def normalize_runtime_request(
    request: Any, schedule: Any, observations: Any
) -> dict[str, Any]:
    from invarlock.evidence_pack_contract import sha256_digest

    return cast(
        dict[str, Any],
        transact(
            "normalize-runtime-request",
            {
                "authored_request": runtime_authored_request(request),
                "schedule": schedule.to_payload(),
                "observations": [
                    {
                        "id": row.observation_id,
                        "kind": row.kind,
                        "scope": row.scope,
                        "payload_digest": sha256_digest(row.payload),
                    }
                    for row in observations
                ],
            },
        )["payload"],
    )
