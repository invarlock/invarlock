"""Strict execution and replay of one directed runtime behavioral side."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator

from invarlock.core.runtime_provider import (
    RUNTIME_BEHAVIORAL_CLAIM_SET,
    ModelArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    RuntimeProvider,
    RuntimeProviderCapabilities,
    RuntimeProviderReceipt,
    RuntimeSession,
    ScoringObservation,
    artifact_identity_sha256,
    evaluate_runtime_claim_compatibility,
    load_runtime_behavioral_schedule,
)
from invarlock.core.runtime_provider.behavioral_schedule import (
    RuntimeBehavioralSchedule,
)
from invarlock.filesystem.atomic_directory import (
    AtomicDirectoryPublicationError,
    publish_directory_no_replace,
)
from invarlock.policy_pack import BEHAVIORAL_POLICY_PACK_FORMAT, verify_policy_pack
from invarlock.public_contracts import load_runtime_manifest_v2_schema
from invarlock.reporting.validation.runtime_behavioral_claim import (
    runtime_execution_settings_sha256,
)
from invarlock.reporting.validation.runtime_behavioral_observation import (
    verify_runtime_behavioral_observation,
)
from invarlock.runtime_behavioral_claim_receipt import (
    RuntimeBehavioralEvidenceBindings,
)
from invarlock.runtime_manifest_v2 import write_runtime_manifest_v2
from invarlock.runtime_provider_evidence import (
    ARTIFACT_IDENTITY_FILENAME,
    PROVIDER_RECEIPT_FILENAME,
    SCORING_OBSERVATION_FILENAME,
    PersistedRuntimeProviderEvidence,
    RuntimeProviderEvidencePaths,
    encode_scoring_observation,
    load_runtime_provider_evidence,
    write_runtime_provider_evidence,
)
from invarlock.runtime_security_helpers import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_MANIFEST_V2_VERSION,
    RUNTIME_VERIFIER_V2_CONTRACT_VERSION,
    RuntimeManifestExecution,
    RuntimeProviderManifestFiles,
    resolve_runtime_image,
    resolve_runtime_image_digest,
    strict_container_boundary_present,
)

from .contracts import (
    RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME,
    RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME,
    RuntimeBehavioralRole,
    RuntimeBehaviorError,
    RuntimeSideBundle,
    config_payload,
    report_payload,
    require_exact_payload,
    require_role,
)
from .io import (
    canonical_json_bytes,
    read_json_object,
    read_policy_pack_bounded,
    require_real_parent,
)

_SIDE_FILENAMES = frozenset(
    {
        RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME,
        RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME,
        RUNTIME_MANIFEST_FILENAME,
        ARTIFACT_IDENTITY_FILENAME,
        PROVIDER_RECEIPT_FILENAME,
        SCORING_OBSERVATION_FILENAME,
    }
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _image_ref_matches_digest(image_ref: object, image_digest: str) -> bool:
    if image_ref == image_digest:
        return True
    if not isinstance(image_ref, str):
        return False
    repository, separator, digest = image_ref.rpartition("@")
    return bool(repository and separator and digest == image_digest)


def _observed_execution(context: RuntimeExecutionContext) -> RuntimeManifestExecution:
    if not strict_container_boundary_present():
        raise RuntimeBehaviorError(
            "runtime behavioral side evidence requires actual container execution"
        )
    image_digest = context.container_image_digest
    if image_digest is None:
        raise RuntimeBehaviorError("strict side evidence requires a pinned outer image")
    observed_digest = resolve_runtime_image_digest()
    image_ref = resolve_runtime_image()
    if observed_digest != image_digest:
        raise RuntimeBehaviorError(
            "observed container image digest does not match runtime context"
        )
    if not _image_ref_matches_digest(image_ref, image_digest):
        raise RuntimeBehaviorError(
            "runtime image reference must embed the exact observed digest"
        )
    return RuntimeManifestExecution(
        execution_mode="container",
        container_execution=True,
        image_ref=image_ref,
        image_digest=image_digest,
        allow_network=False,
        allow_remote_code=False,
        allow_third_party_plugins=False,
    )


def _claim_binding(
    *,
    role: RuntimeBehavioralRole,
    capabilities: RuntimeProviderCapabilities,
    artifact_identity: ModelArtifactIdentity,
    schedule: RuntimeBehavioralSchedule,
    policy_pack: Mapping[str, object],
    image_digest: str,
    receipt: RuntimeProviderReceipt | None = None,
) -> str:
    errors = verify_policy_pack(dict(policy_pack))
    if errors:
        raise RuntimeBehaviorError("policy-pack-v3: " + "; ".join(errors))
    if policy_pack.get("format") != BEHAVIORAL_POLICY_PACK_FORMAT:
        raise RuntimeBehaviorError(
            f"runtime behavioral execution requires {BEHAVIORAL_POLICY_PACK_FORMAT}"
        )
    claim = policy_pack.get("behavioral_claim")
    if not isinstance(claim, Mapping) or (
        claim.get("claim_set") != RUNTIME_BEHAVIORAL_CLAIM_SET
    ):
        raise RuntimeBehaviorError(
            f"runtime behavioral execution requires {RUNTIME_BEHAVIORAL_CLAIM_SET}"
        )
    if claim.get("schedule_sha256") != schedule.schedule_sha256:
        raise RuntimeBehaviorError(
            "schedule does not match the directed policy binding"
        )
    compatibility = evaluate_runtime_claim_compatibility(
        RUNTIME_BEHAVIORAL_CLAIM_SET,
        baseline=capabilities,
        subject=capabilities,
    )
    if not compatibility.ok or "exact_match" not in compatibility.shared_metrics:
        raise RuntimeBehaviorError(" ".join(compatibility.errors))

    compatibility_block = policy_pack.get("compatibility")
    expected_dataset = (
        compatibility_block.get("dataset_identity")
        if isinstance(compatibility_block, Mapping)
        else None
    )
    if expected_dataset != schedule.dataset_identity.to_payload():
        raise RuntimeBehaviorError(
            "authenticated schedule dataset identity does not match policy-pack-v3"
        )
    required = claim.get("required_capabilities")
    if not isinstance(required, Mapping):
        raise RuntimeBehaviorError("policy pack is missing required capabilities")
    for key, available in (
        ("tasks", capabilities.tasks),
        ("metrics", capabilities.metrics),
        ("evidence_surfaces", capabilities.evidence_surfaces),
    ):
        values = required.get(key)
        if not isinstance(values, list) or any(
            value not in available for value in values
        ):
            raise RuntimeBehaviorError(
                f"provider lacks policy-required {key.replace('_', ' ')}"
            )
    metric_policy = claim.get("metric_policy")
    if not isinstance(metric_policy, Mapping) or (
        metric_policy.get("kind") != "exact_match"
    ):
        raise RuntimeBehaviorError(
            "runtime behavioral execution supports exact_match only"
        )

    binding = claim.get(role)
    if not isinstance(binding, Mapping):
        raise RuntimeBehaviorError(f"policy pack is missing directed {role} binding")
    observed: dict[str, object] = {
        "provider_name": capabilities.provider_name,
        "artifact_format": artifact_identity.artifact_format,
        "artifact_identity_sha256": artifact_identity_sha256(artifact_identity),
        "outer_image_digest": image_digest,
    }
    for field, value in observed.items():
        if binding.get(field) != value:
            raise RuntimeBehaviorError(
                f"{role} {field} does not match the directed policy binding"
            )
    if receipt is not None:
        if receipt.capabilities != capabilities:
            raise RuntimeBehaviorError(
                "session receipt capabilities do not match provider"
            )
        if receipt.artifact_identity != artifact_identity:
            raise RuntimeBehaviorError(
                "session receipt artifact does not match provider"
            )
        if receipt.outer_image_digest != image_digest:
            raise RuntimeBehaviorError("session receipt image does not match container")
        if receipt.execution_settings.allow_network:
            raise RuntimeBehaviorError("session receipt must bind offline execution")
        settings_sha256 = runtime_execution_settings_sha256(receipt.execution_settings)
        if binding.get("execution_settings_sha256") != settings_sha256:
            raise RuntimeBehaviorError(
                f"{role} execution_settings_sha256 does not match the directed "
                "policy binding"
            )
    policy_digest = policy_pack.get("policy_digest")
    if not isinstance(policy_digest, str):
        raise RuntimeBehaviorError("policy-pack-v3 is missing policy_digest")
    return policy_digest


def _side_bindings(
    *,
    report_bytes: bytes,
    manifest_bytes: bytes,
    evidence: PersistedRuntimeProviderEvidence,
) -> RuntimeBehavioralEvidenceBindings:
    return RuntimeBehavioralEvidenceBindings(
        runtime_manifest_sha256=_sha256(manifest_bytes),
        evaluation_report_sha256=_sha256(report_bytes),
        provider_receipt_sidecar_sha256=evidence.receipt_sha256,
        scoring_observation_sidecar_sha256=evidence.scoring_observation_sha256,
        artifact_identity_sidecar_sha256=evidence.artifact_identity_sha256,
    )


def _require_manifest_binding(
    *,
    manifest: Mapping[str, object],
    report_bytes: bytes,
    config_bytes: bytes,
    evidence: PersistedRuntimeProviderEvidence,
) -> None:
    errors = sorted(
        Draft202012Validator(load_runtime_manifest_v2_schema()).iter_errors(
            dict(manifest)
        ),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise RuntimeBehaviorError(
            f"runtime manifest v2 schema violation at {path}: {error.message}"
        )
    if manifest.get("manifest_version") != RUNTIME_MANIFEST_V2_VERSION or (
        manifest.get("verifier_contract_version")
        != RUNTIME_VERIFIER_V2_CONTRACT_VERSION
    ):
        raise RuntimeBehaviorError("runtime manifest v2 contract version is invalid")
    if manifest.get("execution_mode") != "container":
        raise RuntimeBehaviorError("runtime manifest v2 requires container execution")
    outer = manifest.get("outer_container")
    if not isinstance(outer, Mapping):
        raise RuntimeBehaviorError("runtime manifest v2 is missing outer_container")
    if any(
        outer.get(field) is not False
        for field in ("allow_network", "allow_remote_code", "allow_third_party_plugins")
    ):
        raise RuntimeBehaviorError("strict runtime manifest v2 permissions are invalid")
    image_digest = evidence.receipt.outer_image_digest
    if (
        image_digest is None
        or outer.get("container_execution") is not True
        or outer.get("image_digest") != image_digest
        or not _image_ref_matches_digest(outer.get("image_ref"), image_digest)
    ):
        raise RuntimeBehaviorError("runtime manifest v2 outer image binding is invalid")
    if evidence.receipt.execution_settings.allow_network:
        raise RuntimeBehaviorError("runtime provider receipt is not offline")

    expected_report = {
        "path": RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME,
        "filename": RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME,
        "sha256": _sha256(report_bytes),
    }
    expected_config = {
        "path": RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME,
        "sha256": _sha256(config_bytes),
        "source": "file",
    }
    expected_provider = {
        "receipt": {
            "filename": evidence.paths.receipt.name,
            "sha256": evidence.receipt_sha256,
        },
        "scoring_observation": {
            "filename": evidence.paths.scoring_observation.name,
            "sha256": evidence.scoring_observation_sha256,
        },
        "artifact_identity": {
            "filename": evidence.paths.artifact_identity.name,
            "sha256": evidence.artifact_identity_sha256,
        },
    }
    if manifest.get("report") != expected_report:
        raise RuntimeBehaviorError("runtime manifest v2 report binding is invalid")
    if manifest.get("config") != expected_config:
        raise RuntimeBehaviorError("runtime manifest v2 config binding is invalid")
    if manifest.get("runtime_provider") != expected_provider:
        raise RuntimeBehaviorError("runtime manifest v2 provider bindings are invalid")


def load_side_bundle(
    directory: Path,
    *,
    role: RuntimeBehavioralRole,
    schedule: RuntimeBehavioralSchedule,
    policy_pack: Mapping[str, object],
) -> RuntimeSideBundle:
    expected_role = require_role(role)
    root = Path(directory)
    if root.is_symlink():
        raise RuntimeBehaviorError("side bundle must be a real directory")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeBehaviorError("side bundle directory does not exist") from exc
    if not resolved.is_dir():
        raise RuntimeBehaviorError("side bundle must be a real directory")
    if {entry.name for entry in resolved.iterdir()} != _SIDE_FILENAMES:
        raise RuntimeBehaviorError(
            "side bundle must contain exactly the closed file set"
        )

    evidence = load_runtime_provider_evidence(
        RuntimeProviderEvidencePaths.in_directory(resolved)
    )
    image_digest = evidence.receipt.outer_image_digest
    if image_digest is None:
        raise RuntimeBehaviorError("strict side bundle is missing outer image digest")
    report_path = resolved / RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME
    config_path = resolved / RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME
    manifest_path = resolved / RUNTIME_MANIFEST_FILENAME
    report_bytes, report = read_json_object(report_path, label="side report")
    config_bytes, config = read_json_object(config_path, label="side config")
    manifest_bytes, manifest = read_json_object(manifest_path, label="runtime manifest")
    _require_manifest_binding(
        manifest=manifest,
        report_bytes=report_bytes,
        config_bytes=config_bytes,
        evidence=evidence,
    )
    policy_digest = _claim_binding(
        role=expected_role,
        capabilities=evidence.capabilities,
        artifact_identity=evidence.artifact_identity,
        schedule=schedule,
        policy_pack=policy_pack,
        image_digest=image_digest,
        receipt=evidence.receipt,
    )
    artifact_sha256 = artifact_identity_sha256(evidence.artifact_identity)
    metric_result = verify_runtime_behavioral_observation(
        cast(Mapping[str, object], json.loads(evidence.scoring_observation_bytes)),
        expected_provider_name=evidence.capabilities.provider_name,
        expected_artifact_identity_sha256=artifact_sha256,
        expected_batch=schedule.evaluation_batch(),
        metric="exact_match",
    )
    require_exact_payload(
        config,
        config_payload(
            role=expected_role,
            provider_name=evidence.capabilities.provider_name,
            artifact_sha256=artifact_sha256,
            schedule_sha256=schedule.schedule_sha256,
            policy_digest=policy_digest,
        ),
        label="side config",
    )
    require_exact_payload(
        report,
        report_payload(
            role=expected_role,
            provider_name=evidence.capabilities.provider_name,
            artifact_sha256=artifact_sha256,
            schedule_sha256=schedule.schedule_sha256,
            policy_digest=policy_digest,
            result=metric_result,
        ),
        label="side report",
    )
    return RuntimeSideBundle(
        role=expected_role,
        directory=resolved,
        report_path=report_path,
        config_path=config_path,
        manifest_path=manifest_path,
        evidence=evidence,
        metric_result=metric_result,
        bindings=_side_bindings(
            report_bytes=report_bytes,
            manifest_bytes=manifest_bytes,
            evidence=evidence,
        ),
    )


def run_side(
    *,
    role: RuntimeBehavioralRole,
    provider: RuntimeProvider,
    spec: ModelRuntimeSpec,
    context: RuntimeExecutionContext,
    schedule_path: Path,
    policy_pack_path: Path,
    output_directory: Path,
) -> RuntimeSideBundle:
    """Run and atomically publish one strictly verified directed side bundle."""

    directed_role = require_role(role)
    if not isinstance(spec, ModelRuntimeSpec):
        raise TypeError("spec must be ModelRuntimeSpec")
    if not isinstance(context, RuntimeExecutionContext):
        raise TypeError("context must be RuntimeExecutionContext")
    if not context.strict or context.allow_network:
        raise RuntimeBehaviorError(
            "runtime behavioral side evidence requires strict offline execution"
        )
    execution = _observed_execution(context)
    output = Path(output_directory)
    parent = require_real_parent(output)
    schedule = load_runtime_behavioral_schedule(Path(schedule_path))
    policy = read_policy_pack_bounded(Path(policy_pack_path))
    provider.validate_config(spec)
    if spec.settings.get("batch_size") != 1:
        raise RuntimeBehaviorError(
            "strict cross-runtime behavior requires batch_size=1 (one sequence per record)"
        )
    capabilities = provider.capabilities()
    if (
        capabilities.provider_name != provider.name
        or spec.provider_name != provider.name
    ):
        raise RuntimeBehaviorError(
            "provider, capabilities, and model spec do not agree"
        )
    artifact_identity = provider.identify_artifact(spec)
    artifact_sha256 = artifact_identity_sha256(artifact_identity)
    if context.artifact_identity_sha256 != artifact_sha256:
        raise RuntimeBehaviorError(
            "runtime context artifact identity does not match the exact model spec"
        )
    image_digest = cast(str, context.container_image_digest)
    policy_digest = _claim_binding(
        role=directed_role,
        capabilities=capabilities,
        artifact_identity=artifact_identity,
        schedule=schedule,
        policy_pack=policy,
        image_digest=image_digest,
    )

    session: RuntimeSession | None = None
    try:
        session = provider.open(spec, context)
        observation: ScoringObservation = session.score(schedule.evaluation_batch())
        receipt: RuntimeProviderReceipt = session.runtime_receipt()
    finally:
        if session is not None:
            session.close()
    _claim_binding(
        role=directed_role,
        capabilities=capabilities,
        artifact_identity=artifact_identity,
        schedule=schedule,
        policy_pack=policy,
        image_digest=image_digest,
        receipt=receipt,
    )
    metric_result = verify_runtime_behavioral_observation(
        cast(Mapping[str, object], json.loads(encode_scoring_observation(observation))),
        expected_provider_name=capabilities.provider_name,
        expected_artifact_identity_sha256=artifact_sha256,
        expected_batch=schedule.evaluation_batch(),
        metric="exact_match",
    )

    staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{output.name}.staging."))
    try:
        evidence = write_runtime_provider_evidence(
            staging,
            artifact_identity=artifact_identity,
            scoring_observation=observation,
            receipt=receipt,
            expected_outer_image_digest=image_digest,
        )
        report_path = staging / RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME
        config_path = staging / RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME
        report_path.write_bytes(
            canonical_json_bytes(
                report_payload(
                    role=directed_role,
                    provider_name=capabilities.provider_name,
                    artifact_sha256=artifact_sha256,
                    schedule_sha256=schedule.schedule_sha256,
                    policy_digest=policy_digest,
                    result=metric_result,
                )
            )
        )
        config_path.write_bytes(
            canonical_json_bytes(
                config_payload(
                    role=directed_role,
                    provider_name=capabilities.provider_name,
                    artifact_sha256=artifact_sha256,
                    schedule_sha256=schedule.schedule_sha256,
                    policy_digest=policy_digest,
                )
            )
        )
        write_runtime_manifest_v2(
            report_path,
            provider_files=RuntimeProviderManifestFiles(
                receipt=evidence.paths.receipt,
                scoring_observation=evidence.paths.scoring_observation,
                artifact_identity=evidence.paths.artifact_identity,
            ),
            config_path=config_path,
            execution=execution,
        )
        load_side_bundle(
            staging,
            role=directed_role,
            schedule=schedule,
            policy_pack=policy,
        )
        publish_directory_no_replace(staging, output)
    except AtomicDirectoryPublicationError as exc:
        raise RuntimeBehaviorError(
            "could not publish side bundle without clobber"
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return load_side_bundle(
        output,
        role=directed_role,
        schedule=schedule,
        policy_pack=policy,
    )


__all__ = ["load_side_bundle", "run_side"]
