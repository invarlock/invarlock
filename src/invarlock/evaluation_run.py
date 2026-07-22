"""Execute both sides of a paired evaluation transaction."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from invarlock.core.evaluation_request import ComparisonSideRequest, EvaluationRequest
from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    parse_runtime_behavioral_schedule_json,
    validate_runtime_evaluation_inputs,
)
from invarlock.evaluation_runtime import RuntimeResourceResolver, RuntimeSideRole
from invarlock.evidence_pack import RuntimeSideEvidence
from invarlock.runtime_behavior.transaction import run_evidence_side
from invarlock.runtime_provider_evidence import RuntimeProviderEvidencePaths


@dataclass(frozen=True)
class EvaluationRunResult:
    """Typed evidence and exact runtime identities for both executed sides."""

    baseline: RuntimeSideEvidence
    subject: RuntimeSideEvidence
    baseline_runtime_digest: str
    subject_runtime_digest: str


class RuntimeComparisonExecutor(Protocol):
    """Execute two isolated sides and return only their immutable side evidence."""

    def execute(
        self,
        request: EvaluationRequest,
        *,
        registry: CoreRegistry,
        schedule_bytes: bytes,
        policy_digest: str,
    ) -> EvaluationRunResult: ...


def _runtime_side_evidence(bundle) -> RuntimeSideEvidence:
    return RuntimeSideEvidence(
        run_report=bundle.report_path.read_bytes(),
        runtime_manifest=bundle.manifest_path.read_bytes(),
        runtime_config=bundle.config_path.read_bytes(),
        artifact_identity=bundle.evidence.artifact_identity_bytes,
        provider_receipt=bundle.evidence.receipt_bytes,
        scoring_observation=bundle.evidence.scoring_observation_bytes,
    )


def load_runtime_side_evidence(directory: Path) -> RuntimeSideEvidence:
    """Read the exact six files emitted by one closed side worker."""

    root = Path(directory)
    if root.is_symlink():
        raise ValueError("runtime side output directory must not be a symlink")
    provider_paths = RuntimeProviderEvidencePaths.in_directory(root)
    expected = {
        "run report": root / "report.json",
        "runtime manifest": root / "runtime.manifest.json",
        "runtime config": root / "run.yaml",
        "artifact identity": provider_paths.artifact_identity,
        "provider receipt": provider_paths.receipt,
        "scoring observation": provider_paths.scoring_observation,
    }
    entries = {path.name for path in root.iterdir()} if root.is_dir() else set()
    expected_names = {path.name for path in expected.values()}
    if entries != expected_names:
        missing = sorted(expected_names - entries)
        unexpected = sorted(entries - expected_names)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            "runtime side output is not the closed six-file bundle: "
            + "; ".join(details)
        )
    payloads: dict[str, bytes] = {}
    for label, path in expected.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"runtime side {label} must be a regular file")
        payloads[label] = path.read_bytes()
    return RuntimeSideEvidence(
        run_report=payloads["run report"],
        runtime_manifest=payloads["runtime manifest"],
        runtime_config=payloads["runtime config"],
        artifact_identity=payloads["artifact identity"],
        provider_receipt=payloads["provider receipt"],
        scoring_observation=payloads["scoring observation"],
    )


def execute_runtime_comparison(
    request: EvaluationRequest,
    *,
    registry: CoreRegistry,
    resource_resolver: RuntimeResourceResolver,
    schedule_bytes: bytes,
    policy_digest: str,
) -> EvaluationRunResult:
    """Prepare and execute both sides under caller-owned runtime bindings."""

    evidence: dict[RuntimeSideRole, RuntimeSideEvidence] = {}
    runtime_digests: dict[RuntimeSideRole, str] = {}
    try:
        validated_schedule = parse_runtime_behavioral_schedule_json(
            schedule_bytes.decode("utf-8")
        )
    except (UnicodeError, ValueError) as exc:
        raise ValueError(
            "canonical schedule is invalid before runtime execution"
        ) from exc
    with tempfile.TemporaryDirectory(prefix="invarlock-evaluate-") as temporary:
        work = Path(temporary)
        schedule_path = work / "schedule.json"
        schedule_path.write_bytes(schedule_bytes)
        sides: tuple[tuple[RuntimeSideRole, ComparisonSideRequest], ...] = (
            ("baseline", request.comparison.baseline),
            ("subject", request.comparison.subject),
        )
        for role, side in sides:
            provider = registry.get_runtime_provider(side.runtime.provider)
            spec = ModelRuntimeSpec(
                provider_name=side.runtime.provider,
                model_id=side.artifact.model_id,
                settings=side.runtime.settings,
            )
            resources = resource_resolver.resolve(
                request_root=request.root,
                role=role,
                side=side,
                provider=provider,
            )
            validate_runtime_evaluation_inputs(
                provider, spec, resources, validated_schedule
            )
            context = provider.prepare_execution(spec, resources)
            bundle = run_evidence_side(
                role=role,
                provider=provider,
                spec=spec,
                context=context,
                schedule_path=schedule_path,
                policy_digest=policy_digest,
                output_directory=work / role,
                metric=request.comparison.collection_metric,
                _validated_schedule=validated_schedule,
            )
            runtime_digest = bundle.evidence.receipt.outer_image_digest
            if runtime_digest is None:
                raise ValueError(f"{role} provider receipt lacks an outer image digest")
            evidence[role] = _runtime_side_evidence(bundle)
            runtime_digests[role] = runtime_digest
    return EvaluationRunResult(
        baseline=evidence["baseline"],
        subject=evidence["subject"],
        baseline_runtime_digest=runtime_digests["baseline"],
        subject_runtime_digest=runtime_digests["subject"],
    )


__all__ = [
    "EvaluationRunResult",
    "RuntimeComparisonExecutor",
    "execute_runtime_comparison",
    "load_runtime_side_evidence",
]
