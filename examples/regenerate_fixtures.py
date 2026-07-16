#!/usr/bin/env python3
"""Regenerate the deterministic imported-evidence fixtures in this directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from invarlock.core.runtime_provider import (
    HFSnapshotArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeBehavioralSchedule,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeScoringRecord,
    artifact_identity_sha256,
    build_runtime_behavioral_schedule_from_material,
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.evidence_pack_contract import sha256_digest
from invarlock.runtime_import_authoring import (
    write_runtime_import_paired_records,
    write_runtime_import_side,
)

BASELINE_RUNTIME = "sha256:" + "1" * 64
SUBJECT_RUNTIME = "sha256:" + "2" * 64
GENERATED_AT = "2026-07-16T00:00:00+00:00"


def _identity(model_id: str) -> HFSnapshotArtifactIdentity:
    return HFSnapshotArtifactIdentity(
        model_id=model_id,
        immutable_revision="b" * 40,
        checkpoint_tree_sha256="a" * 64,
        tokenizer_metadata_sha256="c" * 64,
    )


def _records(
    schedule: RuntimeBehavioralSchedule,
    outputs: Sequence[str],
) -> tuple[RuntimeScoringRecord, ...]:
    return tuple(
        RuntimeScoringRecord(
            record_id=record.record_id,
            input_sha256=record.input_sha256,
            status="ok",
            output_text=output,
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        )
        for record, output in zip(schedule.records, outputs, strict=True)
    )


def _copy_generated(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o644)


def regenerate(root: Path) -> None:
    policy_bytes = (root / "policy/acceptance.json").read_bytes()
    policy_digest = sha256_digest(policy_bytes)
    example_records = [
        {
            "record_id": f"record-{index:02d}",
            "input_text": f"Return token-{index:02d}",
            "expected_output": f"token-{index:02d}",
        }
        for index in range(50)
    ]
    expected_outputs = tuple(
        str(record["expected_output"]) for record in example_records
    )
    schedule = build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": "acceptance",
            "config_name": None,
            "revision": "e" * 40,
            "split": "validation",
        },
        records=example_records,
        task="text_causal",
    )
    capabilities = RuntimeProviderCapabilities(
        provider_name="hf_transformers",
        artifact_formats=("hf_snapshot",),
        tasks=("text_causal",),
        metrics=("exact_match",),
        execution_modes=("container",),
        required_extra=None,
        required_image=None,
    )
    plugin = RuntimeProviderPluginIdentity(
        name="hf_transformers",
        distribution="invarlock",
        distribution_version="test",
    )
    backend = RuntimeBackendIdentity(
        name="transformers",
        version="test",
        source_sha256="d" * 64,
        binary_sha256=None,
        build_sha256=None,
    )
    execution = RuntimeExecutionSettings(
        seed=0,
        context_length=128,
        batch_size=1,
        max_output_tokens=16,
        timeout_seconds=30,
        allow_network=False,
    )
    device = RuntimeDeviceFacts(device_kind="cpu", device_name="fixture-cpu")

    with tempfile.TemporaryDirectory(prefix="invarlock-example-fixtures-") as temporary:
        generated = Path(temporary)

        def side(
            name: str,
            *,
            role: Literal["baseline", "subject"],
            model_id: str,
            outputs: Sequence[str],
            runtime_digest: str,
        ):
            return write_runtime_import_side(
                generated / name,
                role=role,
                schedule=schedule,
                policy_digest=policy_digest,
                artifact_identity=_identity(model_id),
                records=_records(schedule, outputs),
                plugin=plugin,
                backend=backend,
                capabilities=capabilities,
                execution_settings=execution,
                device=device,
                runtime_image_ref=f"ghcr.io/invarlock/runtime@{runtime_digest}",
                runtime_image_digest=runtime_digest,
                generated_at_utc=GENERATED_AT,
            )

        baseline = side(
            "baseline",
            role="baseline",
            model_id="org/baseline",
            outputs=expected_outputs,
            runtime_digest=BASELINE_RUNTIME,
        )
        subject = side(
            "subject",
            role="subject",
            model_id="org/subject",
            outputs=expected_outputs,
            runtime_digest=SUBJECT_RUNTIME,
        )
        rejected = side(
            "rejected-subject",
            role="subject",
            model_id="org/subject",
            outputs=(*expected_outputs[:-1], "wrong"),
            runtime_digest=SUBJECT_RUNTIME,
        )
        accepted_pairs = write_runtime_import_paired_records(
            generated / "paired-records.json",
            schedule=schedule,
            metric="exact_match",
            baseline=baseline,
            subject=subject,
        )
        rejected_pairs = write_runtime_import_paired_records(
            generated / "rejected-paired-records.json",
            schedule=schedule,
            metric="exact_match",
            baseline=baseline,
            subject=rejected,
        )

        _copy_generated(
            generated / "paired-records.json", root / "import/paired-records.json"
        )
        _copy_generated(
            generated / "rejected-paired-records.json",
            root / "import/rejected-paired-records.json",
        )
        for name in ("baseline", "subject", "rejected-subject"):
            for filename in (
                "model-artifact.identity.json",
                "report.json",
                "run.yaml",
                "runtime-provider.receipt.json",
                "runtime-scoring.observation.json",
                "runtime.manifest.json",
            ):
                _copy_generated(
                    generated / name / filename,
                    root / "import" / name / filename,
                )

        schedule_bytes = canonical_runtime_behavioral_schedule_json(schedule)
        (root / "inputs/schedule.json").write_bytes(schedule_bytes)
        (root / "trusted-inputs/input-digests.json").write_text(
            json.dumps(
                {
                    "baseline_artifact": "sha256:"
                    + artifact_identity_sha256(_identity("org/baseline")),
                    "canonical_schedule": sha256_digest(schedule_bytes),
                    "subject_artifact": "sha256:"
                    + artifact_identity_sha256(_identity("org/subject")),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        assert accepted_pairs.payload["schedule_sha256"] == schedule.schedule_sha256
        assert rejected_pairs.payload["schedule_sha256"] == schedule.schedule_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the checked-in fixtures beneath the examples directory",
    )
    arguments = parser.parse_args()
    if not arguments.write:
        parser.error("--write is required")
    regenerate(Path(__file__).resolve().parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
