#!/usr/bin/env python3
"""Run the maintained Hugging Face vision-text tutorial transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from huggingface_hub import snapshot_download

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if exc.name != "examples":
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]
from examples.integrations.trust_material import (
    create_trust_material,
    load_external_key,
    validate_new_trust_root,
)
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    HFSnapshotArtifactIdentity,
    artifact_identity_sha256,
)
from invarlock.core.runtime_provider.behavioral_schedule import (
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evidence_pack_contract import canonical_json_bytes, normalize_digest
from invarlock.evidence_pack_integrity import public_key_fingerprint

_SEED = 20_260_721
IMAGE_CONTENT_ID = "tutorial_color_grid"
_SNAPSHOT_REPOSITORY_METADATA = (".gitattributes", "LICENSE", "README.md")


@dataclass(frozen=True)
class ModelProfile:
    """Immutable model coordinates already qualified by the vision add-in."""

    model_id: str
    revision: str
    checkpoint_tree_sha256: str
    tokenizer_metadata_sha256: str
    processor_metadata_sha256: str


MODEL_PROFILES = {
    "baseline": ModelProfile(
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        revision="895c3a49bc3fa70a340399125c650a463535e71c",
        checkpoint_tree_sha256=(
            "sha256:b364c4120702e7824d2b5b33da2309c41369ade1b46780bebdabfb4e6b343727"
        ),
        tokenizer_metadata_sha256=(
            "343a2aec7d93185db270af42c7f4ad787c0862a99e3bd5075afcf62397d68382"
        ),
        processor_metadata_sha256=(
            "51fb54a1bd5dad68b2a827a7e62b267e775e0b8f2df2f1fb0e37139f8b0ef4f3"
        ),
    ),
    "subject": ModelProfile(
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        revision="eed13092ef92e448dd6875b2a00151bd3f7db0ac",
        checkpoint_tree_sha256=(
            "sha256:1fdd524df34f2a4f91c1c976af93c7d054f9cae66cc888b14697736e870717ac"
        ),
        tokenizer_metadata_sha256=(
            "343a2aec7d93185db270af42c7f4ad787c0862a99e3bd5075afcf62397d68382"
        ),
        processor_metadata_sha256=(
            "51fb54a1bd5dad68b2a827a7e62b267e775e0b8f2df2f1fb0e37139f8b0ef4f3"
        ),
    ),
}


@dataclass(frozen=True)
class VisionExamplePaths:
    """Files produced by preparation and consumed by the public commands."""

    root: Path
    evaluation: Path
    request: Path
    records: Path
    schedule: Path
    content: Path
    policy: Path
    evidence: Path
    trusted_inputs: Path
    evidence_key: Path
    verifier_key: Path
    receipt: Path
    html_report: Path


def _paths(
    root: Path,
    *,
    evidence_key: Path | None = None,
    trust_root: Path | None = None,
) -> VisionExamplePaths:
    evaluation = root / "evaluation"
    verifier = trust_root or root / "verifier"
    receipt_root = root / "verifier-output" if trust_root is not None else verifier
    return VisionExamplePaths(
        root=root,
        evaluation=evaluation,
        request=evaluation / "request.yaml",
        records=evaluation / "inputs" / "records.jsonl",
        schedule=evaluation / "inputs" / "runtime-behavioral-schedule.json",
        content=evaluation / "inputs" / "content",
        policy=evaluation / "inputs" / "acceptance.json",
        evidence=evaluation / "evidence",
        trusted_inputs=verifier / "trusted-inputs.json",
        evidence_key=evidence_key or root / "keys" / "evidence-signer.pem",
        verifier_key=(
            verifier / "verifier.pem"
            if trust_root is not None
            else verifier / "keys" / "verifier.pem"
        ),
        receipt=receipt_root / "verification.receipt.json",
        html_report=root / "comparison-report.html",
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def tutorial_image_png() -> bytes:
    """Return a deterministic four-color PNG without depending on image tooling."""

    width = height = 96
    colors = (
        ((220, 38, 38), (38, 92, 220)),
        ((34, 160, 78), (244, 196, 38)),
    )
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(
                colors[1 if y >= height // 2 else 0][1 if x >= width // 2 else 0]
            )
    return b"\x89PNG\r\n\x1a\n" + b"".join(
        (
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


IMAGE_SHA256 = hashlib.sha256(tutorial_image_png()).hexdigest()


def _write_private_key(path: Path) -> str:
    key = ed25519.Ed25519PrivateKey.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return public_key_fingerprint(key.public_key())


def _make_worker_readable(root: Path) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def download_models(models_root: Path) -> None:
    """Materialize and authenticate both immutable model snapshots."""

    models_root.mkdir(parents=True, exist_ok=True)
    for role, profile in MODEL_PROFILES.items():
        destination = models_root / role
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"model destination already exists: {destination}")
        snapshot_download(
            repo_id=profile.model_id,
            revision=profile.revision,
            local_dir=destination,
            ignore_patterns=_SNAPSHOT_REPOSITORY_METADATA,
        )
        _make_worker_readable(destination)
        observed = checkpoint_tree_sha256(destination)
        if observed != profile.checkpoint_tree_sha256:
            raise RuntimeError(
                f"{role} checkpoint tree digest mismatch: expected "
                f"{profile.checkpoint_tree_sha256}, observed {observed}"
            )


def _write_model_placeholders(models_root: Path) -> None:
    models_root.mkdir(parents=True)
    for role, profile in MODEL_PROFILES.items():
        destination = models_root / role
        destination.mkdir()
        destination.joinpath(".model_id").write_bytes(
            canonical_json_bytes(
                {
                    "model_id": profile.model_id,
                    "revision": profile.revision,
                    "checkpoint_tree_sha256": profile.checkpoint_tree_sha256,
                    "weights_materialized": False,
                }
            )
        )


def _settings(profile: ModelProfile) -> dict[str, object]:
    return {
        "batch_size": 1,
        "checkpoint_tree_sha256": profile.checkpoint_tree_sha256,
        "context_length": 1024,
        "immutable_revision": profile.revision,
        "max_output_tokens": 8,
        "offline": True,
        "processor_metadata_sha256": profile.processor_metadata_sha256,
        "seed": _SEED,
        "timeout_seconds": 900,
        "tokenizer_metadata_sha256": profile.tokenizer_metadata_sha256,
    }


def _tutorial_records(image_bytes: bytes) -> tuple[dict[str, object], ...]:
    questions = (
        ("bottom-left", "green"),
        ("bottom-right", "yellow"),
        ("top-left", "red"),
        ("top-right", "blue"),
    )
    common = {
        "content_bytes": len(image_bytes),
        "content_id": IMAGE_CONTENT_ID,
        "content_media_type": "image/png",
        "content_sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    return tuple(
        {
            **common,
            "expected": expected,
            "id": f"color-grid-{position}",
            "prompt": (
                "Look at the four-color grid. Reply with exactly one lowercase "
                f"word: what color is the {position} square?"
            ),
        }
        for position, expected in questions
    )


def prepare_workspace(
    root: Path,
    *,
    runtime_image_digest: str,
    materialize_models: bool,
    evidence_signing_key: Path | None = None,
    verifier_signing_key: Path | None = None,
    trust_root: Path | None = None,
    ephemeral_trust_root: bool = True,
) -> tuple[VisionExamplePaths, dict[str, str]]:
    """Author the complete tutorial closure, optionally materializing models."""

    external_trust = any(
        value is not None
        for value in (evidence_signing_key, verifier_signing_key, trust_root)
    )
    if external_trust and not all(
        value is not None
        for value in (evidence_signing_key, verifier_signing_key, trust_root)
    ):
        raise ValueError(
            "evidence key, verifier key, and trust root must be supplied together"
        )
    if external_trust and ephemeral_trust_root:
        raise ValueError("external trust material cannot use ephemeral mode")
    if not external_trust and not ephemeral_trust_root:
        raise ValueError(
            "caller-owned evidence/verifier keys and trust root are required"
        )
    root = root.expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(
            f"workspace already exists: {root}; choose a new disposable path"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    paths = _paths(
        root,
        evidence_key=(
            evidence_signing_key.expanduser().absolute() if external_trust else None
        ),
        trust_root=(trust_root.expanduser().absolute() if external_trust else None),
    )
    try:
        paths.content.mkdir(parents=True)
        evidence_key_bytes: bytes | None = None
        verifier_key_bytes: bytes | None = None
        if external_trust:
            assert evidence_signing_key is not None
            assert verifier_signing_key is not None
            assert trust_root is not None
            trust_root = validate_new_trust_root(trust_root, transaction_root=root)
            evidence_key_path, evidence_key_bytes, evidence_signer = load_external_key(
                evidence_signing_key,
                transaction_root=root,
                label="evidence signing key",
            )
            _verifier_key_path, verifier_key_bytes, verifier_signer = load_external_key(
                verifier_signing_key,
                transaction_root=root,
                label="verifier signing key",
            )
            if evidence_signer == verifier_signer:
                raise ValueError("evidence and verifier signing keys must be distinct")
            paths = _paths(
                root,
                evidence_key=evidence_key_path,
                trust_root=trust_root.expanduser().absolute()
                if trust_root is not None
                else None,
            )
        else:
            paths.evidence_key.parent.mkdir(parents=True)
            paths.verifier_key.parent.mkdir(parents=True)
            (paths.trusted_inputs.parent / "policy").mkdir(parents=True)
        paths.receipt.parent.mkdir(parents=True, exist_ok=True)
        if materialize_models:
            download_models(paths.evaluation / "models")
        else:
            _write_model_placeholders(paths.evaluation / "models")

        image_bytes = tutorial_image_png()
        paths.content.joinpath(IMAGE_CONTENT_ID).write_bytes(image_bytes)
        records = _tutorial_records(image_bytes)
        records_bytes = b"".join(canonical_json_bytes(record) for record in records)
        paths.records.write_bytes(records_bytes)
        records_sha256 = hashlib.sha256(records_bytes).hexdigest()
        schedule = prepare_local_evaluation_schedule_bytes(
            LocalDatasetRequest(
                path=paths.records,
                sha256=records_sha256,
                name="qwen2-vl-color-grid-tutorial",
                split="tutorial",
                input_field="prompt",
                expected_output_field="expected",
                id_field="id",
                content_role="image",
                content_id_field="content_id",
                content_sha256_field="content_sha256",
                content_byte_length_field="content_bytes",
                content_media_type_field="content_media_type",
            ),
            records_bytes,
            task="vision_text_generation",
        )
        paths.schedule.write_bytes(canonical_runtime_behavioral_schedule_json(schedule))

        policy = {
            "resolved_policy": {
                "metrics": {
                    "exact_match": {
                        "delta_min_pp": -100.0,
                        "maximum_interval_width_pp": 200.0,
                        "minimum_record_count": 4,
                    }
                }
            }
        }
        policy_bytes = canonical_json_bytes(policy)
        paths.policy.write_bytes(policy_bytes)
        if not external_trust:
            paths.trusted_inputs.parent.joinpath("policy/acceptance.json").write_bytes(
                policy_bytes
            )

        def side(role: str) -> dict[str, object]:
            profile = MODEL_PROFILES[role]
            return {
                "artifact": {
                    "path": f"models/{role}",
                    "model_id": profile.model_id,
                    "locator": f"hf://{profile.model_id}@{profile.revision}",
                },
                "runtime": {
                    "provider": "hf_vision_text",
                    "settings": _settings(profile),
                },
            }

        request = {
            "format_version": "invarlock/evaluation-request-v1",
            "comparison": {
                "baseline": side("baseline"),
                "subject": side("subject"),
                "dataset": {
                    "path": "inputs/records.jsonl",
                    "sha256": records_sha256,
                    "format": "jsonl",
                    "name": "qwen2-vl-color-grid-tutorial",
                    "split": "tutorial",
                    "input_field": "prompt",
                    "expected_output_field": "expected",
                    "id_field": "id",
                    "content_role": "image",
                    "content_id_field": "content_id",
                    "content_sha256_field": "content_sha256",
                    "content_byte_length_field": "content_bytes",
                    "content_media_type_field": "content_media_type",
                },
                "policy": "inputs/acceptance.json",
                "task": "vision_text_generation",
                "metric": "exact_match",
            },
            "execution": {"mode": "run"},
            "output": {"evidence": "evidence"},
        }
        paths.request.write_text(
            yaml.safe_dump(request, sort_keys=False), encoding="utf-8"
        )

        artifact_anchors = {
            f"{role}_artifact_digest": "sha256:"
            + artifact_identity_sha256(
                HFSnapshotArtifactIdentity(
                    model_id=profile.model_id,
                    immutable_revision=profile.revision,
                    checkpoint_tree_sha256=profile.checkpoint_tree_sha256.removeprefix(
                        "sha256:"
                    ),
                    tokenizer_metadata_sha256=profile.tokenizer_metadata_sha256,
                )
            )
            for role, profile in MODEL_PROFILES.items()
        }
        if not external_trust:
            evidence_signer = _write_private_key(paths.evidence_key)
            verifier_signer = _write_private_key(paths.verifier_key)
        image_digest = normalize_digest(
            runtime_image_digest, label="runtime image digest"
        )
        anchors = {
            **artifact_anchors,
            "schedule_digest": f"sha256:{schedule.schedule_sha256}",
            "baseline_runtime_digest": image_digest,
            "subject_runtime_digest": image_digest,
            "evidence_signer_fingerprint": evidence_signer,
        }
        if external_trust:
            assert trust_root is not None
            assert verifier_key_bytes is not None
            material = create_trust_material(
                transaction_root=root,
                evidence_key=paths.evidence_key,
                verifier_key_bytes=verifier_key_bytes,
                evidence_fingerprint=evidence_signer,
                verifier_fingerprint=verifier_signer,
                trust_root=trust_root,
                policy_bytes=policy_bytes,
                verifier_identity="invarlock-example/hf-vision-text-verifier",
                anchors=anchors,
            )
            if material.trusted_inputs != paths.trusted_inputs:
                raise ValueError(
                    "external trust material resolved to an unexpected root"
                )
        else:
            paths.trusted_inputs.write_bytes(
                canonical_json_bytes(
                    {
                        "format": "invarlock/trust-inputs-v1",
                        "policy": {"path": "policy/acceptance.json"},
                        "anchors": anchors,
                        "verifier": {
                            "identity": "invarlock-example/hf-vision-text-verifier",
                            "signing_key_path": "keys/verifier.pem",
                        },
                        "allow_installed_scorers": False,
                    }
                )
            )
            paths.evidence_key.with_suffix(".fingerprint").write_text(
                evidence_signer + "\n", encoding="ascii"
            )
            paths.verifier_key.with_suffix(".fingerprint").write_text(
                verifier_signer + "\n", encoding="ascii"
            )
        return paths, anchors
    except Exception:
        shutil.rmtree(root)
        raise


def _run(command: list[str], *, environment: dict[str, str]) -> None:
    print("+ " + " ".join(command))
    completed = run_bounded_command(
        command,
        capture_output=True,
        environment=environment,
        label="HF vision integration command",
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"command exited with status {completed.returncode}: {' '.join(command)}"
        )


def execute(
    paths: VisionExamplePaths,
    *,
    container_engine: str,
    runtime_image: str,
    runtime_image_digest: str,
    runtime_device: str,
) -> None:
    """Complete evaluate, independent verification, and HTML reporting."""

    environment = os.environ.copy()
    environment.update(
        {
            "INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT": str(paths.evaluation),
            "INVARLOCK_HF_VISION_TEXT_CONTENT_STORE": "inputs/content",
        }
    )
    base = [sys.executable, "-m", "invarlock"]
    _run(
        [
            *base,
            "evaluate",
            str(paths.request),
            "--signing-key",
            str(paths.evidence_key),
            "--container-engine",
            container_engine,
            "--runtime-image",
            runtime_image,
            "--runtime-image-digest",
            runtime_image_digest,
            "--runtime-device",
            runtime_device,
            "--json",
        ],
        environment=environment,
    )
    _run(
        [
            *base,
            "verify",
            str(paths.evidence),
            "--trust-profile",
            str(paths.trusted_inputs),
            "--receipt",
            str(paths.receipt),
            "--json",
        ],
        environment=environment,
    )
    _run(
        [*base, "report", str(paths.evidence), "--html", str(paths.html_report)],
        environment=environment,
    )
    report = json.loads(
        paths.evidence.joinpath("reports/evaluation.report.json").read_text(
            encoding="utf-8"
        )
    )
    value = report.get("comparison", {}).get("value")
    if report.get("verdict") != "pass" or not isinstance(value, int | float):
        raise RuntimeError("the vision-text tutorial did not produce a passing result")
    print(f"PASS exact-match change: {value:.6f} percentage points")
    print(f"Workspace: {paths.root}")
    print(f"Evidence: {paths.evidence}")
    print(f"Receipt: {paths.receipt}")
    print(f"Report: {paths.html_report}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runtime-image")
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--runtime-device", default="cuda")
    parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evidence-signing-key", type=Path)
    parser.add_argument("--verifier-signing-key", type=Path)
    parser.add_argument("--trust-root", type=Path)
    parser.add_argument(
        "--ephemeral-trust-root",
        action="store_true",
        help="Use disposable generated keys; never use this mode for acceptance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    trust_values = (
        arguments.evidence_signing_key,
        arguments.verifier_signing_key,
        arguments.trust_root,
    )
    provided_trust = any(value is not None for value in trust_values)
    external_trust = all(value is not None for value in trust_values)
    if provided_trust and not external_trust:
        raise SystemExit(
            "--evidence-signing-key, --verifier-signing-key, and --trust-root "
            "must be supplied together"
        )
    if not external_trust and not arguments.ephemeral_trust_root:
        raise SystemExit(
            "caller-owned --evidence-signing-key, --verifier-signing-key, and "
            "--trust-root are required; use --ephemeral-trust-root only for a "
            "disposable non-acceptance demo"
        )
    if external_trust and arguments.ephemeral_trust_root:
        raise SystemExit(
            "--ephemeral-trust-root cannot be combined with caller-owned trust"
        )
    if not arguments.prepare_only and not arguments.runtime_image:
        raise SystemExit("full execution requires --runtime-image")
    try:
        paths, _anchors = prepare_workspace(
            arguments.workspace,
            runtime_image_digest=arguments.runtime_image_digest,
            materialize_models=not arguments.prepare_only,
            evidence_signing_key=arguments.evidence_signing_key,
            verifier_signing_key=arguments.verifier_signing_key,
            trust_root=arguments.trust_root,
            ephemeral_trust_root=arguments.ephemeral_trust_root,
        )
        print(f"Prepared: {paths.root}")
        print(f"Request: {paths.request}")
        print(f"Schedule: {paths.schedule}")
        print(f"Content store: {paths.content}")
        print(f"Policy: {paths.policy}")
        print(f"Independent trust inputs: {paths.trusted_inputs}")
        print(f"Keys outside request tree: {paths.evidence_key.parent}")
        if arguments.prepare_only:
            print(
                "Preparation stops before checkpoint download, image build, or "
                "model execution. Run the maintained command without "
                "--prepare-only to complete evaluate, verify, and report."
            )
            return 0
        assert arguments.runtime_image is not None
        execute(
            paths,
            container_engine=arguments.container_engine,
            runtime_image=arguments.runtime_image,
            runtime_image_digest=arguments.runtime_image_digest,
            runtime_device=arguments.runtime_device,
        )
    except (FileExistsError, RuntimeError, OSError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
