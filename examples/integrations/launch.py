#!/usr/bin/env python3
"""Build the exact runtime image and launch one integration journey."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]

_COMMAND_TIMEOUT_SECONDS = 24 * 60 * 60
_COMMAND_STDOUT_LIMIT = 4 * 1024 * 1024
_COMMAND_STDERR_LIMIT = 4 * 1024 * 1024


try:
    from examples.integrations.local_registry import published_local_image
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from local_registry import published_local_image  # type: ignore[no-redef]
try:
    from examples.integrations.trust_material import read_external_file
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from trust_material import read_external_file  # type: ignore[no-redef]
try:
    from examples.integrations.evaluator_transaction.build_attestation import (
        EvaluatorBuildAttestationError,
        load_evaluator_build_attestation,
        make_evaluator_build_attestation,
        sign_evaluator_build_attestation,
        verify_evaluator_build_attestation,
        write_evaluator_build_attestation,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    try:
        from evaluator_transaction.build_attestation import (
            EvaluatorBuildAttestationError,
            load_evaluator_build_attestation,
            make_evaluator_build_attestation,
            sign_evaluator_build_attestation,
            verify_evaluator_build_attestation,
            write_evaluator_build_attestation,
        )
    except (
        ModuleNotFoundError
    ) as nested_exc:  # pragma: no cover - flat-script compatibility
        if nested_exc.name not in {
            "evaluator_transaction",
            "evaluator_transaction.build_attestation",
        }:
            raise
        from evaluator_transaction import (  # type: ignore[no-redef]
            EvaluatorBuildAttestationError,
            load_evaluator_build_attestation,
            make_evaluator_build_attestation,
            sign_evaluator_build_attestation,
            verify_evaluator_build_attestation,
            write_evaluator_build_attestation,
        )


_INTEGRATIONS = (
    "hf-transformers",
    "hf-vision-text",
    "peft-lora",
    "torchao-int8",
)
_ZERO_DIGEST = "sha256:" + ("0" * 64)
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


def _run(
    command: list[str],
    *,
    cwd: Path,
    capture_output: bool = False,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = run_bounded_command(
            command,
            cwd=cwd,
            environment=environment,
            capture_output=True,
            check=True,
            timeout_seconds=_COMMAND_TIMEOUT_SECONDS,
            stdout_limit=_COMMAND_STDOUT_LIMIT,
            stderr_limit=_COMMAND_STDERR_LIMIT,
            label="integration launcher command",
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            diagnostic = (exc.stderr or exc.output or "").strip()
            detail = f"\n{diagnostic}" if diagnostic else ""
            raise RuntimeError(
                f"command exited with status {exc.returncode}: {' '.join(command)}"
                f"{detail}"
            ) from exc
        raise
    return subprocess.CompletedProcess(
        command,
        completed.returncode,
        completed.stdout if capture_output else None,
        completed.stderr if capture_output else None,
    )


def _git(repository: Path, *arguments: str) -> str:
    return _run(
        ["git", "-C", str(repository), *arguments],
        cwd=repository,
        capture_output=True,
    ).stdout.strip()


def _require_committed_checkout(repository: Path) -> str:
    commit = _git(repository, "rev-parse", "--verify", "HEAD^{commit}")
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "the example runtime is built from committed source; commit or stash "
            "all changes before running the full journey"
        )
    return commit


def _load_runtime_build_statement(
    path: Path,
    *,
    image: str,
    source_commit: str,
    source_bundle_sha256: str,
) -> str:
    """Load the authenticated builder's immutable image identity.

    The image tag is only a build input.  It must not become the execution
    authority after the authenticated builder has completed, because another
    local build can retag it between the builder's final inspection and this
    launcher.
    """

    try:
        raw = read_regular_file_bytes(
            path,
            label="runtime build statement",
            max_bytes=64 * 1024,
        )
        value = parse_json_bytes(raw, label="runtime build statement")
    except StrictJsonError as exc:
        raise RuntimeError(f"runtime build statement is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("runtime build statement must be a JSON object")
    expected_fields = {
        "base_image",
        "build_arguments",
        "dockerfile",
        "format_version",
        "image",
        "ok",
        "platform",
        "runtime_image_id",
        "source_bundle_sha256",
        "source_commit",
    }
    if set(value) != expected_fields:
        raise RuntimeError("runtime build statement has unexpected fields")
    runtime_image_id = value.get("runtime_image_id")
    if (
        value.get("format_version") != "invarlock/runtime-image-build-v1"
        or value.get("ok") is not True
        or value.get("image") != image
        or value.get("source_commit") != source_commit
        or value.get("source_bundle_sha256") != source_bundle_sha256
        or not isinstance(runtime_image_id, str)
        or _IMAGE_ID.fullmatch(runtime_image_id) is None
    ):
        raise RuntimeError("runtime build statement does not bind the requested build")
    return runtime_image_id


def _verify_runtime_image_identity(
    *,
    repository: Path,
    container_engine: str,
    runtime_image_id: str,
    source_commit: str,
    source_bundle_sha256: str,
) -> None:
    """Verify the immutable image is the source-bound image just built."""

    inspected = _run(
        [
            container_engine,
            "image",
            "inspect",
            "--format",
            '{{.Id}}\t{{index .Config.Labels "org.opencontainers.image.revision"}}\t{{index .Config.Labels "dev.invarlock.source-bundle-sha256"}}',
            runtime_image_id,
        ],
        cwd=repository,
        capture_output=True,
    ).stdout.strip()
    fields = inspected.split("\t")
    if fields != [runtime_image_id, source_commit, source_bundle_sha256]:
        raise RuntimeError(
            "authenticated runtime image identity is not bound to the source"
        )


def _load_image_id_file(path: Path) -> str:
    """Read one builder-published immutable image ID without following links."""

    try:
        raw = read_regular_file_bytes(
            path, label="container build image ID", max_bytes=128
        )
        value = raw.decode("ascii").strip()
    except (StrictJsonError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "container build did not publish an immutable image ID"
        ) from exc
    if _IMAGE_ID.fullmatch(value) is None:
        raise RuntimeError("container build did not publish an immutable image ID")
    return value


_ALLOWED_CHILD_CONFIG_FIELDS = {"Entrypoint", "Env", "Labels"}


def _parse_image_config(raw: str, *, label: str) -> dict[str, object]:
    """Parse one complete OCI ``Config`` object from an image inspection."""

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} configuration inspection was not JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} configuration inspection is not an object")
    return value


def _string_map(value: object, *, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, list):
        result: dict[str, str] = {}
        for item in value:
            if not isinstance(item, str) or "=" not in item:
                raise RuntimeError(f"{label} environment is invalid")
            key, rendered = item.split("=", 1)
            if not key:
                raise RuntimeError(f"{label} environment is invalid")
            result[key] = rendered
        return result
    if isinstance(value, Mapping):
        result = {}
        for key, rendered in value.items():
            if not isinstance(key, str) or not key or not isinstance(rendered, str):
                raise RuntimeError(f"{label} labels are invalid")
            result[key] = rendered
        return result
    raise RuntimeError(f"{label} configuration map is invalid")


def _require_child_image_config(
    base: Mapping[str, object],
    child: Mapping[str, object],
    *,
    allowed_environment: set[str] | Mapping[str, str],
    allowed_labels: set[str] | Mapping[str, str],
    expected_entrypoint: Sequence[str] | None = None,
) -> None:
    """Require that a child preserves every authenticated base config field.

    A filesystem-layer prefix does not bind OCI configuration.  The child
    Dockerfiles intentionally replace ``ENTRYPOINT`` and add a small, closed
    set of environment variables and labels; every other inherited setting is
    required to remain byte-for-byte equivalent after JSON decoding.
    """

    for field in set(base) | set(child):
        if field in _ALLOWED_CHILD_CONFIG_FIELDS:
            continue
        if base.get(field) != child.get(field):
            raise RuntimeError(
                f"child image configuration field {field!r} does not match "
                "the authenticated base"
            )
    if expected_entrypoint is not None and child.get("Entrypoint") != list(
        expected_entrypoint
    ):
        raise RuntimeError("child image entrypoint does not match its contract")
    base_environment = _string_map(base.get("Env"), label="base image")
    child_environment = _string_map(child.get("Env"), label="child image")
    allowed_environment_values = (
        dict(allowed_environment) if isinstance(allowed_environment, Mapping) else None
    )
    allowed_environment_keys = (
        set(allowed_environment)
        if allowed_environment_values is None
        else set(allowed_environment_values)
    )
    base_non_overridable = {
        key: value
        for key, value in base_environment.items()
        if key not in allowed_environment_keys
    }
    child_non_overridable = {
        key: value
        for key, value in child_environment.items()
        if key not in allowed_environment_keys
    }
    if child_non_overridable != base_non_overridable:
        raise RuntimeError(
            "child image environment does not preserve the authenticated base"
        )
    unexpected_environment = set(child_environment) - set(base_environment)
    if unexpected_environment - allowed_environment_keys:
        raise RuntimeError(
            "child image adds environment outside its authenticated contract"
        )
    if allowed_environment_values is not None and any(
        child_environment.get(key) != value
        for key, value in allowed_environment_values.items()
    ):
        raise RuntimeError("child image environment does not match its contract")
    base_labels = _string_map(base.get("Labels"), label="base image")
    child_labels = _string_map(child.get("Labels"), label="child image")
    allowed_label_values = (
        dict(allowed_labels) if isinstance(allowed_labels, Mapping) else None
    )
    allowed_label_keys = (
        set(allowed_labels)
        if allowed_label_values is None
        else set(allowed_label_values)
    )
    base_non_overridable = {
        key: value
        for key, value in base_labels.items()
        if key not in allowed_label_keys
    }
    child_non_overridable = {
        key: value
        for key, value in child_labels.items()
        if key not in allowed_label_keys
    }
    if child_non_overridable != base_non_overridable:
        raise RuntimeError("child image labels do not preserve the authenticated base")
    unexpected_labels = set(child_labels) - set(base_labels)
    if unexpected_labels - allowed_label_keys:
        raise RuntimeError("child image adds labels outside its authenticated contract")
    if allowed_label_values is not None and any(
        child_labels.get(key) != value for key, value in allowed_label_values.items()
    ):
        raise RuntimeError("child image labels do not match their contract")


def _evaluator_config_digest(config: Mapping[str, object]) -> str:
    return (
        "sha256:"
        + hashlib.sha256(canonical_json_bytes(dict(config), newline=False)).hexdigest()
    )


def load_builder_signing_key(path: Path) -> ed25519.Ed25519PrivateKey:
    """Load the caller-owned Ed25519 key used to authenticate image builds."""

    try:
        payload = read_external_file(path, label="builder signing key")
        key = serialization.load_pem_private_key(payload, password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("builder signing key is not an Ed25519 private key") from exc
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise RuntimeError("builder signing key is not an Ed25519 private key")
    return key


def load_builder_public_key(path: Path) -> ed25519.Ed25519PublicKey:
    """Load the caller-owned Ed25519 public key trusted for image builds."""

    try:
        payload = read_external_file(path, label="builder public key")
        key = serialization.load_pem_public_key(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("builder public key is not an Ed25519 public key") from exc
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise RuntimeError("builder public key is not an Ed25519 public key")
    return key


def require_builder_key_pair(
    signing_key: ed25519.Ed25519PrivateKey,
    public_key: ed25519.Ed25519PublicKey,
) -> None:
    """Ensure the build signer and the independently supplied trust key agree."""

    if public_key_fingerprint(signing_key.public_key()) != public_key_fingerprint(
        public_key
    ):
        raise RuntimeError("builder signing and public keys do not match")


def write_evaluator_attestation(
    *,
    path: Path,
    evaluator: str,
    evaluator_version: str,
    runtime_image_id: str,
    base_image_id: str,
    source_commit: str,
    source_bundle_sha256: str,
    lock_sha256: str,
    entrypoint: Sequence[str],
    base_layers: Sequence[str],
    image_layers: Sequence[str],
    config: Mapping[str, object],
    builder_signing_key: ed25519.Ed25519PrivateKey,
) -> None:
    """Persist one engine-observed evaluator build attestation."""

    payload = make_evaluator_build_attestation(
        evaluator=evaluator,
        evaluator_version=evaluator_version,
        runtime_image_id=runtime_image_id,
        base_image_id=base_image_id,
        source_commit=source_commit,
        source_bundle_sha256=source_bundle_sha256,
        lock_sha256=lock_sha256,
        entrypoint=entrypoint,
        base_layers=base_layers,
        image_layers=image_layers,
        config=config,
    )
    write_evaluator_build_attestation(
        path,
        sign_evaluator_build_attestation(payload, builder_signing_key),
    )


def inspect_evaluator_image(
    *,
    engine: str,
    image: str,
    repository: Path,
    attestation_path: Path,
    evaluator: str,
    evaluator_version: str,
    lock_sha256: str,
    expected_entrypoint: Sequence[str],
    source_commit: str,
    base_image_id: str,
    builder_public_key: ed25519.Ed25519PublicKey,
) -> dict[str, object]:
    """Re-inspect the exact image used by an evaluator worker."""

    try:
        attestation = load_evaluator_build_attestation(attestation_path)
        statement = (
            attestation.get("statement") if isinstance(attestation, dict) else None
        )
        if not isinstance(statement, dict):
            raise EvaluatorBuildAttestationError(
                "signed evaluator build statement is missing"
            )
        source_bundle_sha256 = statement.get("source_bundle_sha256")
        if not isinstance(source_bundle_sha256, str):
            raise EvaluatorBuildAttestationError("source bundle digest is missing")
        verify_evaluator_build_attestation(
            attestation,
            builder_public_key=builder_public_key,
            evaluator=evaluator,
            evaluator_version=evaluator_version,
            runtime_image_id=image,
            base_image_id=base_image_id,
            source_commit=source_commit,
            source_bundle_sha256=source_bundle_sha256,
            lock_sha256=lock_sha256,
            entrypoint=expected_entrypoint,
        )
        actual_id = _run(
            [engine, "image", "inspect", "--format", "{{.Id}}", image],
            cwd=repository,
            capture_output=True,
        ).stdout.strip()
        if actual_id != image:
            raise EvaluatorBuildAttestationError(
                "engine image identity changed after build"
            )
        actual_layers = json.loads(
            _run(
                [
                    engine,
                    "image",
                    "inspect",
                    "--format",
                    "{{json .RootFS.Layers}}",
                    image,
                ],
                cwd=repository,
                capture_output=True,
            ).stdout
        )
        base_layers = json.loads(
            _run(
                [
                    engine,
                    "image",
                    "inspect",
                    "--format",
                    "{{json .RootFS.Layers}}",
                    base_image_id,
                ],
                cwd=repository,
                capture_output=True,
            ).stdout
        )
        config = _parse_image_config(
            _run(
                [engine, "image", "inspect", "--format", "{{json .Config}}", image],
                cwd=repository,
                capture_output=True,
            ).stdout,
            label=image,
        )
        labels = config.get("Labels")
        if not isinstance(labels, dict):
            raise EvaluatorBuildAttestationError("evaluator image labels are invalid")
        expected_labels = {
            "org.invarlock.example.base-image-id": base_image_id,
            "org.invarlock.example.evaluator": evaluator,
            "org.invarlock.example.evaluator-version": evaluator_version,
            "org.invarlock.example.evaluator-lock-sha256": lock_sha256,
            "org.invarlock.example.source-commit": source_commit,
            "org.invarlock.example.source-bundle-sha256": statement[
                "source_bundle_sha256"
            ],
        }
        if any(labels.get(key) != value for key, value in expected_labels.items()):
            raise EvaluatorBuildAttestationError(
                "evaluator image labels are not authenticated"
            )
        if config.get("Entrypoint") != list(expected_entrypoint):
            raise EvaluatorBuildAttestationError(
                "evaluator image entrypoint is not exact"
            )
        if not isinstance(actual_layers, list) or not isinstance(base_layers, list):
            raise EvaluatorBuildAttestationError(
                "evaluator image layer inspection is invalid"
            )
        if actual_layers[: len(base_layers)] != base_layers:
            raise EvaluatorBuildAttestationError(
                "evaluator image does not derive from its base"
            )
        if actual_layers != statement["image_layers"]:
            raise EvaluatorBuildAttestationError("evaluator image layer chain changed")
        if base_layers != statement["base_layers"]:
            raise EvaluatorBuildAttestationError("evaluator base layer chain changed")
        if _evaluator_config_digest(config) != statement["config_sha256"]:
            raise EvaluatorBuildAttestationError(
                "evaluator image configuration changed"
            )
        return attestation
    except EvaluatorBuildAttestationError as exc:
        raise RuntimeError(str(exc)) from exc


def _runtime_image(
    *,
    repository: Path,
    build_root: Path,
    container_engine: str,
    dockerfile: str = "runtime/Dockerfile",
    image_prefix: str = "invarlock-example-runtime",
    image_tag: str | None = None,
    build_arguments: tuple[str, ...] = (),
    authenticated_base_image: str | None = None,
) -> tuple[str, str]:
    commit = _require_committed_checkout(repository)
    source_bundle = build_root / "source.tar"
    source = _run(
        [
            sys.executable,
            str(repository / "scripts/qualification_source.py"),
            "create",
            "--repository",
            str(repository),
            "--commit",
            commit,
            "--output",
            str(source_bundle),
        ],
        cwd=repository,
        capture_output=True,
    )
    identity = json.loads(source.stdout)
    bundle_digest = identity.get("source_bundle_sha256")
    if not isinstance(bundle_digest, str):
        raise RuntimeError("source-bundle creation did not return its digest")
    epoch = _git(repository, "show", "-s", "--format=%ct", commit)
    image = image_tag or f"{image_prefix}:{commit[:12]}"
    build_command = [
        sys.executable,
        str(repository / "scripts/authenticated_runtime_build.py"),
        "--repository",
        str(repository),
        "--source-commit",
        commit,
        "--source-bundle",
        str(source_bundle),
        "--source-bundle-sha256",
        bundle_digest,
        "--container-engine",
        container_engine,
        "--dockerfile",
        dockerfile,
        "--image",
        image,
        "--statement",
        str(build_root / "runtime-build.json"),
        "--build-arg",
        f"SOURCE_DATE_EPOCH={epoch}",
    ]
    for argument in build_arguments:
        build_command.extend(("--build-arg", argument))
    if authenticated_base_image is not None:
        build_command.extend(
            (
                "--build-arg",
                f"RUNTIME_BASE_IMAGE={authenticated_base_image}",
                "--require-base-source-labels",
                authenticated_base_image,
            )
        )
    _run(
        build_command,
        cwd=repository,
    )
    runtime_image_id = _load_runtime_build_statement(
        build_root / "runtime-build.json",
        image=image,
        source_commit=commit,
        source_bundle_sha256=bundle_digest,
    )
    _verify_runtime_image_identity(
        repository=repository,
        container_engine=container_engine,
        runtime_image_id=runtime_image_id,
        source_commit=commit,
        source_bundle_sha256=bundle_digest,
    )
    # The statement identity is the only execution coordinate returned from the
    # authenticated build.  The mutable tag is deliberately never executed.
    return runtime_image_id, runtime_image_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("integration", choices=_INTEGRATIONS)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    parser.add_argument(
        "--runtime-device",
        type=_runtime_device_argument,
        default="auto",
        help=(
            "Use CUDA when available, or select CPU, CUDA, or a concrete "
            "CUDA device such as cuda:1 explicitly."
        ),
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


def _resolve_runtime_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _runtime_device_argument(value: str) -> str:
    if value in {"auto", "cpu", "cuda"}:
        return value
    prefix, separator, index = value.partition(":")
    if (
        prefix == "cuda"
        and separator
        and index
        and all("0" <= character <= "9" for character in index)
    ):
        return value
    raise argparse.ArgumentTypeError(
        "runtime device must be auto, cpu, cuda, or cuda:<non-negative-index>"
    )


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
        print(
            "FAIL --evidence-signing-key, --verifier-signing-key, and "
            "--trust-root must be supplied together",
            file=sys.stderr,
        )
        return 2
    if not external_trust and not arguments.ephemeral_trust_root:
        print(
            "FAIL caller-owned --evidence-signing-key, --verifier-signing-key, "
            "and --trust-root are required; use --ephemeral-trust-root only for "
            "a disposable non-acceptance demo",
            file=sys.stderr,
        )
        return 2
    if external_trust and arguments.ephemeral_trust_root:
        print(
            "FAIL --ephemeral-trust-root cannot be combined with caller-owned trust",
            file=sys.stderr,
        )
        return 2
    if (
        arguments.integration == "hf-vision-text"
        and arguments.container_engine != "docker"
    ):
        print(
            "FAIL the vision-text example requires Docker for its authenticated "
            "layered runtime build",
            file=sys.stderr,
        )
        return 2
    repository = Path(__file__).resolve().parents[2]
    workspace = (
        Path(os.path.abspath(arguments.workspace.expanduser()))
        if arguments.workspace is not None
        else Path(
            tempfile.mkdtemp(prefix=f"invarlock-{arguments.integration}-")
        ).resolve()
    )
    if arguments.workspace is not None and (
        workspace.exists() or workspace.is_symlink()
    ):
        print(f"FAIL workspace already exists: {workspace}", file=sys.stderr)
        return 2
    if arguments.workspace is not None:
        workspace.parent.mkdir(parents=True, exist_ok=True)
        workspace.mkdir()
    transaction = workspace / "transaction"
    try:
        runtime_device = (
            "cuda"
            if arguments.integration == "hf-vision-text"
            and arguments.runtime_device == "auto"
            else _resolve_runtime_device(arguments.runtime_device)
        )
        if arguments.prepare_only:
            image = None
            image_digest = _ZERO_DIGEST
        else:
            build_root = workspace / "build"
            build_root.mkdir()
            if arguments.integration == "hf-vision-text":
                base_build = build_root / "base"
                vision_build = build_root / "vision"
                base_build.mkdir()
                vision_build.mkdir()
                base_prefix = "invarlock-example-runtime-cuda"
                base_image, base_digest = _runtime_image(
                    repository=repository,
                    build_root=base_build,
                    container_engine=arguments.container_engine,
                    dockerfile="runtime/Dockerfile.cuda",
                    image_prefix=base_prefix,
                )
                with published_local_image(
                    repository=repository,
                    container_engine=arguments.container_engine,
                    image=base_image,
                    image_digest=base_digest,
                    repository_name=base_prefix,
                ) as published_base:
                    image, image_digest = _runtime_image(
                        repository=repository,
                        build_root=vision_build,
                        container_engine=arguments.container_engine,
                        dockerfile="addins/multimodal/runtime/Dockerfile",
                        image_prefix="invarlock-example-hf-vision-text",
                        authenticated_base_image=published_base,
                    )
            else:
                image, image_digest = _runtime_image(
                    repository=repository,
                    build_root=build_root,
                    container_engine=arguments.container_engine,
                    dockerfile=(
                        "runtime/Dockerfile.cuda"
                        if runtime_device.startswith("cuda")
                        else "runtime/Dockerfile"
                    ),
                    image_prefix=(
                        "invarlock-example-runtime-cuda"
                        if runtime_device.startswith("cuda")
                        else "invarlock-example-runtime"
                    ),
                )
        worker = (
            repository / "examples/integrations/hf_vision_text.py"
            if arguments.integration == "hf-vision-text"
            else repository / "examples/integrations/run.py"
        )
        command = [
            sys.executable,
            str(worker),
        ]
        if arguments.integration != "hf-vision-text":
            command.append(arguments.integration)
        command.extend(
            [
                "--workspace",
                str(transaction),
                "--runtime-image-digest",
                image_digest,
                "--container-engine",
                arguments.container_engine,
                "--runtime-device",
                runtime_device,
            ]
        )
        if external_trust:
            command.extend(
                [
                    "--evidence-signing-key",
                    str(arguments.evidence_signing_key),
                    "--verifier-signing-key",
                    str(arguments.verifier_signing_key),
                    "--trust-root",
                    str(arguments.trust_root),
                ]
            )
        else:
            command.append("--ephemeral-trust-root")
        if arguments.prepare_only:
            command.append("--prepare-only")
        else:
            assert image is not None
            command.extend(("--runtime-image", image))
        _run(command, cwd=repository)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"Complete integration workspace: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
