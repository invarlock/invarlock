#!/usr/bin/env python3
"""Build one pinned evaluator image and run its signed transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]

_COMMAND_TIMEOUT_SECONDS = 24 * 60 * 60
_COMMAND_STDOUT_LIMIT = 4 * 1024 * 1024
_COMMAND_STDERR_LIMIT = 256 * 1024

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

try:
    from examples.integrations.evaluator_transaction.image_cleanup import (
        OwnedImageTag,
        record_owned_image_tag,
        remove_owned_image_tags,
        temporary_image_tag,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from evaluator_transaction.image_cleanup import (  # type: ignore[no-redef]
        OwnedImageTag,
        record_owned_image_tag,
        remove_owned_image_tags,
        temporary_image_tag,
    )

DOCKERFILES = {
    "inspect-ai": "examples/integrations/inspect-ai/Dockerfile",
    "openai-evals": "examples/integrations/openai-evals/Dockerfile",
}
LOCKS = {
    "inspect-ai": "requirements/workflows/inspect-ai-runtime-py312.txt",
    "openai-evals": "requirements/workflows/openai-evals-runtime-py312.txt",
}
CUDA_LOCKS = {
    "inspect-ai": "requirements/workflows/inspect-ai-runtime-py312-cu129.txt",
    "openai-evals": "requirements/workflows/openai-evals-runtime-py312-cu129.txt",
}
EVALUATOR_VERSIONS = {
    "inspect-ai": "0.3.254",
    "openai-evals": "3.0.1.post1+invarlock.match.1",
}
try:
    from examples.integrations.launch import (
        load_builder_public_key,
        load_builder_signing_key,
        require_builder_key_pair,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from launch import (  # type: ignore[no-redef]
        load_builder_public_key,
        load_builder_signing_key,
        require_builder_key_pair,
    )


def run(command: list[str], *, cwd: Path, stdin_path: Path | None = None) -> str:
    try:
        completed = run_bounded_command(
            command,
            cwd=cwd,
            stdin_path=stdin_path,
            capture_output=True,
            check=True,
            timeout_seconds=_COMMAND_TIMEOUT_SECONDS,
            stdout_limit=_COMMAND_STDOUT_LIMIT,
            stderr_limit=_COMMAND_STDERR_LIMIT,
            label="evaluator launcher command",
        )
    except subprocess.CalledProcessError as exc:
        diagnostic = (exc.stderr or exc.output or "").strip()
        raise RuntimeError(
            diagnostic or f"command exited with status {exc.returncode}"
        ) from exc
    return (completed.stdout or "").strip()


def mount_source(path: Path) -> str:
    rendered = str(path)
    if any(character in rendered for character in (",", "\n", "\r", "\x00")):
        raise ValueError("workspace path cannot be represented in an OCI mount")
    return rendered


def status(message: str) -> None:
    print(message, flush=True)


def _image_layers(engine: str, image: str, repository: Path) -> tuple[str, ...]:
    """Read the immutable filesystem-layer chain for one local image."""

    raw = run(
        [engine, "image", "inspect", "--format", "{{json .RootFS.Layers}}", image],
        cwd=repository,
    )
    try:
        layers = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("container image layer inspection was not JSON") from exc
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(layer, str) or not layer for layer in layers)
    ):
        raise RuntimeError("container image layer inspection is invalid")
    return tuple(layers)


def _image_config(engine: str, image: str, repository: Path) -> dict[str, object]:
    from examples.integrations.launch import _parse_image_config

    raw = run(
        [engine, "image", "inspect", "--format", "{{json .Config}}", image],
        cwd=repository,
    )
    return _parse_image_config(raw, label=image)


def _build_image(
    evaluator: str,
    repository: Path,
    build: Path,
    engine: str,
    *,
    builder_signing_key: ed25519.Ed25519PrivateKey,
    cleanup_tags: list[OwnedImageTag],
    runtime_profile: str = "cpu",
) -> tuple[str, str, str]:
    try:
        from examples.integrations.launch import (
            _load_image_id_file,
            _require_child_image_config,
            _require_child_image_layers,
            _require_committed_checkout,
            _runtime_image,
            write_evaluator_attestation,
        )
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith("examples"):
            raise
        from launch import (  # type: ignore[no-redef]
            _load_image_id_file,
            _require_child_image_config,
            _require_child_image_layers,
            _require_committed_checkout,
            _runtime_image,
            write_evaluator_attestation,
        )

    commit = _require_committed_checkout(repository)
    if runtime_profile not in {"cpu", "cu129"}:
        raise ValueError("evaluator runtime profile is invalid")
    base_tag = temporary_image_tag(
        f"invarlock-example-runtime-{runtime_profile}", commit
    )
    base_build = build / "base"
    base_build.mkdir(parents=True, exist_ok=True)
    base_id, _ = _runtime_image(
        repository=repository,
        build_root=base_build,
        container_engine=engine,
        image_tag=base_tag,
        dockerfile=(
            "runtime/Dockerfile.cuda"
            if runtime_profile == "cu129"
            else "runtime/Dockerfile"
        ),
        build_arguments=(("CUDA_PROFILE=cu129",) if runtime_profile == "cu129" else ()),
    )
    cleanup_tags.append(
        record_owned_image_tag(run, engine, base_tag, base_id, repository)
    )
    base_layers = _image_layers(engine, base_id, repository)
    base_config = _image_config(engine, base_id, repository)
    source = build / "evaluator-source.tar"
    run(
        ["git", "archive", "--format=tar", f"--output={source}", commit], cwd=repository
    )
    source_bundle_sha256 = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    image_tag = temporary_image_tag(f"invarlock-{evaluator}-evaluator", commit)
    lock_path = (CUDA_LOCKS if runtime_profile == "cu129" else LOCKS)[evaluator]
    lock_digest = (
        "sha256:" + hashlib.sha256((repository / lock_path).read_bytes()).hexdigest()
    )
    image_id_file = build / "evaluator-image-id"
    if image_id_file.exists() or image_id_file.is_symlink():
        raise RuntimeError("evaluator image identity file already exists")
    pull_policy = "--pull=never" if engine == "podman" else "--pull=false"
    status(f"Building the pinned {evaluator} evaluator image...")
    run(
        [
            engine,
            "build",
            pull_policy,
            "--iidfile",
            str(image_id_file),
            "--file",
            DOCKERFILES[evaluator],
            "--build-arg",
            f"BASE_IMAGE={base_tag}",
            "--build-arg",
            f"BASE_IMAGE_ID={base_id}",
            "--build-arg",
            f"SOURCE_COMMIT={commit}",
            "--build-arg",
            f"SOURCE_BUNDLE_SHA256={source_bundle_sha256}",
            "--build-arg",
            f"EVALUATOR_LOCK_SHA256={lock_digest.removeprefix('sha256:')}",
            "--build-arg",
            f"EVALUATOR_RUNTIME={runtime_profile}",
            "--tag",
            image_tag,
            "-",
        ],
        cwd=repository,
        stdin_path=source,
    )
    image_id = _load_image_id_file(image_id_file)
    cleanup_tags.append(
        record_owned_image_tag(run, engine, image_tag, image_id, repository)
    )
    evaluator_layers = _image_layers(engine, image_id, repository)
    evaluator_config = _image_config(engine, image_id, repository)
    _require_child_image_layers(base_layers, evaluator_layers)
    _require_child_image_config(
        base_config,
        evaluator_config,
        allowed_environment={
            "INVARLOCK_EVALUATOR": evaluator,
            "INVARLOCK_EVALUATOR_LOCK_SHA256": lock_digest,
        },
        allowed_labels={
            "org.invarlock.example.base-image-id": base_id,
            "org.invarlock.example.evaluator": evaluator,
            "org.invarlock.example.evaluator-version": EVALUATOR_VERSIONS[evaluator],
            "org.invarlock.example.evaluator-lock-sha256": lock_digest,
            "org.invarlock.example.evaluator-runtime": runtime_profile,
            "org.invarlock.example.source-commit": commit,
            "org.invarlock.example.source-bundle-sha256": source_bundle_sha256,
        },
        expected_entrypoint=[
            "python",
            "-m",
            "evaluator_transaction.cli",
            "worker",
        ],
        expected_working_directory="/opt/invarlock/examples",
    )
    inspected_id = run(
        [engine, "image", "inspect", "--format", "{{.Id}}", image_id], cwd=repository
    )
    if inspected_id != image_id:
        raise RuntimeError("evaluator image identity file disagrees with inspection")
    embedded_base = run(
        [
            engine,
            "image",
            "inspect",
            "--format",
            '{{index .Config.Labels "org.invarlock.example.base-image-id"}}',
            image_id,
        ],
        cwd=repository,
    )
    if embedded_base != base_id:
        raise RuntimeError("evaluator image does not bind the inspected base image ID")
    embedded_evaluator = run(
        [
            engine,
            "image",
            "inspect",
            "--format",
            '{{index .Config.Labels "org.invarlock.example.evaluator"}}',
            image_id,
        ],
        cwd=repository,
    )
    embedded_lock = run(
        [
            engine,
            "image",
            "inspect",
            "--format",
            '{{index .Config.Labels "org.invarlock.example.evaluator-lock-sha256"}}',
            image_id,
        ],
        cwd=repository,
    )
    if embedded_evaluator != evaluator or embedded_lock != lock_digest:
        raise RuntimeError("evaluator image does not bind its evaluator lock")
    write_evaluator_attestation(
        path=build / "evaluator-build-attestation.json",
        evaluator=evaluator,
        evaluator_version=EVALUATOR_VERSIONS[evaluator],
        runtime_image_id=image_id,
        base_image_id=base_id,
        source_commit=commit,
        source_bundle_sha256=source_bundle_sha256,
        lock_sha256=lock_digest,
        entrypoint=[
            "python",
            "-m",
            "evaluator_transaction.cli",
            "worker",
        ],
        base_layers=base_layers,
        image_layers=evaluator_layers,
        config=evaluator_config,
        builder_signing_key=builder_signing_key,
    )
    # The child tag is only a local build coordinate.  All later preparation
    # and worker commands must use the inspected immutable child identity.
    return image_id, commit, base_id


def main(evaluator: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    parser.add_argument("--evidence-signing-key", type=Path, required=True)
    parser.add_argument("--verifier-signing-key", type=Path, required=True)
    parser.add_argument("--trust-root", type=Path, required=True)
    parser.add_argument("--builder-signing-key", type=Path, required=True)
    parser.add_argument("--builder-public-key", type=Path, required=True)
    parser.add_argument(
        "--corpus-profile",
        choices=("quick", "deployment", "flagship", "portability"),
        default="quick",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--allow-policy-fail",
        action="store_true",
        help="retain a verified policy rejection as a completed evidence transaction",
    )
    args = parser.parse_args(argv)
    repository = REPOSITORY
    if args.workspace is None:
        workspace = Path(
            tempfile.mkdtemp(prefix=f"invarlock-{evaluator}-transaction-")
        ).resolve(strict=True)
    else:
        workspace = Path(os.path.abspath(args.workspace.expanduser()))
        if workspace.exists() or workspace.is_symlink():
            print(f"FAIL workspace already exists: {workspace}", file=sys.stderr)
            return 2
        workspace.mkdir(parents=True)
    cleanup_tags: list[OwnedImageTag] = []
    cleanup_error: RuntimeError | None = None
    output: str | None = None
    result = 2
    try:
        from examples.integrations.evaluator_transaction.model_profiles import (
            model_profile,
        )

        build = workspace / "build"
        build.mkdir()
        builder_signing_key = load_builder_signing_key(
            Path(os.path.abspath(args.builder_signing_key.expanduser()))
        )
        builder_public_key = load_builder_public_key(
            Path(os.path.abspath(args.builder_public_key.expanduser()))
        )
        require_builder_key_pair(builder_signing_key, builder_public_key)
        selected_models = model_profile(args.corpus_profile)
        runtime_profile = "cu129" if selected_models.device == "cuda" else "cpu"
        device = args.device or selected_models.device
        if (runtime_profile == "cpu") != (device == "cpu"):
            raise ValueError(
                "the selected corpus and device require different runtimes"
            )
        image, source_commit, base_image_id = _build_image(
            evaluator,
            repository,
            build,
            args.container_engine,
            builder_signing_key=builder_signing_key,
            cleanup_tags=cleanup_tags,
            runtime_profile=runtime_profile,
        )
        prepared = workspace / "prepared"
        status(
            "Preparing the pinned evaluator checkpoints and "
            f"{args.corpus_profile} corpus..."
        )
        prepare_command = [
            sys.executable,
            str(
                repository
                / "examples/integrations/lm-evaluation-harness/model_inputs.py"
            ),
            "--workspace",
            str(prepared),
            "--runtime-image",
            image,
            "--corpus-profile",
            args.corpus_profile,
        ]
        run(prepare_command, cwd=repository)
        status(
            "Running the evaluator in the inspected image and independently verifying "
            "the signed evidence transaction..."
        )
        output = run(
            [
                sys.executable,
                "-m",
                "examples.integrations.evaluator_transaction.cli",
                "complete",
                "--workspace",
                str(workspace / "transaction"),
                "--prepared",
                str(prepared),
                "--runtime-image",
                image,
                "--evaluator",
                evaluator,
                "--container-engine",
                args.container_engine,
                "--device",
                device,
                "--evidence-signing-key",
                str(Path(os.path.abspath(args.evidence_signing_key.expanduser()))),
                "--verifier-signing-key",
                str(Path(os.path.abspath(args.verifier_signing_key.expanduser()))),
                "--trust-root",
                str(Path(os.path.abspath(args.trust_root.expanduser()))),
                "--builder-public-key",
                str(Path(os.path.abspath(args.builder_public_key.expanduser()))),
                "--source-commit",
                source_commit,
                "--base-image-id",
                base_image_id,
                "--build-attestation",
                str(build / "evaluator-build-attestation.json"),
                *(["--allow-policy-fail"] if args.allow_policy_fail else []),
            ],
            cwd=repository,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
    else:
        result = 0
    finally:
        try:
            remove_owned_image_tags(
                run, args.container_engine, repository, cleanup_tags
            )
        except RuntimeError as exc:
            cleanup_error = exc
            print(f"FAIL {exc}", file=sys.stderr)
            result = 2
        if cleanup_tags and cleanup_error is None:
            status("Removed temporary evaluator image tags.")
    if result:
        return result
    assert output is not None
    print(output)
    print(f"Complete integration workspace: {workspace}")
    return 0
