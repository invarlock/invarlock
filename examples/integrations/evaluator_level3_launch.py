#!/usr/bin/env python3
"""Build one pinned evaluator image and run its signed Level 3 journey."""

from __future__ import annotations

import argparse
import hashlib
import json
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

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

try:
    from examples.integrations.evaluator_transaction.image_cleanup import (
        remove_temporary_image_tags,
        require_image_tag_available,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from evaluator_transaction.image_cleanup import (  # type: ignore[no-redef]
        remove_temporary_image_tags,
        require_image_tag_available,
    )

DOCKERFILES = {
    "inspect-ai": "examples/integrations/inspect-ai/Dockerfile",
    "openai-evals": "examples/integrations/openai-evals/Dockerfile",
}
LOCKS = {
    "inspect-ai": "requirements/workflows/inspect-ai-level3-py312.txt",
    "openai-evals": "requirements/workflows/openai-evals-level3-py312.txt",
}
# Each integration image has one requirements COPY, one dependency-install
# layer, one worker COPY, and six flat helper COPY layers.
ADDED_LAYERS = {"inspect-ai": 9, "openai-evals": 9}
EVALUATOR_VERSIONS = {
    "inspect-ai": "0.3.254",
    "openai-evals": "3.0.1.post1",
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
            label="Level 3 launcher command",
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
    cleanup_tags: list[str],
) -> tuple[str, str, str]:
    try:
        from examples.integrations.launch import (
            _load_image_id_file,
            _require_child_image_config,
            _require_committed_checkout,
            _runtime_image,
            write_level3_attestation,
        )
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith("examples"):
            raise
        from launch import (  # type: ignore[no-redef]
            _load_image_id_file,
            _require_child_image_config,
            _require_committed_checkout,
            _runtime_image,
            write_level3_attestation,
        )

    commit = _require_committed_checkout(repository)
    base_tag = f"invarlock-example-runtime:{commit[:12]}"
    require_image_tag_available(run, engine, base_tag, repository)
    cleanup_tags.append(base_tag)
    base_build = build / "base"
    base_build.mkdir(parents=True, exist_ok=True)
    base_id, _ = _runtime_image(
        repository=repository, build_root=base_build, container_engine=engine
    )
    base_layers = _image_layers(engine, base_id, repository)
    base_config = _image_config(engine, base_id, repository)
    source = build / "evaluator-source.tar"
    run(
        ["git", "archive", "--format=tar", f"--output={source}", commit], cwd=repository
    )
    source_bundle_sha256 = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    image_tag = f"invarlock-{evaluator}-level3:{commit[:12]}"
    require_image_tag_available(run, engine, image_tag, repository)
    cleanup_tags.append(image_tag)
    lock_digest = (
        "sha256:"
        + hashlib.sha256((repository / LOCKS[evaluator]).read_bytes()).hexdigest()
    )
    image_id_file = build / "evaluator-image-id"
    if image_id_file.exists() or image_id_file.is_symlink():
        raise RuntimeError("evaluator image identity file already exists")
    pull_policy = "--pull=never" if engine == "podman" else "--pull=false"
    status(f"Building the pinned {evaluator} Level 3 image...")
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
            "--tag",
            image_tag,
            "-",
        ],
        cwd=repository,
        stdin_path=source,
    )
    image_id = _load_image_id_file(image_id_file)
    evaluator_layers = _image_layers(engine, image_id, repository)
    evaluator_config = _image_config(engine, image_id, repository)
    expected_layers = len(base_layers) + ADDED_LAYERS[evaluator]
    if len(evaluator_layers) != expected_layers or (
        evaluator_layers[: len(base_layers)] != base_layers
    ):
        raise RuntimeError(
            "evaluator image filesystem does not derive from the authenticated base"
        )
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
            "org.invarlock.example.source-commit": commit,
            "org.invarlock.example.source-bundle-sha256": source_bundle_sha256,
        },
        expected_entrypoint=[
            "python",
            "/opt/invarlock/examples/evaluator-level3.py",
            "worker",
        ],
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
    write_level3_attestation(
        path=build / "level3-build-attestation.json",
        evaluator=evaluator,
        evaluator_version=EVALUATOR_VERSIONS[evaluator],
        runtime_image_id=image_id,
        base_image_id=base_id,
        source_commit=commit,
        source_bundle_sha256=source_bundle_sha256,
        lock_sha256=lock_digest,
        entrypoint=[
            "python",
            "/opt/invarlock/examples/evaluator-level3.py",
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
    args = parser.parse_args(argv)
    repository = REPOSITORY
    if args.workspace is None:
        workspace = Path(
            tempfile.mkdtemp(prefix=f"invarlock-{evaluator}-level3-")
        ).resolve(strict=True)
    else:
        workspace = args.workspace.expanduser().resolve()
        if workspace.exists() or workspace.is_symlink():
            print(f"FAIL workspace already exists: {workspace}", file=sys.stderr)
            return 2
        workspace.mkdir(parents=True)
    cleanup_tags: list[str] = []
    cleanup_error: RuntimeError | None = None
    output: str | None = None
    result = 2
    try:
        build = workspace / "build"
        build.mkdir()
        builder_signing_key = load_builder_signing_key(
            args.builder_signing_key.expanduser().resolve()
        )
        builder_public_key = load_builder_public_key(
            args.builder_public_key.expanduser().resolve()
        )
        require_builder_key_pair(builder_signing_key, builder_public_key)
        image, source_commit, base_image_id = _build_image(
            evaluator,
            repository,
            build,
            args.container_engine,
            builder_signing_key=builder_signing_key,
            cleanup_tags=cleanup_tags,
        )
        prepared = workspace / "prepared"
        status("Preparing the pinned Qwen3-0.6B checkpoints and 102-record dataset...")
        run(
            [
                sys.executable,
                str(
                    repository
                    / "examples/integrations/lm-evaluation-harness/model_inputs.py"
                ),
                "--workspace",
                str(prepared),
                "--runtime-image",
                image,
            ],
            cwd=repository,
        )
        status(
            "Running the evaluator in the inspected image and independently verifying "
            "the signed evidence transaction..."
        )
        output = run(
            [
                sys.executable,
                str(repository / "examples/integrations/evaluator_level3.py"),
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
                "--evidence-signing-key",
                str(args.evidence_signing_key.expanduser().resolve()),
                "--verifier-signing-key",
                str(args.verifier_signing_key.expanduser().resolve()),
                "--trust-root",
                str(args.trust_root.expanduser().absolute()),
                "--builder-public-key",
                str(args.builder_public_key.expanduser().resolve()),
                "--source-commit",
                source_commit,
                "--base-image-id",
                base_image_id,
                "--build-attestation",
                str(build / "level3-build-attestation.json"),
            ],
            cwd=repository,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
    else:
        result = 0
    finally:
        try:
            remove_temporary_image_tags(
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
