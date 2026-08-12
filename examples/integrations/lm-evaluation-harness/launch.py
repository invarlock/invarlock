#!/usr/bin/env python3
"""Build the pinned Harness runtime and run its complete InvarLock journey."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from examples.integrations.bounded_command import run_bounded_command
except ModuleNotFoundError as exc:  # pragma: no cover - flat-script compatibility
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from bounded_command import run_bounded_command  # type: ignore[no-redef]

REPOSITORY = Path(__file__).resolve().parents[3]
ADDED_LAYERS = 10
_COMMAND_TIMEOUT_SECONDS = 24 * 60 * 60
_COMMAND_STDOUT_LIMIT = 4 * 1024 * 1024
_COMMAND_STDERR_LIMIT = 4 * 1024 * 1024
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from examples.integrations.evaluator_transaction.image_cleanup import (  # noqa: E402
    OwnedImageTag,
    record_owned_image_tag,
    remove_owned_image_tags,
    temporary_image_tag,
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
            label="LM Evaluation Harness launcher command",
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
    """Show progress while long-running subprocess output remains captured."""

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


def main(argv: list[str] | None = None) -> int:
    from examples.integrations.launch import (
        _load_image_id_file,
        _require_child_image_config,
        _require_committed_checkout,
        _runtime_image,
        load_builder_public_key,
        load_builder_signing_key,
        require_builder_key_pair,
        write_evaluator_attestation,
    )

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
    builder_signing_key = load_builder_signing_key(
        Path(os.path.abspath(args.builder_signing_key.expanduser()))
    )
    builder_public_key = load_builder_public_key(
        Path(os.path.abspath(args.builder_public_key.expanduser()))
    )
    require_builder_key_pair(builder_signing_key, builder_public_key)
    repository = REPOSITORY
    if args.workspace is None:
        workspace = Path(tempfile.mkdtemp(prefix="invarlock-lm-eval-")).resolve(
            strict=True
        )
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
        commit = _require_committed_checkout(repository)
        build = workspace / "build"
        build.mkdir()
        base_tag = temporary_image_tag("invarlock-example-runtime", commit)
        status("Building the source-bound InvarLock runtime image...")
        base_id, _ = _runtime_image(
            repository=repository,
            build_root=build,
            container_engine=args.container_engine,
            image_tag=base_tag,
        )
        cleanup_tags.append(
            record_owned_image_tag(
                run, args.container_engine, base_tag, base_id, repository
            )
        )
        base_layers = _image_layers(args.container_engine, base_id, repository)
        base_config = _image_config(args.container_engine, base_id, repository)
        source = build / "harness-source.tar"
        run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                f"--output={source}",
                commit,
            ],
            cwd=repository,
        )
        source_bundle_sha256 = (
            "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
        )
        lock_sha256 = (
            "sha256:"
            + hashlib.sha256(
                (
                    repository
                    / "requirements/workflows/lm-evaluation-harness-py312.txt"
                ).read_bytes()
            ).hexdigest()
        )
        image_tag = temporary_image_tag("invarlock-lm-eval-example", commit)
        image_id_file = build / "harness-image-id"
        if image_id_file.exists() or image_id_file.is_symlink():
            raise RuntimeError("Harness image identity file already exists")
        pull_policy = (
            "--pull=never" if args.container_engine == "podman" else "--pull=false"
        )
        status("Building the pinned LM Evaluation Harness image...")
        run(
            [
                args.container_engine,
                "build",
                pull_policy,
                "--iidfile",
                str(image_id_file),
                "--file",
                "examples/integrations/lm-evaluation-harness/Dockerfile",
                "--build-arg",
                f"BASE_IMAGE={base_tag}",
                "--build-arg",
                f"BASE_IMAGE_ID={base_id}",
                "--build-arg",
                f"SOURCE_COMMIT={commit}",
                "--build-arg",
                f"SOURCE_BUNDLE_SHA256={source_bundle_sha256}",
                "--build-arg",
                f"EVALUATOR_LOCK_SHA256={lock_sha256.removeprefix('sha256:')}",
                "--tag",
                image_tag,
                "-",
            ],
            cwd=repository,
            stdin_path=source,
        )
        image_id = _load_image_id_file(image_id_file)
        cleanup_tags.append(
            record_owned_image_tag(
                run, args.container_engine, image_tag, image_id, repository
            )
        )
        harness_layers = _image_layers(args.container_engine, image_id, repository)
        harness_config = _image_config(args.container_engine, image_id, repository)
        if len(harness_layers) != len(base_layers) + ADDED_LAYERS or (
            harness_layers[: len(base_layers)] != base_layers
        ):
            raise RuntimeError(
                "Harness image filesystem does not derive from the authenticated base"
            )
        _require_child_image_config(
            base_config,
            harness_config,
            allowed_environment=set(),
            allowed_labels={
                "org.invarlock.example.base-image-id": base_id,
                "org.invarlock.example.source-commit": commit,
                "org.invarlock.example.source-bundle-sha256": source_bundle_sha256,
                "org.invarlock.example.evaluator": "lm-evaluation-harness",
                "org.invarlock.example.evaluator-version": "0.4.12+invarlock.nocache.1",
                "org.invarlock.example.evaluator-lock-sha256": lock_sha256,
            },
            expected_entrypoint=[
                "python",
                "/opt/invarlock/examples/lm-evaluation-harness-example.py",
                "worker",
            ],
        )
        inspected_id = run(
            [
                args.container_engine,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                image_id,
            ],
            cwd=repository,
        )
        if inspected_id != image_id:
            raise RuntimeError("Harness image identity file disagrees with inspection")
        embedded_base_id = run(
            [
                args.container_engine,
                "image",
                "inspect",
                "--format",
                '{{index .Config.Labels "org.invarlock.example.base-image-id"}}',
                image_id,
            ],
            cwd=repository,
        )
        if embedded_base_id != base_id:
            raise RuntimeError(
                "Harness image does not bind the inspected base image ID"
            )
        write_evaluator_attestation(
            path=build / "evaluator-build-attestation.json",
            evaluator="lm-evaluation-harness",
            evaluator_version="0.4.12+invarlock.nocache.1",
            runtime_image_id=image_id,
            base_image_id=base_id,
            source_commit=commit,
            source_bundle_sha256=source_bundle_sha256,
            lock_sha256=lock_sha256,
            entrypoint=[
                "python",
                "/opt/invarlock/examples/lm-evaluation-harness-example.py",
                "worker",
            ],
            base_layers=base_layers,
            image_layers=harness_layers,
            config=harness_config,
            builder_signing_key=builder_signing_key,
        )

        prepared = workspace / "prepared"
        status("Preparing the pinned Qwen3-0.6B checkpoints and 102-record dataset...")
        run(
            [
                sys.executable,
                str(Path(__file__).with_name("model_inputs.py")),
                "--workspace",
                str(prepared),
                "--runtime-image",
                image_id,
            ],
            cwd=repository,
        )
        status(
            "Running the Harness in the inspected image and independently verifying "
            "the evidence transaction..."
        )
        output = run(
            [
                sys.executable,
                str(Path(__file__).with_name("example.py")),
                "complete",
                "--workspace",
                str(workspace / "transaction"),
                "--prepared",
                str(prepared),
                "--runtime-image",
                image_id,
                "--container-engine",
                args.container_engine,
                "--evidence-signing-key",
                str(Path(os.path.abspath(args.evidence_signing_key.expanduser()))),
                "--verifier-signing-key",
                str(Path(os.path.abspath(args.verifier_signing_key.expanduser())))
                if args.verifier_signing_key is not None
                else "",
                "--trust-root",
                str(Path(os.path.abspath(args.trust_root.expanduser()))),
                "--builder-public-key",
                str(Path(os.path.abspath(args.builder_public_key.expanduser()))),
                "--source-commit",
                commit,
                "--base-image-id",
                base_id,
                "--build-attestation",
                str(build / "evaluator-build-attestation.json"),
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
            print(f"FAIL {exc}", file=sys.stderr)
            cleanup_error = exc
            result = 2
        if cleanup_tags and cleanup_error is None:
            status("Removed temporary evaluator image tags.")
    if result:
        return result
    assert output is not None
    print(output)
    print(f"Complete integration workspace: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
