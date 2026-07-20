#!/usr/bin/env python3
"""Build the exact runtime image and launch one integration journey."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_INTEGRATIONS = ("hf-transformers", "peft-lora", "torchao-int8")
_ZERO_DIGEST = "sha256:" + ("0" * 64)


def _run(
    command: list[str],
    *,
    cwd: Path,
    capture_output: bool = False,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=capture_output,
        text=True,
        env=environment if environment is not None else os.environ.copy(),
    )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"command exited with status {completed.returncode}: {' '.join(command)}"
            + (f"\n{diagnostic}" if diagnostic else "")
        )
    return completed


def _git(repository: Path, *arguments: str) -> str:
    return _run(
        ["git", "-C", str(repository), *arguments],
        cwd=repository,
        capture_output=True,
    ).stdout.strip()


def _require_committed_checkout(repository: Path) -> str:
    commit = _git(repository, "rev-parse", "--verify", "HEAD^{commit}")
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(
            "the example runtime is built from committed source; commit or stash "
            "tracked changes before running the full journey"
        )
    return commit


def _runtime_image(
    *,
    repository: Path,
    build_root: Path,
    container_engine: str,
    dockerfile: str = "runtime/Dockerfile",
    image_prefix: str = "invarlock-example-runtime",
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
    image = f"{image_prefix}:{commit[:12]}"
    _run(
        [
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
        ],
        cwd=repository,
    )
    inspected = _run(
        [container_engine, "image", "inspect", "--format", "{{.Id}}", image],
        cwd=repository,
        capture_output=True,
    ).stdout.strip()
    if not inspected.startswith("sha256:") or len(inspected) != 71:
        raise RuntimeError("container inspection did not return a sha256 image ID")
    # The engine-reported local image ID is an immutable execution coordinate.
    # The runtime boundary requires that coordinate as both the reference and
    # digest; combining a mutable tag with a separate identity must fail closed.
    return inspected, inspected


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
    repository = Path(__file__).resolve().parents[2]
    workspace = (
        arguments.workspace.expanduser().resolve()
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
        runtime_device = _resolve_runtime_device(arguments.runtime_device)
        if arguments.prepare_only:
            image = None
            image_digest = _ZERO_DIGEST
        else:
            build_root = workspace / "build"
            build_root.mkdir()
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
        command = [
            sys.executable,
            str(repository / "examples/integrations/run.py"),
            arguments.integration,
            "--workspace",
            str(transaction),
            "--runtime-image-digest",
            image_digest,
            "--container-engine",
            arguments.container_engine,
            "--runtime-device",
            runtime_device,
        ]
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
