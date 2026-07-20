#!/usr/bin/env python3
"""Build the pinned Harness runtime and run its complete InvarLock journey."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def run(command: list[str], *, cwd: Path, stdin_path: Path | None = None) -> str:
    if stdin_path is None:
        completed = subprocess.run(
            command, cwd=cwd, check=False, capture_output=True, text=True
        )
    else:
        with stdin_path.open("rb") as handle:
            completed = subprocess.run(
                command,
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                stdin=handle,
            )
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed.stdout.strip()


def mount_source(path: Path) -> str:
    rendered = str(path)
    if any(character in rendered for character in (",", "\n", "\r", "\x00")):
        raise ValueError("workspace path cannot be represented in an OCI mount")
    return rendered


def status(message: str) -> None:
    """Show progress while long-running subprocess output remains captured."""

    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    from examples.integrations.launch import (
        _require_committed_checkout,
        _runtime_image,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    args = parser.parse_args(argv)
    repository = REPOSITORY
    if args.workspace is None:
        workspace = Path(tempfile.mkdtemp(prefix="invarlock-lm-eval-")).resolve(
            strict=True
        )
    else:
        workspace = args.workspace.expanduser().resolve()
        if workspace.exists() or workspace.is_symlink():
            print(f"FAIL workspace already exists: {workspace}", file=sys.stderr)
            return 2
        workspace.mkdir(parents=True)
    try:
        commit = _require_committed_checkout(repository)
        build = workspace / "build"
        build.mkdir()
        status("Building the source-bound InvarLock runtime image...")
        base_id, _ = _runtime_image(
            repository=repository,
            build_root=build,
            container_engine=args.container_engine,
        )
        base_tag = f"invarlock-example-runtime:{commit[:12]}"
        if (
            run(
                [
                    args.container_engine,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    base_tag,
                ],
                cwd=repository,
            )
            != base_id
        ):
            raise RuntimeError("base runtime tag changed after source-bound build")
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
        image_tag = f"invarlock-lm-eval-example:{commit[:12]}"
        status("Building the pinned LM Evaluation Harness image...")
        run(
            [
                args.container_engine,
                "build",
                "--file",
                "examples/integrations/lm-evaluation-harness/Dockerfile",
                "--build-arg",
                f"BASE_IMAGE={base_tag}",
                "--build-arg",
                f"BASE_IMAGE_ID={base_id}",
                "--tag",
                image_tag,
                "-",
            ],
            cwd=repository,
            stdin_path=source,
        )
        if (
            run(
                [
                    args.container_engine,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    base_tag,
                ],
                cwd=repository,
            )
            != base_id
        ):
            raise RuntimeError("base runtime tag changed during Harness image build")
        image_id = run(
            [
                args.container_engine,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                image_tag,
            ],
            cwd=repository,
        )
        if IMAGE_ID.fullmatch(image_id) is None:
            raise RuntimeError(
                "Harness image inspection did not return an immutable ID"
            )
        embedded_base_id = run(
            [
                args.container_engine,
                "image",
                "inspect",
                "--format",
                '{{index .Config.Labels "org.invarlock.example.base-image-id"}}',
                image_tag,
            ],
            cwd=repository,
        )
        if embedded_base_id != base_id:
            raise RuntimeError(
                "Harness image does not bind the inspected base image ID"
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
        harness_output = prepared / "harness"
        harness_output.mkdir()
        user = (
            f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "65532:65532"
        )
        for role in ("baseline", "subject"):
            status(f"Running the {role} Harness evaluation (CPU, batch 8)...")
            run(
                [
                    args.container_engine,
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--pull=never",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    "1024",
                    "--user",
                    user,
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,nodev,size=2g",
                    "--env",
                    "HOME=/tmp",
                    "--env",
                    "USER=invarlock-example",
                    "--env",
                    "LOGNAME=invarlock-example",
                    "--mount",
                    (
                        "type=bind,src="
                        f"{mount_source(prepared / 'evaluation/models' / role)},"
                        "dst=/model,readonly"
                    ),
                    "--mount",
                    (
                        "type=bind,src="
                        f"{mount_source(prepared / 'evaluation/inputs/records.jsonl')},"
                        "dst=/records.jsonl,readonly"
                    ),
                    "--mount",
                    (f"type=bind,src={mount_source(harness_output)},dst=/outputs"),
                    image_id,
                    "--role",
                    role,
                    "--model",
                    "/model",
                    "--dataset",
                    "/records.jsonl",
                    "--output",
                    f"/outputs/{role}",
                ],
                cwd=repository,
            )
        status("Creating and independently verifying the evidence transaction...")
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
            ],
            cwd=repository,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(output)
    print(f"Complete integration workspace: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
