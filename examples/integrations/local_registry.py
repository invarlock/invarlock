"""Disposable loopback registry for authenticated layered example images."""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
_REGISTRY_ENDPOINT = re.compile(r"^127\.0\.0\.1:([1-9][0-9]{0,4})$")
REGISTRY_IMAGE = (
    "registry@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373"
)


def _run(
    command: list[str], *, cwd: Path, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=capture_output,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"command exited with status {completed.returncode}: {' '.join(command)}"
            + (f"\n{diagnostic}" if diagnostic else "")
        )
    return completed


def _cleanup_command(command: list[str], *, cwd: Path) -> None:
    try:
        _run(command, cwd=cwd, capture_output=True)
    except (OSError, RuntimeError):
        pass


@contextmanager
def published_local_image(
    *,
    repository: Path,
    container_engine: str,
    image: str,
    image_digest: str,
    repository_name: str,
) -> Iterator[str]:
    """Publish one local image through an isolated loopback registry."""

    if _IMAGE_DIGEST.fullmatch(image_digest) is None:
        raise RuntimeError("local image publication requires a sha256 image digest")
    nonce = secrets.token_hex(8)
    container = f"invarlock-example-registry-{nonce}"
    volume = f"invarlock-example-registry-{nonce}"
    volume_created = False
    container_created = False
    published_tag: str | None = None
    try:
        created_volume = _run(
            [container_engine, "volume", "create", volume],
            cwd=repository,
            capture_output=True,
        ).stdout.strip()
        volume_created = True
        if created_volume != volume:
            raise RuntimeError(
                "container engine returned an unexpected registry volume"
            )
        created_container = _run(
            [
                container_engine,
                "run",
                "--detach",
                "--rm",
                "--pull=missing",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "128",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=64m",
                "--mount",
                f"type=volume,src={volume},dst=/var/lib/registry",
                "--publish",
                "127.0.0.1::5000",
                "--name",
                container,
                REGISTRY_IMAGE,
            ],
            cwd=repository,
            capture_output=True,
        ).stdout.strip()
        container_created = True
        if _CONTAINER_ID.fullmatch(created_container) is None:
            raise RuntimeError("container engine returned an invalid registry identity")
        endpoint = _run(
            [container_engine, "port", container, "5000/tcp"],
            cwd=repository,
            capture_output=True,
        ).stdout.strip()
        match = _REGISTRY_ENDPOINT.fullmatch(endpoint)
        if match is None or int(match.group(1)) > 65535:
            raise RuntimeError("container engine returned an invalid registry endpoint")
        published_tag = (
            f"{endpoint}/{repository_name}:{image_digest.removeprefix('sha256:')[:12]}"
        )
        _run([container_engine, "image", "tag", image, published_tag], cwd=repository)
        push = [container_engine, "push"]
        if container_engine == "podman":
            push.append("--tls-verify=false")
        push.append(published_tag)
        _run(push, cwd=repository)
        inspected = _run(
            [
                container_engine,
                "image",
                "inspect",
                "--format",
                "{{.Id}} {{json .RepoDigests}}",
                published_tag,
            ],
            cwd=repository,
            capture_output=True,
        ).stdout.strip()
        observed_id, separator, raw_digests = inspected.partition(" ")
        if not separator or observed_id != image_digest:
            raise RuntimeError("published registry image changed identity")
        try:
            digests = json.loads(raw_digests)
        except json.JSONDecodeError as exc:
            raise RuntimeError("published registry digests are unreadable") from exc
        if not isinstance(digests, list):
            raise RuntimeError("published registry digests have an invalid shape")
        prefix = f"{endpoint}/{repository_name}@"
        candidates = [
            value
            for value in digests
            if isinstance(value, str)
            and value.startswith(prefix)
            and _IMAGE_DIGEST.fullmatch(value.removeprefix(prefix)) is not None
        ]
        if len(candidates) != 1:
            raise RuntimeError("published registry image lacks one canonical digest")
        yield candidates[0]
    finally:
        if published_tag is not None:
            _cleanup_command(
                [container_engine, "image", "remove", published_tag], cwd=repository
            )
        if container_created:
            _cleanup_command(
                [container_engine, "container", "rm", "--force", container],
                cwd=repository,
            )
        if volume_created:
            _cleanup_command(
                [container_engine, "volume", "rm", "--force", volume], cwd=repository
            )
