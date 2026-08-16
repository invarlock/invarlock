"""Ownership-safe cleanup for temporary evaluator image tags."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

CommandRunner = Callable[..., str]


@dataclass(frozen=True)
class OwnedImageTag:
    """A temporary tag and the immutable image identity it named when built."""

    tag: str
    image_id: str


def temporary_image_tag(prefix: str, source_commit: str) -> str:
    """Return a per-invocation OCI tag that is impractical to collide with."""

    return f"{prefix}:{source_commit[:12]}-{secrets.token_hex(16)}"


def record_owned_image_tag(
    run: CommandRunner,
    engine: str,
    tag: str,
    image_id: str,
    repository: Path,
) -> OwnedImageTag:
    """Bind a newly built temporary tag to its engine-reported image ID."""

    try:
        observed = run(
            [engine, "image", "inspect", "--format", "{{.Id}}", tag],
            cwd=repository,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"temporary image tag was not created: {tag}") from exc
    if observed.strip() != image_id:
        raise RuntimeError(
            f"temporary image tag does not name the image built by this invocation: {tag}"
        )
    return OwnedImageTag(tag=tag, image_id=image_id)


def remove_owned_image_tags(
    run: CommandRunner,
    engine: str,
    repository: Path,
    tags: Sequence[OwnedImageTag],
) -> None:
    """Remove a temporary tag only while it still names the owned image ID."""

    failures: list[str] = []
    unique: dict[str, OwnedImageTag] = {}
    conflicting_tags: set[str] = set()
    for owned in tags:
        previous = unique.setdefault(owned.tag, owned)
        if previous.image_id != owned.image_id and owned.tag not in conflicting_tags:
            conflicting_tags.add(owned.tag)
            failures.append(
                f"{owned.tag}: conflicting owned image identities; refusing cleanup"
            )
    for owned in unique.values():
        if owned.tag in conflicting_tags:
            continue
        try:
            observed = run(
                [engine, "image", "inspect", "--format", "{{.Id}}", owned.tag],
                cwd=repository,
            ).strip()
        except RuntimeError as exc:
            diagnostic = str(exc)
            if any(
                marker in diagnostic.lower()
                for marker in ("no such image", "no such object", "image not known")
            ):
                continue
            failures.append(f"{owned.tag}: {diagnostic}")
            continue
        if observed != owned.image_id:
            failures.append(
                f"{owned.tag}: tag ownership changed; refusing to remove {observed}"
            )
            continue
        try:
            run([engine, "image", "rm", owned.tag], cwd=repository)
        except RuntimeError as exc:
            failures.append(f"{owned.tag}: {exc}")
    if failures:
        raise RuntimeError(
            "temporary evaluator image cleanup failed: " + "; ".join(failures)
        )


__all__ = [
    "OwnedImageTag",
    "record_owned_image_tag",
    "remove_owned_image_tags",
    "temporary_image_tag",
]
