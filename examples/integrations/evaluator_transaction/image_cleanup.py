"""Ownership-safe cleanup for temporary evaluator image tags."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

CommandRunner = Callable[..., str]


def require_image_tag_available(
    run: CommandRunner, engine: str, tag: str, repository: Path
) -> None:
    """Refuse to overwrite a tag that this invocation cannot safely own."""

    try:
        run([engine, "image", "inspect", "--format", "{{.Id}}", tag], cwd=repository)
    except RuntimeError as exc:
        diagnostic = str(exc).lower()
        if any(
            marker in diagnostic
            for marker in ("no such image", "no such object", "image not known")
        ):
            return
        raise
    raise RuntimeError(
        f"temporary evaluator image tag already exists; remove it before running: {tag}"
    )


def remove_temporary_image_tags(
    run: CommandRunner,
    engine: str,
    repository: Path,
    tags: Sequence[str],
) -> None:
    """Remove only the exact temporary tags created by this invocation."""

    failures: list[str] = []
    for tag in dict.fromkeys(tags):
        try:
            run([engine, "image", "rm", "--force", tag], cwd=repository)
        except RuntimeError as exc:
            diagnostic = str(exc)
            if not any(
                marker in diagnostic.lower()
                for marker in ("no such image", "no such object", "image not known")
            ):
                failures.append(f"{tag}: {diagnostic}")
    if failures:
        raise RuntimeError(
            "temporary evaluator image cleanup failed: " + "; ".join(failures)
        )


__all__ = ["remove_temporary_image_tags", "require_image_tag_available"]
