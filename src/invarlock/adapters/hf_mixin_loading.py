"""Loading diagnostics shared by Hugging Face adapter implementations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HFPretrainedLoadDiagnostic:
    """A normalized group of messages emitted by ``from_pretrained``."""

    kind: str
    entries: tuple[str, ...]


def _is_local_loader_cache_miss(error: Exception) -> bool:
    """Return whether a local-only loader failure is a missing-cache signal."""

    if isinstance(error, FileNotFoundError):
        return True
    if not isinstance(error, OSError):
        return False
    message = str(error).strip().lower()
    return any(
        snippet in message
        for snippet in (
            "no such file",
            "not found",
            "could not locate",
            "does not appear to have a file named",
            "missing cached",
            "local files only",
            "cannot find",
            "can't load the model",
        )
    )


__all__ = ["HFPretrainedLoadDiagnostic"]
