"""Shared destination validation and lazy report publication mechanics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path


class ReportPublicationError(ValueError):
    """Raised after a report output could not be published safely."""

    def __init__(
        self,
        message: str,
        *,
        failed_output: str | None = None,
        written_outputs: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_output = failed_output
        self.written_outputs = dict(written_outputs or {})


def validate_report_destinations(
    requested: Mapping[str, str | Path], *, evidence: Path | None = None
) -> dict[str, Path]:
    """Validate all requested destinations before invoking any renderer."""
    evidence_root = evidence.absolute() if evidence is not None else None
    destinations: dict[str, Path] = {}
    canonical: dict[str, Path] = {}
    for name, value in requested.items():
        destination = Path(value).absolute()
        if destination.name in {"", ".", ".."}:
            raise ReportPublicationError("report destination must name a regular file")
        resolved = destination.resolve()
        if evidence_root is not None and (
            destination.is_relative_to(evidence_root)
            or resolved.is_relative_to(evidence_root.resolve())
        ):
            raise ReportPublicationError(
                "report destination must remain outside the immutable evidence pack"
            )
        if any(
            resolved.is_relative_to(other) or other.is_relative_to(resolved)
            for other in canonical.values()
        ):
            raise ReportPublicationError("report destinations collide")
        if destination.exists() or destination.is_symlink():
            raise ReportPublicationError(
                f"report destination already exists: {destination}"
            )
        for parent in destination.parents:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ReportPublicationError(
                    "report destination parent must be a real directory"
                )
        destinations[name] = destination
        canonical[name] = resolved
    return destinations


def publish_report_outputs(
    requested: Mapping[str, str | Path],
    renderers: Mapping[str, Callable[[], bytes]],
    *,
    evidence: Path | None = None,
) -> dict[str, str]:
    """Validate, lazily render, and no-clobber publish requested outputs."""
    destinations = validate_report_destinations(requested, evidence=evidence)
    # Resolve this at publication time so existing captured/evidence tests and
    # their hardened writer instrumentation observe every output uniformly.
    from invarlock.captured_contracts import atomic_write

    rendered: dict[str, bytes] = {}
    for name in destinations:
        try:
            renderer = renderers[name]
            payload = renderer()
            if not isinstance(payload, bytes):
                raise TypeError(f"renderer for {name} did not return bytes")
        except Exception as exc:
            raise ReportPublicationError(
                str(exc), failed_output=None, written_outputs={}
            ) from exc
        rendered[name] = payload

    written: dict[str, str] = {}
    for name, destination in destinations.items():
        try:
            atomic_write(destination, rendered[name])
        except Exception as exc:
            raise ReportPublicationError(
                str(exc), failed_output=name, written_outputs=written
            ) from exc
        # Preserve the caller-facing spelling in the report contract while
        # using the normalized path above for safety checks and I/O.
        written[name] = str(requested[name])
    return written


__all__ = [
    "ReportPublicationError",
    "publish_report_outputs",
    "validate_report_destinations",
]
