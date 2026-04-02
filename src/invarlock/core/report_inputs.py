from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CANONICAL_REPORT_FILENAMES: tuple[str, str] = ("report.json", "evaluation.report.json")


class ReportInputError(ValueError):
    """Raised when a report-like input cannot be resolved or decoded safely."""

    def __init__(self, reason: str, path: Path, *, detail: str | None = None) -> None:
        self.reason = reason
        self.path = path
        self.detail = detail
        super().__init__(self._message())

    def _message(self) -> str:
        if self.reason == "not_found":
            return f"Path not found: {self.path}"
        if self.reason == "directory_forbidden":
            return f"Report must be an explicit JSON file path, not a directory: {self.path}"
        if self.reason == "ambiguous_directory":
            return (
                "Ambiguous report directory "
                f"{self.path}: contains both report.json and evaluation.report.json; "
                "pass an explicit file path."
            )
        if self.reason == "missing_canonical":
            return (
                f"Directory {self.path} does not contain a canonical report file "
                "(report.json or evaluation.report.json); pass an explicit file path."
            )
        if self.reason == "non_regular":
            return f"Path is not a regular report file: {self.path}"
        if self.reason == "unreadable":
            return f"Report is not readable: {self.path} ({self.detail})"
        if self.reason == "invalid_json":
            return f"Report is not valid JSON: {self.path} ({self.detail})"
        if self.reason == "non_object":
            return f"Report must decode to a JSON object: {self.path}"
        return f"Invalid report input: {self.path}"


def resolve_report_input_path(
    path_value: str | Path,
    *,
    allow_canonical_directory: bool = True,
) -> Path:
    """Resolve a report-like input to a concrete JSON file path.

    Explicit files are always accepted. Directories are accepted only when they
    contain exactly one canonical artifact name from
    ``CANONICAL_REPORT_FILENAMES``.
    """

    candidate = Path(path_value).expanduser()
    if not candidate.exists():
        raise ReportInputError("not_found", candidate)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_dir():
        if not allow_canonical_directory:
            raise ReportInputError("directory_forbidden", candidate)
        canonical_matches = [
            candidate / name
            for name in CANONICAL_REPORT_FILENAMES
            if (candidate / name).is_file()
        ]
        if len(canonical_matches) > 1:
            raise ReportInputError("ambiguous_directory", candidate)
        if len(canonical_matches) == 1:
            return canonical_matches[0].resolve()
        raise ReportInputError("missing_canonical", candidate)
    raise ReportInputError("non_regular", candidate)


def load_report_input_json(
    path_value: str | Path,
    *,
    allow_canonical_directory: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and load a report-like JSON object."""

    resolved = resolve_report_input_path(
        path_value,
        allow_canonical_directory=allow_canonical_directory,
    )
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ReportInputError("unreadable", resolved, detail=str(exc)) from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ReportInputError("invalid_json", resolved, detail=str(exc)) from exc
    if not isinstance(payload, dict):
        raise ReportInputError("non_object", resolved)
    return resolved, payload


__all__ = [
    "CANONICAL_REPORT_FILENAMES",
    "ReportInputError",
    "load_report_input_json",
    "resolve_report_input_path",
]
