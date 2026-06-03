from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

RUN_REPORT_FILENAME = "report.json"
EVALUATION_REPORT_FILENAME = "evaluation.report.json"
CANONICAL_REPORT_FILENAMES: tuple[str, str] = (
    RUN_REPORT_FILENAME,
    EVALUATION_REPORT_FILENAME,
)


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
        if self.reason == "missing_run_canonical":
            return (
                f"Directory {self.path} does not contain a canonical run report file "
                f"({RUN_REPORT_FILENAME}); pass an explicit run report path."
            )
        if self.reason == "missing_evaluation_canonical":
            run_report_path = self.path / RUN_REPORT_FILENAME
            if run_report_path.is_file():
                return (
                    f"Directory {self.path} contains {RUN_REPORT_FILENAME}, which is "
                    "a raw run report. Verification expects the canonical "
                    f"{EVALUATION_REPORT_FILENAME}; generate one with "
                    "`invarlock report generate --run <subject report.json> "
                    "--baseline-run-report <baseline report.json> --format report "
                    "-o <output-dir>` or pass an explicit evaluation report path."
                )
            return (
                f"Directory {self.path} does not contain a canonical evaluation report file "
                f"({EVALUATION_REPORT_FILENAME}); pass an explicit evaluation report path."
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
        if self.reason == "expected_run_payload":
            return (
                f"Expected a run report payload, not an evaluation bundle: {self.path} "
                f"({self.detail})"
            )
        if self.reason == "expected_evaluation_payload":
            return (
                f"Expected an evaluation report payload, not a run report artifact: "
                f"{self.path} ({self.detail})"
            )
        if self.reason == "missing_linked_run_reports":
            detail = (
                self.detail
                or "report provenance does not record both linked run report paths"
            )
            return (
                "Evaluation report bundle does not expose both linked run reports: "
                f"{self.path} ({detail})"
            )
        return f"Invalid report input: {self.path}"


def resolve_report_input_path(
    path_value: str | Path,
    *,
    allow_canonical_directory: bool = True,
    expected_kind: Literal["any", "run", "evaluation"] = "any",
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
        if expected_kind == "run":
            run_candidate = candidate / RUN_REPORT_FILENAME
            if run_candidate.is_file():
                return run_candidate.resolve()
            raise ReportInputError("missing_run_canonical", candidate)
        if expected_kind == "evaluation":
            evaluation_candidate = candidate / EVALUATION_REPORT_FILENAME
            if evaluation_candidate.is_file():
                return evaluation_candidate.resolve()
            raise ReportInputError("missing_evaluation_canonical", candidate)
        if len(canonical_matches) == 1:
            return canonical_matches[0].resolve()
        raise ReportInputError("missing_canonical", candidate)
    raise ReportInputError("non_regular", candidate)


def load_report_input_json(
    path_value: str | Path,
    *,
    allow_canonical_directory: bool = True,
    expected_kind: Literal["any", "run", "evaluation"] = "any",
) -> tuple[Path, dict[str, Any]]:
    """Resolve and load a report-like JSON object."""

    resolved = resolve_report_input_path(
        path_value,
        allow_canonical_directory=allow_canonical_directory,
        expected_kind=expected_kind,
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


def load_run_report_input_json(
    path_value: str | Path,
    *,
    allow_canonical_directory: bool = True,
) -> tuple[Path, dict[str, Any]]:
    resolved, payload = load_report_input_json(
        path_value,
        allow_canonical_directory=allow_canonical_directory,
        expected_kind="run",
    )
    if isinstance(payload.get("validation"), dict):
        raise ReportInputError(
            "expected_run_payload",
            resolved,
            detail=(
                "pass the report.json artifact emitted by the baseline or subject "
                "side of invarlock evaluate"
            ),
        )
    return resolved, payload


def load_evaluation_report_input_json(
    path_value: str | Path,
    *,
    allow_canonical_directory: bool = True,
) -> tuple[Path, dict[str, Any]]:
    resolved, payload = load_report_input_json(
        path_value,
        allow_canonical_directory=allow_canonical_directory,
        expected_kind="evaluation",
    )
    if not isinstance(payload.get("validation"), dict):
        raise ReportInputError(
            "expected_evaluation_payload",
            resolved,
            detail=(
                "pass the evaluation.report.json artifact emitted by "
                "invarlock evaluate or invarlock report generate"
            ),
        )
    return resolved, payload


def _resolve_linked_report_path(
    path_value: object,
    *,
    base_dir: Path,
    bundle_path: Path,
) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ReportInputError(
            "missing_linked_run_reports",
            bundle_path,
            detail="expected provenance.edited.report_path and provenance.baseline.report_path",
        )
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    resolved = candidate.resolve()
    if not resolved.exists():
        raise ReportInputError("not_found", resolved)
    if not resolved.is_file():
        raise ReportInputError("non_regular", resolved)
    return resolved


def resolve_run_reports_from_evaluation_input(
    path_value: str | Path,
    *,
    allow_canonical_directory: bool = True,
) -> tuple[Path, Path, Path]:
    """Resolve linked subject and baseline run reports from an evaluation bundle."""

    evaluation_path, payload = load_evaluation_report_input_json(
        path_value,
        allow_canonical_directory=allow_canonical_directory,
    )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ReportInputError(
            "missing_linked_run_reports",
            evaluation_path,
            detail="missing provenance block",
        )
    edited = provenance.get("edited")
    baseline = provenance.get("baseline")
    if not isinstance(edited, dict) or not isinstance(baseline, dict):
        raise ReportInputError(
            "missing_linked_run_reports",
            evaluation_path,
            detail="missing provenance.edited or provenance.baseline block",
        )
    base_dir = evaluation_path.parent
    subject_report_path = _resolve_linked_report_path(
        edited.get("report_path"),
        base_dir=base_dir,
        bundle_path=evaluation_path,
    )
    baseline_report_path = _resolve_linked_report_path(
        baseline.get("report_path"),
        base_dir=base_dir,
        bundle_path=evaluation_path,
    )
    load_run_report_input_json(
        subject_report_path,
        allow_canonical_directory=False,
    )
    load_run_report_input_json(
        baseline_report_path,
        allow_canonical_directory=False,
    )
    return evaluation_path, subject_report_path, baseline_report_path


__all__ = [
    "CANONICAL_REPORT_FILENAMES",
    "EVALUATION_REPORT_FILENAME",
    "ReportInputError",
    "RUN_REPORT_FILENAME",
    "load_evaluation_report_input_json",
    "load_report_input_json",
    "load_run_report_input_json",
    "resolve_run_reports_from_evaluation_input",
    "resolve_report_input_path",
]
