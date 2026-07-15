"""Core file, digest, and strict report contracts for release evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def existing_globs(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(matches)


def require_file(path: Path, label: str, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"{label} missing: {path}")


def require_any(
    root: Path, patterns: tuple[str, ...], label: str, failures: list[str]
) -> None:
    if not existing_globs(root, patterns):
        joined = ", ".join(patterns)
        failures.append(f"{label} missing under {root}: {joined}")


def load_json(path: Path, label: str, failures: list[str]) -> object | None:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        return payload
    except FileNotFoundError:
        failures.append(f"{label} missing: {path}")
    except json.JSONDecodeError as exc:
        failures.append(f"{label} is not valid JSON: {path}: {exc}")
    return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r}")


def _read_regular_snapshot(
    path: Path,
    *,
    label: str,
    failures: list[str],
) -> bytes | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        failures.append(f"{label} must be a readable regular file: {path}: {exc}")
        return None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            failures.append(f"{label} must be a regular file: {path}")
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
    except OSError as exc:
        failures.append(f"{label} could not be read: {path}: {exc}")
        return None
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        failures.append(f"{label} changed while it was being read: {path}")
        return None
    return raw


def _strict_json_snapshot(
    path: Path,
    *,
    label: str,
    failures: list[str],
) -> object | None:
    raw = _read_regular_snapshot(path, label=label, failures=failures)
    if raw is None:
        return None
    try:
        return cast(
            object,
            json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            ),
        )
    except (UnicodeError, ValueError) as exc:
        failures.append(f"{label} is not strict JSON: {path}: {exc}")
        return None


@dataclass(frozen=True)
class DistHashManifest:
    entries: dict[str, str]

    @classmethod
    def load(cls, path: Path, failures: list[str]) -> DistHashManifest:
        entries: dict[str, str] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return cls(entries)
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0].lower()):
                failures.append(
                    f"wheel/sdist hashes line {line_number} must use sha256sum format."
                )
                continue
            digest = parts[0].lower()
            name = parts[1].strip()
            if name.startswith("*"):
                name = name[1:]
            name = name.removeprefix("./")
            entries[name] = digest
            entries[Path(name).name] = digest
        return cls(entries)

    def validate_artifacts(self, *, dist_root: Path, failures: list[str]) -> None:
        artifacts = existing_globs(dist_root, ("*.whl", "*.tar.gz"))
        if not artifacts:
            return
        if not self.entries:
            failures.append("wheel/sdist hashes file has no valid entries.")
            return
        for artifact in artifacts:
            candidates = {
                artifact.name,
                artifact.relative_to(dist_root).as_posix(),
                f"{dist_root.name}/{artifact.name}",
            }
            expected = next(
                (self.entries[name] for name in candidates if name in self.entries),
                None,
            )
            if expected is None:
                failures.append(
                    f"wheel/sdist hash missing for artifact: {artifact.name}"
                )
                continue
            if sha256(artifact) != expected:
                failures.append(
                    f"wheel/sdist hash mismatch for artifact: {artifact.name}"
                )


@dataclass(frozen=True)
class StrictReportEvidence:
    path: Path
    payload: dict[str, Any] | None

    @classmethod
    def load(cls, path: Path, failures: list[str]) -> StrictReportEvidence:
        payload = load_json(path, "strict example report", failures)
        if payload is not None and not isinstance(payload, dict):
            failures.append("strict example report must be a JSON object.")
            return cls(path, None)
        return cls(path, payload if isinstance(payload, dict) else None)

    def validate(self, failures: list[str]) -> None:
        if self.payload is None:
            return
        assurance = self.payload.get("assurance")
        if not isinstance(assurance, dict):
            failures.append("strict example report missing assurance object.")
            return
        if assurance.get("mode") != "strict":
            failures.append("strict example report assurance.mode must be strict.")
        if assurance.get("verdict") not in {"pending_verifier", "pass"}:
            failures.append(
                "strict example report assurance.verdict must be pending_verifier or pass."
            )
        if assurance.get("fallback_fields_used") is not False:
            failures.append(
                "strict example report assurance.fallback_fields_used must be false."
            )
        report_build = self.payload.get("report_build")
        if not isinstance(report_build, dict):
            failures.append("strict example report missing report_build object.")
            return
        for field in ("synthesized_fields", "repaired_fields", "fallback_fields"):
            if report_build.get(field, []):
                failures.append(
                    f"strict example report report_build.{field} must be empty."
                )


@dataclass(frozen=True)
class StrictVerifyEvidence:
    path: Path
    payload: dict[str, Any] | None

    @classmethod
    def load(cls, path: Path, failures: list[str]) -> StrictVerifyEvidence:
        payload = load_json(path, "strict verifier output", failures)
        if payload is not None and not isinstance(payload, dict):
            failures.append("strict verifier output must be a JSON object.")
            return cls(path, None)
        return cls(path, payload if isinstance(payload, dict) else None)

    def validate(self, *, report_path: Path, failures: list[str]) -> None:
        if self.payload is None:
            return
        summary = self.payload.get("summary")
        if not isinstance(summary, dict) or summary.get("ok") is not True:
            failures.append("strict verifier output summary.ok must be true.")
        results = self.payload.get("results")
        if not isinstance(results, list) or not results:
            failures.append("strict verifier output must include at least one result.")
            return
        expected_report = report_path.resolve()
        report_results = []
        for result in results:
            if not isinstance(result, dict):
                continue
            raw_id = result.get("id")
            if not isinstance(raw_id, str) or not raw_id:
                continue
            try:
                candidate = Path(raw_id).resolve()
            except OSError:
                continue
            if candidate == expected_report:
                report_results.append(result)
        if not report_results:
            failures.append(
                "strict verifier output does not reference the strict report."
            )
        provenance_pinned = False
        for result in report_results:
            verification = result.get("verification")
            if not isinstance(verification, dict):
                continue
            provenance = verification.get("runtime_provenance")
            if not isinstance(provenance, dict):
                continue
            provenance_pinned = (
                provenance.get("status") == "expected_image_digest_matched"
                and provenance.get("verified") is True
                and provenance.get("binding_verified") is True
                and provenance.get("expected_digest_matched") is True
            )
            if provenance_pinned:
                break
        if not provenance_pinned:
            failures.append(
                "strict verifier output must prove report/manifest binding to a "
                "independently supplied runtime image digest pin."
            )
