"""Typed contracts for release and empirical evidence validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.release.evidence_contracts_empirical import (
        _MANIFEST_OBJECT_ERROR,
        ALLOWED_EVIDENCE_KINDS,
        EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA,
        EMPIRICAL_GUARD_EVIDENCE_SCHEMA,
        REAL_PRODUCER_MARKERS,
        REQUIRED_GUARDS,
        EmpiricalGuardEvidenceManifest,
        GuardEvidenceRow,
        ModelFamilyEvidenceRow,
        resolve_artifact,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from evidence_contracts_empirical import (
        _MANIFEST_OBJECT_ERROR,
        ALLOWED_EVIDENCE_KINDS,
        EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA,
        EMPIRICAL_GUARD_EVIDENCE_SCHEMA,
        REAL_PRODUCER_MARKERS,
        REQUIRED_GUARDS,
        EmpiricalGuardEvidenceManifest,
        GuardEvidenceRow,
        ModelFamilyEvidenceRow,
        resolve_artifact,
    )

RELEASE_CHECK_SCHEMA = "invarlock/release-evidence-check-v1"
OFFLINE_BUNDLE_SCHEMA = "invarlock/release-offline-bundle-v1"
GUARD_VALIDATION_SMOKE_SCHEMA = "invarlock/guard-validation-smoke-v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

__all__ = [
    "ALLOWED_EVIDENCE_KINDS",
    "EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA",
    "EMPIRICAL_GUARD_EVIDENCE_SCHEMA",
    "EmpiricalGuardEvidenceManifest",
    "GuardEvidenceRow",
    "ModelFamilyEvidenceRow",
    "REAL_PRODUCER_MARKERS",
    "REQUIRED_GUARDS",
]


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
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(f"{label} missing: {path}")
    except json.JSONDecodeError as exc:
        failures.append(f"{label} is not valid JSON: {path}: {exc}")
    return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        result_names = {
            Path(str(result.get("id", ""))).name
            for result in results
            if isinstance(result, dict)
        }
        if report_path.name not in result_names:
            failures.append(
                "strict verifier output does not reference the strict report."
            )
        provenance_verified = False
        for result in results:
            if not isinstance(result, dict):
                continue
            verification = result.get("verification")
            if not isinstance(verification, dict):
                continue
            provenance = verification.get("runtime_provenance")
            if not isinstance(provenance, dict):
                continue
            provenance_verified = (
                provenance.get("status") == "verified"
                and provenance.get("verified") is True
            )
            if provenance_verified:
                break
        if not provenance_verified:
            failures.append(
                "strict verifier output must contain verified runtime provenance."
            )


@dataclass(frozen=True)
class GuardValidationSmokeManifest:
    json_path: Path
    markdown_path: Path
    payload: dict[str, Any] | None

    @classmethod
    def load(
        cls, *, json_path: Path, markdown_path: Path, failures: list[str]
    ) -> GuardValidationSmokeManifest:
        payload = load_json(json_path, "guard-validation JSON", failures)
        if payload is not None and not isinstance(payload, dict):
            failures.append("guard-validation JSON must be a JSON object.")
            payload = None
        return cls(json_path, markdown_path, payload)

    def validate(self, failures: list[str]) -> None:
        if self.payload is not None:
            if self.payload.get("schema") != GUARD_VALIDATION_SMOKE_SCHEMA:
                failures.append("guard-validation JSON schema is not recognized.")
            rows = self.payload.get("rate_rows")
            if not isinstance(rows, list):
                failures.append("guard-validation JSON missing rate_rows list.")
            else:
                guards = {
                    str(row.get("guard"))
                    for row in rows
                    if isinstance(row, dict) and row.get("guard")
                }
                missing = sorted(REQUIRED_GUARDS - guards)
                if missing:
                    failures.append(
                        "guard-validation JSON missing guard rows: "
                        + ", ".join(missing)
                    )
        require_file(self.markdown_path, "guard-validation markdown", failures)
        if (
            self.markdown_path.is_file()
            and not self.markdown_path.read_text(encoding="utf-8").strip()
        ):
            failures.append("guard-validation markdown must not be empty.")


@dataclass(frozen=True)
class OfflineBundleManifest:
    bundle_path: Path
    payload: dict[str, Any] | None

    @classmethod
    def load_from_tarball(
        cls, bundle: Path, failures: list[str]
    ) -> OfflineBundleManifest:
        try:
            with tarfile.open(bundle, "r:gz") as tar:
                manifest_members = [
                    member
                    for member in tar.getmembers()
                    if member.isfile()
                    and Path(member.name).name == "release_manifest.json"
                ]
                if not manifest_members:
                    failures.append(
                        f"offline release bundle manifest missing: {bundle}"
                    )
                    return cls(bundle, None)
                extracted = tar.extractfile(manifest_members[0])
                if extracted is None:
                    failures.append(
                        f"offline release bundle manifest unreadable: {bundle}"
                    )
                    return cls(bundle, None)
                manifest = json.loads(extracted.read().decode("utf-8"))
        except (
            tarfile.TarError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            failures.append(f"offline release bundle invalid: {bundle}: {exc}")
            return cls(bundle, None)
        if not isinstance(manifest, dict):
            failures.append(
                f"offline release bundle manifest must be an object: {bundle}"
            )
            return cls(bundle, None)
        return cls(bundle, manifest)

    def validate(self, failures: list[str]) -> bool:
        if self.payload is None:
            return False
        if self.payload.get("schema") != OFFLINE_BUNDLE_SCHEMA:
            failures.append(
                f"offline release bundle schema is not recognized: {self.bundle_path}"
            )
            return False
        distributions = self.payload.get("distributions")
        if not isinstance(distributions, list) or not distributions:
            failures.append(
                f"offline release bundle has no distributions: {self.bundle_path}"
            )
            return False
        dist_paths = {
            str(item.get("path", ""))
            for item in distributions
            if isinstance(item, dict)
        }
        if not any(path.endswith(".whl") for path in dist_paths):
            failures.append(
                f"offline release bundle missing wheel distribution: {self.bundle_path}"
            )
            return False
        if not any(path.endswith(".tar.gz") for path in dist_paths):
            failures.append(
                f"offline release bundle missing sdist distribution: {self.bundle_path}"
            )
            return False
        return True


@dataclass(frozen=True)
class ReleaseEvidenceManifest:
    release_root: Path
    dist_root: Path
    sbom_path: Path
    guard_validation_json: Path
    guard_validation_markdown: Path
    offline_bundle_dir: Path

    def validate(self) -> list[str]:
        failures: list[str] = []
        require_any(self.dist_root, ("*.whl",), "wheel artifact", failures)
        require_any(self.dist_root, ("*.tar.gz",), "sdist artifact", failures)
        require_file(self.sbom_path, "SBOM", failures)
        hash_path = self.release_root / "wheel-sdist-hashes.txt"
        runtime_digest_path = self.release_root / "runtime-image-digest.txt"
        strict_report_path = self.release_root / "strict" / "evaluation.report.json"
        strict_verify_path = self.release_root / "strict" / "verify.json"
        require_file(hash_path, "wheel/sdist hashes", failures)
        require_file(runtime_digest_path, "runtime image digest", failures)
        require_file(strict_report_path, "strict example report", failures)
        require_file(strict_verify_path, "strict verifier output", failures)
        self._validate_sbom(failures)
        if self.dist_root.is_dir() and hash_path.is_file():
            hash_manifest = DistHashManifest.load(hash_path, failures)
            if not hash_manifest.entries:
                failures.append(
                    f"wheel/sdist hashes file has no valid entries: {hash_path}"
                )
            else:
                hash_manifest.validate_artifacts(
                    dist_root=self.dist_root, failures=failures
                )
        self._validate_runtime_digest(runtime_digest_path, failures)
        StrictReportEvidence.load(strict_report_path, failures).validate(failures)
        StrictVerifyEvidence.load(strict_verify_path, failures).validate(
            report_path=strict_report_path,
            failures=failures,
        )
        GuardValidationSmokeManifest.load(
            json_path=self.guard_validation_json,
            markdown_path=self.guard_validation_markdown,
            failures=failures,
        ).validate(failures)
        self._validate_offline_bundles(failures)
        return failures

    def _validate_sbom(self, failures: list[str]) -> None:
        payload = load_json(self.sbom_path, "SBOM", failures)
        if payload is not None and not isinstance(payload, dict):
            failures.append("SBOM must be a JSON object.")

    @staticmethod
    def _validate_runtime_digest(path: Path, failures: list[str]) -> None:
        if not path.is_file():
            return
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(lines) != 1 or not RUNTIME_DIGEST_RE.fullmatch(lines[0]):
            failures.append(
                "runtime image digest must contain exactly one sha256:<64 hex> digest."
            )

    def _validate_offline_bundles(self, failures: list[str]) -> None:
        bundles = existing_globs(self.offline_bundle_dir, ("*.tar.gz",))
        if not bundles:
            failures.append(
                f"offline release bundle missing under {self.offline_bundle_dir}: *.tar.gz"
            )
            return
        valid_manifest_found = False
        for bundle in bundles:
            manifest = OfflineBundleManifest.load_from_tarball(bundle, failures)
            valid_manifest_found = manifest.validate(failures) or valid_manifest_found
        if not valid_manifest_found:
            failures.append("no valid offline release bundle manifest found.")

    def summary(self, failures: list[str]) -> dict[str, object]:
        return {
            "schema": RELEASE_CHECK_SCHEMA,
            "release_root": str(self.release_root),
            "dist_root": str(self.dist_root),
            "sbom_path": str(self.sbom_path),
            "guard_validation_json": str(self.guard_validation_json),
            "guard_validation_markdown": str(self.guard_validation_markdown),
            "offline_bundle_dir": str(self.offline_bundle_dir),
            "ok": not failures,
            "failures": failures,
        }


def _existing_globs(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    return existing_globs(root, patterns)


def _require_file(path: Path, label: str, failures: list[str]) -> None:
    require_file(path, label, failures)


def _require_any(
    root: Path, patterns: tuple[str, ...], label: str, failures: list[str]
) -> None:
    require_any(root, patterns, label, failures)


def _dist_artifacts(dist_root: Path) -> list[Path]:
    return existing_globs(dist_root, ("*.whl", "*.tar.gz"))


def _sha256(path: Path) -> str:
    return sha256(path)


def _load_json(path: Path, label: str, failures: list[str]) -> object | None:
    return load_json(path, label, failures)


def _parse_hash_entries(path: Path, failures: list[str]) -> dict[str, str]:
    return DistHashManifest.load(path, failures).entries


def _validate_dist_hashes(
    *, dist_root: Path, hash_path: Path, failures: list[str]
) -> None:
    if not _dist_artifacts(dist_root) or not hash_path.is_file():
        return
    manifest = DistHashManifest.load(hash_path, failures)
    if not manifest.entries:
        failures.append(f"wheel/sdist hashes file has no valid entries: {hash_path}")
        return
    manifest.validate_artifacts(dist_root=dist_root, failures=failures)


def _validate_runtime_digest(path: Path, failures: list[str]) -> None:
    ReleaseEvidenceManifest._validate_runtime_digest(path, failures)


def _validate_strict_report(path: Path, failures: list[str]) -> None:
    StrictReportEvidence.load(path, failures).validate(failures)


def _validate_strict_verify(path: Path, report_path: Path, failures: list[str]) -> None:
    StrictVerifyEvidence.load(path, failures).validate(
        report_path=report_path,
        failures=failures,
    )


def _validate_guard_validation(
    *, json_path: Path, markdown_path: Path, failures: list[str]
) -> None:
    GuardValidationSmokeManifest.load(
        json_path=json_path,
        markdown_path=markdown_path,
        failures=failures,
    ).validate(failures)


def _validate_sbom(path: Path, failures: list[str]) -> None:
    payload = load_json(path, "SBOM", failures)
    if payload is not None and not isinstance(payload, dict):
        failures.append("SBOM must be a JSON object.")


def _validate_offline_bundle(offline_bundle_dir: Path, failures: list[str]) -> None:
    manifest = ReleaseEvidenceManifest(
        release_root=Path(),
        dist_root=Path(),
        sbom_path=Path(),
        guard_validation_json=Path(),
        guard_validation_markdown=Path(),
        offline_bundle_dir=offline_bundle_dir,
    )
    manifest._validate_offline_bundles(failures)


def check_release_evidence(
    *,
    release_root: Path,
    dist_root: Path,
    sbom_path: Path,
    guard_validation_json: Path,
    guard_validation_markdown: Path,
    offline_bundle_dir: Path,
) -> list[str]:
    manifest = ReleaseEvidenceManifest(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
    )
    return manifest.validate()


def _build_release_summary(
    *,
    release_root: Path,
    dist_root: Path,
    sbom_path: Path,
    guard_validation_json: Path,
    guard_validation_markdown: Path,
    offline_bundle_dir: Path,
    failures: list[str],
) -> dict[str, object]:
    manifest = ReleaseEvidenceManifest(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
    )
    return manifest.summary(failures)


def _resolve_artifact(
    root: Path, value: object, label: str, failures: list[str]
) -> Path | None:
    return resolve_artifact(root, value, label, failures)


def _validate_source_commands(payload: dict[str, object], failures: list[str]) -> None:
    manifest = EmpiricalGuardEvidenceManifest(root=Path(), payload=payload)
    manifest._validate_source_commands(failures)


def _validate_guard_rows(
    root: Path, payload: dict[str, object], failures: list[str]
) -> None:
    rows = payload.get("guard_rows")
    if not isinstance(rows, list) or not rows:
        failures.append("empirical evidence guard_rows must be a non-empty list.")
        return
    observed: set[str] = set()
    for index, row in enumerate(rows):
        label = f"guard_rows[{index}]"
        if not isinstance(row, dict):
            failures.append(f"{label} must be an object.")
            continue
        guard = GuardEvidenceRow(index=index, payload=row).validate(
            root=root,
            failures=failures,
        )
        if guard is not None:
            observed.add(guard)
    missing = sorted(REQUIRED_GUARDS - observed)
    if missing:
        failures.append("empirical evidence missing guard rows: " + ", ".join(missing))


def _validate_model_family_rows(
    root: Path, payload: dict[str, object], failures: list[str]
) -> None:
    rows = payload.get("model_family_rows")
    if not isinstance(rows, list) or not rows:
        failures.append(
            "empirical evidence model_family_rows must be a non-empty list."
        )
        return
    for index, row in enumerate(rows):
        label = f"model_family_rows[{index}]"
        if not isinstance(row, dict):
            failures.append(f"{label} must be an object.")
            continue
        ModelFamilyEvidenceRow(index=index, payload=row).validate(
            root=root,
            failures=failures,
        )


def check_empirical_guard_evidence(*, root: Path) -> list[str]:
    failures: list[str] = []
    manifest = EmpiricalGuardEvidenceManifest.load(root=root, failures=failures)
    if manifest.payload is None:
        if _MANIFEST_OBJECT_ERROR not in failures:
            failures.append(_MANIFEST_OBJECT_ERROR)
        return failures
    failures.extend(manifest.validate())
    return failures


def _build_empirical_summary(*, root: Path, failures: list[str]) -> dict[str, object]:
    manifest = EmpiricalGuardEvidenceManifest(root=root, payload={})
    return manifest.summary(failures)


def _add_release_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="artifacts/release")
    parser.add_argument("--dist", default="dist")
    parser.add_argument("--sbom", default="artifacts/supply-chain/sbom.json")
    parser.add_argument(
        "--guard-validation-json",
        default="artifacts/guard-validation/guard-validation-smoke.json",
    )
    parser.add_argument(
        "--guard-validation-md",
        default="artifacts/guard-validation/guard-validation-smoke.md",
    )
    parser.add_argument(
        "--offline-bundle-dir",
        default="artifacts/release/offline",
    )
    parser.add_argument("--json", action="store_true")


def _add_empirical_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        default="artifacts/guard-validation/empirical",
    )
    parser.add_argument("--json", action="store_true")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate InvarLock release and empirical evidence artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    release_parser = subparsers.add_parser(
        "release",
        help="Validate the local release-evidence bundle.",
    )
    _add_release_args(release_parser)
    empirical_parser = subparsers.add_parser(
        "empirical",
        help="Validate non-synthetic empirical guard-evidence artifacts.",
    )
    _add_empirical_args(empirical_parser)
    return parser.parse_args(argv)


def _run_release(args: argparse.Namespace) -> int:
    release_root = Path(args.root)
    dist_root = Path(args.dist)
    sbom_path = Path(args.sbom)
    guard_validation_json = Path(args.guard_validation_json)
    guard_validation_markdown = Path(args.guard_validation_md)
    offline_bundle_dir = Path(args.offline_bundle_dir)
    failures = check_release_evidence(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
    )
    summary = _build_release_summary(
        release_root=release_root,
        dist_root=dist_root,
        sbom_path=sbom_path,
        guard_validation_json=guard_validation_json,
        guard_validation_markdown=guard_validation_markdown,
        offline_bundle_dir=offline_bundle_dir,
        failures=failures,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
    else:
        print("Release evidence check passed.")
    return 1 if failures else 0


def _run_empirical(args: argparse.Namespace) -> int:
    root = Path(args.root)
    failures = check_empirical_guard_evidence(root=root)
    summary = _build_empirical_summary(root=root, failures=failures)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
    else:
        print("Empirical guard evidence check passed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "release":
        return _run_release(args)
    if args.command == "empirical":
        return _run_empirical(args)
    raise AssertionError(f"Unhandled evidence command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
