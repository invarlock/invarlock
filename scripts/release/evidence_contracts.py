"""Typed contracts for release and empirical evidence validation."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RELEASE_CHECK_SCHEMA = "invarlock/release-evidence-check-v1"
OFFLINE_BUNDLE_SCHEMA = "invarlock/release-offline-bundle-v1"
GUARD_VALIDATION_SMOKE_SCHEMA = "invarlock/guard-validation-smoke-v1"
EMPIRICAL_GUARD_EVIDENCE_SCHEMA = "invarlock/empirical-guard-evidence-v1"
EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA = "invarlock/empirical-guard-evidence-check-v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REQUIRED_GUARDS = frozenset({"spectral", "rmt", "variance"})
ALLOWED_EVIDENCE_KINDS = frozenset(
    {
        "calibration_null_sweep",
        "calibration_ve_sweep",
        "evidence_pack",
        "model_evidence_sweep",
        "checkpoint_probe",
    }
)
REAL_PRODUCER_MARKERS = (
    "model-evidence-sweep",
    "model_evidence_sweep.py",
    "calibrate null-sweep",
    "calibrate ve-sweep",
    "run_pack.sh",
    "run_suite.sh",
    "run_model_evidence_remote.py",
    "run_qwen14_sentinels.sh",
)


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


def resolve_artifact(
    root: Path, value: object, label: str, failures: list[str]
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        failures.append(f"{label} artifact must be a non-empty relative path.")
        return None
    raw = Path(value)
    if raw.is_absolute():
        failures.append(f"{label} artifact must be relative: {value}")
        return None
    root_resolved = root.resolve()
    candidate = (root_resolved / raw).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        failures.append(f"{label} artifact escapes evidence root: {value}")
        return None
    if not candidate.is_file():
        failures.append(f"{label} artifact missing: {candidate}")
        return None
    if candidate.stat().st_size == 0:
        failures.append(f"{label} artifact must not be empty: {candidate}")
    return candidate


@dataclass(frozen=True)
class GuardEvidenceRow:
    index: int
    payload: dict[str, Any]

    def validate(self, *, root: Path, failures: list[str]) -> str | None:
        label = f"guard_rows[{self.index}]"
        guard = self.payload.get("guard")
        if guard not in REQUIRED_GUARDS:
            failures.append(
                f"{label}.guard must be one of: {', '.join(sorted(REQUIRED_GUARDS))}."
            )
            return None
        kind = self.payload.get("evidence_kind")
        if kind not in ALLOWED_EVIDENCE_KINDS:
            failures.append(
                f"{label}.evidence_kind must be one of: "
                + ", ".join(sorted(ALLOWED_EVIDENCE_KINDS))
                + "."
            )
        status = self.payload.get("status")
        if status != "empirical":
            failures.append(f"{label}.status must be empirical.")
        scope = str(self.payload.get("scope", "")).lower()
        if self.payload.get("synthetic") is True or "synthetic" in scope:
            failures.append(f"{label} must not be synthetic evidence.")
        if not isinstance(
            self.payload.get("model_family"), str
        ) or not self.payload.get("model_family"):
            failures.append(f"{label}.model_family must be a non-empty string.")
        resolve_artifact(root, self.payload.get("artifact"), label, failures)
        if (
            guard in REQUIRED_GUARDS
            and kind in ALLOWED_EVIDENCE_KINDS
            and status == "empirical"
            and self.payload.get("synthetic") is not True
            and "synthetic" not in scope
        ):
            return str(guard)
        return None


@dataclass(frozen=True)
class ModelFamilyEvidenceRow:
    index: int
    payload: dict[str, Any]

    def validate(self, *, root: Path, failures: list[str]) -> None:
        label = f"model_family_rows[{self.index}]"
        if not isinstance(
            self.payload.get("model_family"), str
        ) or not self.payload.get("model_family"):
            failures.append(f"{label}.model_family must be a non-empty string.")
        if self.payload.get("status") not in {"observed", "empirical"}:
            failures.append(f"{label}.status must be observed or empirical.")
        resolve_artifact(root, self.payload.get("artifact"), label, failures)


@dataclass(frozen=True)
class EmpiricalGuardEvidenceManifest:
    root: Path
    payload: dict[str, Any] | None

    @classmethod
    def load(cls, *, root: Path, failures: list[str]) -> EmpiricalGuardEvidenceManifest:
        payload = load_json(
            root / "manifest.json", "empirical guard evidence manifest", failures
        )
        if payload is not None and not isinstance(payload, dict):
            failures.append("empirical guard evidence manifest must be a JSON object.")
            payload = None
        return cls(root, payload)

    def validate(self) -> list[str]:
        failures: list[str] = []
        if self.payload is None:
            load_failures: list[str] = []
            load_json(
                self.root / "manifest.json",
                "empirical guard evidence manifest",
                load_failures,
            )
            return load_failures + [
                "empirical guard evidence manifest must be a JSON object."
            ]
        if self.payload.get("schema") != EMPIRICAL_GUARD_EVIDENCE_SCHEMA:
            failures.append(
                f"empirical guard evidence schema must be {EMPIRICAL_GUARD_EVIDENCE_SCHEMA}."
            )
        self._validate_source_commands(failures)
        self._validate_guard_rows(failures)
        self._validate_model_family_rows(failures)
        return failures

    def _validate_source_commands(self, failures: list[str]) -> None:
        if self.payload is None:
            return
        commands = self.payload.get("source_commands")
        if not isinstance(commands, list) or not commands:
            failures.append(
                "empirical evidence source_commands must be a non-empty list."
            )
            return
        valid_commands: list[str] = []
        for index, command in enumerate(commands):
            if not isinstance(command, str) or not command.strip():
                failures.append(
                    f"empirical evidence source_commands[{index}] must be a string."
                )
                continue
            valid_commands.append(command)
        if valid_commands and not any(
            marker in command
            for command in valid_commands
            for marker in REAL_PRODUCER_MARKERS
        ):
            failures.append(
                "empirical evidence source_commands must include a real evidence producer."
            )

    def _validate_guard_rows(self, failures: list[str]) -> None:
        if self.payload is None:
            return
        rows = self.payload.get("guard_rows")
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
                root=self.root, failures=failures
            )
            if guard is not None:
                observed.add(guard)
        missing = sorted(REQUIRED_GUARDS - observed)
        if missing:
            failures.append(
                "empirical evidence missing guard rows: " + ", ".join(missing)
            )

    def _validate_model_family_rows(self, failures: list[str]) -> None:
        if self.payload is None:
            return
        rows = self.payload.get("model_family_rows")
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
                root=self.root,
                failures=failures,
            )

    def summary(self, failures: list[str]) -> dict[str, object]:
        return {
            "schema": EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA,
            "root": str(self.root),
            "ok": not failures,
            "failures": failures,
        }
