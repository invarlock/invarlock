"""Typed contracts for release shape and diagnostic evidence inventory."""

from __future__ import annotations

import json
import math
import random
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from scripts.release.evidence_contract_cli import run_cli
except ImportError:  # pragma: no cover - direct script execution path
    from evidence_contract_cli import run_cli

from invarlock.guards.policies import (
    get_rmt_policy,
    get_spectral_policy,
    get_variance_policy,
)
from invarlock.guards.rmt_policy import compute_epsilon_violations
from invarlock.guards.spectral_detection import summarize_family_z_scores
from invarlock.guards.variance_policy import predictive_gate_outcome

try:
    from scripts.release.evidence_contracts_empirical import (
        _MANIFEST_OBJECT_ERROR,
        ALLOWED_EVIDENCE_KINDS,
        EMPIRICAL_GUARD_EVIDENCE_AUTHORITY,
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
        EMPIRICAL_GUARD_EVIDENCE_AUTHORITY,
        EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA,
        EMPIRICAL_GUARD_EVIDENCE_SCHEMA,
        REAL_PRODUCER_MARKERS,
        REQUIRED_GUARDS,
        EmpiricalGuardEvidenceManifest,
        GuardEvidenceRow,
        ModelFamilyEvidenceRow,
        resolve_artifact,
    )

# This checker only validates the presence and shape of local supporting files.
# It deliberately cannot authorize a release: use release_preflight.py for the
# fail-closed checkout, distribution, strict-closure, and public-evidence gate.
RELEASE_CHECK_SCHEMA = "invarlock/release-artifact-shape-check-v1"
OFFLINE_BUNDLE_SCHEMA = "invarlock/release-offline-bundle-v1"
GUARD_VALIDATION_SMOKE_SCHEMA = "invarlock/guard-validation-smoke-v1"
GUARD_VALIDATION_SCOPE = (
    "deterministic synthetic production-primitive smoke; not empirical "
    "model-family proof or threshold calibration"
)
GUARD_VALIDATION_PRODUCTION_PRIMITIVES = {
    "spectral": {
        "entrypoint": ("invarlock.guards.spectral_detection.summarize_family_z_scores"),
        "role": "violation_summary",
    },
    "rmt": {
        "entrypoint": "invarlock.guards.rmt_policy.compute_epsilon_violations",
        "role": "violation_detection",
    },
    "variance": {
        "entrypoint": "invarlock.guards.variance_policy.predictive_gate_outcome",
        "role": "gate_outcome",
    },
}
GUARD_VALIDATION_WINDOWS = (16, 32, 64, 128)
GUARD_VALIDATION_MAX_REPLICATES = 10_000
GUARD_VALIDATION_SOURCE_FILES = {
    "producer": "scripts/smoke/guard_validation_smoke.py",
    "policy": "src/invarlock/guards/policies.py",
    "spectral": "src/invarlock/guards/spectral_detection.py",
    "rmt": "src/invarlock/guards/rmt_policy.py",
    "variance": "src/invarlock/guards/variance_policy.py",
}
GUARD_VALIDATION_TOP_FIELDS = frozenset(
    {
        "evidence_sha256",
        "markdown_sha256",
        "production_primitives",
        "rate_rows",
        "replicates",
        "schema",
        "scope",
        "seed",
        "source_identity",
    }
)
GUARD_VALIDATION_ROW_FIELDS = frozenset(
    {
        "calibration_windows",
        "derived_seed",
        "guard",
        "null_outcomes",
        "null_trigger_count",
        "null_trigger_rate",
        "primitive_role",
        "production_entrypoint",
        "shifted_outcomes",
        "shifted_trigger_count",
        "shifted_trigger_rate",
        "threshold",
    }
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

try:
    from scripts.release.release_evidence_core_contracts import (
        DistHashManifest,
        StrictReportEvidence,
        StrictVerifyEvidence,
        _finite_number,
        _read_regular_snapshot,
        _sha256_bytes,
        _strict_json_snapshot,
        existing_globs,
        load_json,
        require_any,
        require_file,
        sha256,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from release_evidence_core_contracts import (
        DistHashManifest,
        StrictReportEvidence,
        StrictVerifyEvidence,
        _finite_number,
        _read_regular_snapshot,
        _sha256_bytes,
        _strict_json_snapshot,
        existing_globs,
        load_json,
        require_any,
        require_file,
        sha256,
    )
RUNTIME_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

__all__ = [
    "ALLOWED_EVIDENCE_KINDS",
    "EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA",
    "EMPIRICAL_GUARD_EVIDENCE_AUTHORITY",
    "EMPIRICAL_GUARD_EVIDENCE_SCHEMA",
    "EmpiricalGuardEvidenceManifest",
    "GuardEvidenceRow",
    "ModelFamilyEvidenceRow",
    "REAL_PRODUCER_MARKERS",
    "REQUIRED_GUARDS",
]


def _canonical_guard_evidence_digest(payload: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"evidence_sha256", "markdown_sha256"}
    }
    encoded = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _guard_thresholds() -> dict[str, float]:
    return {
        "spectral": float(
            get_spectral_policy("balanced")["family_caps"]["ffn"]["kappa"]
        ),
        "rmt": float(get_rmt_policy("balanced")["epsilon_by_family"]["ffn"]),
        "variance": float(get_variance_policy("aggressive")["min_effect_lognll"]),
    }


def _guard_distribution(guard: str) -> tuple[float, float, float]:
    return {
        "spectral": (0.0, 1.0, 4.5),
        "rmt": (0.002, 0.002, 0.02),
        "variance": (0.005, 0.005, 0.05),
    }[guard]


def _guard_triggers(guard: str, score: float, threshold: float) -> bool:
    if guard == "spectral":
        summary = summarize_family_z_scores(
            {"synthetic.ffn": score},
            {"synthetic.ffn": "ffn"},
            {"ffn": {"kappa": threshold}},
        )
        return int(summary["ffn"]["violations"]) > 0
    if guard == "rmt":
        state = SimpleNamespace(
            baseline_edge_risk_by_family={"ffn": 1.0},
            edge_risk_by_family={"ffn": 1.0 + score},
            epsilon_by_family={"ffn": threshold},
            epsilon_default=threshold,
        )
        return bool(compute_epsilon_violations(state))
    if guard == "variance":
        mean_delta = -score
        passed, _ = predictive_gate_outcome(
            mean_delta,
            (mean_delta - 0.001, mean_delta),
            threshold,
            one_sided=True,
        )
        return passed
    raise ValueError(f"unknown guard: {guard}")


def _replay_guard_outcomes(
    *,
    guard: str,
    windows: int,
    replicates: int,
    seed: int,
    threshold: float,
) -> tuple[list[bool], list[bool]]:
    null_mean, null_sd, defect_shift = _guard_distribution(guard)
    rng = random.Random(seed)
    scale = math.sqrt(32.0) / math.sqrt(float(windows))
    sample_sd = null_sd * scale
    null_outcomes: list[bool] = []
    shifted_outcomes: list[bool] = []
    for _ in range(replicates):
        null_score = sum(
            rng.gauss(null_mean, sample_sd) for _ in range(windows)
        ) / float(windows)
        shifted_score = sum(
            rng.gauss(null_mean + defect_shift, sample_sd) for _ in range(windows)
        ) / float(windows)
        null_outcomes.append(_guard_triggers(guard, null_score, threshold))
        shifted_outcomes.append(_guard_triggers(guard, shifted_score, threshold))
    return null_outcomes, shifted_outcomes


def _render_guard_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Guard Validation Smoke",
        "",
        "This generated artifact is a deterministic synthetic smoke, not a",
        "replacement for empirical guard calibration on real checkpoints.",
        "",
        f"Evidence digest: `{payload['evidence_sha256']}`",
        "",
        "| Guard | Windows | Synthetic Null Trigger | Synthetic Shifted Trigger |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["rate_rows"]:
        lines.append(
            "| {guard} | {windows} | {null_rate:.3f} | {shifted_rate:.3f} |".format(
                guard=row["guard"],
                windows=row["calibration_windows"],
                null_rate=row["null_trigger_rate"],
                shifted_rate=row["shifted_trigger_rate"],
            )
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class GuardValidationSmokeManifest:
    json_path: Path
    markdown_path: Path
    payload: dict[str, Any] | None
    markdown_bytes: bytes | None

    @classmethod
    def load(
        cls, *, json_path: Path, markdown_path: Path, failures: list[str]
    ) -> GuardValidationSmokeManifest:
        payload = _strict_json_snapshot(
            json_path,
            label="guard-validation JSON",
            failures=failures,
        )
        if payload is not None and not isinstance(payload, dict):
            failures.append("guard-validation JSON must be a JSON object.")
            payload = None
        markdown_bytes = _read_regular_snapshot(
            markdown_path,
            label="guard-validation markdown",
            failures=failures,
        )
        return cls(json_path, markdown_path, payload, markdown_bytes)

    def validate(self, failures: list[str]) -> None:
        if self.payload is None:
            return
        payload = self.payload
        actual_fields = frozenset(payload)
        if actual_fields != GUARD_VALIDATION_TOP_FIELDS:
            failures.append(
                "guard-validation JSON top-level fields must match v1 exactly."
            )
            return
        if payload["schema"] != GUARD_VALIDATION_SMOKE_SCHEMA:
            failures.append("guard-validation JSON schema is not recognized.")
            return
        if payload["scope"] != GUARD_VALIDATION_SCOPE:
            failures.append("guard-validation JSON scope is not recognized.")
        seed = payload["seed"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not -(2**63) <= seed < 2**63
        ):
            failures.append("guard-validation seed must be a signed 64-bit integer.")
            return
        replicates = payload["replicates"]
        if (
            isinstance(replicates, bool)
            or not isinstance(replicates, int)
            or not 1 <= replicates <= GUARD_VALIDATION_MAX_REPLICATES
        ):
            failures.append(
                "guard-validation replicates must be an integer in [1, 10000]."
            )
            return
        if payload["production_primitives"] != GUARD_VALIDATION_PRODUCTION_PRIMITIVES:
            failures.append(
                "guard-validation JSON must name the production primitive "
                "and role for every required guard."
            )

        self._validate_source_identity(failures)
        rows = payload["rate_rows"]
        if not isinstance(rows, list) or len(rows) != 12:
            failures.append("guard-validation rate_rows must contain exactly 12 rows.")
            return
        thresholds = _guard_thresholds()
        expected_pairs = [
            (guard, windows)
            for guard in ("spectral", "rmt", "variance")
            for windows in GUARD_VALIDATION_WINDOWS
        ]
        row_failure_count = len(failures)
        for index, (row, (guard, windows)) in enumerate(
            zip(rows, expected_pairs, strict=True)
        ):
            self._validate_row(
                index=index,
                row=row,
                guard=guard,
                windows=windows,
                seed=seed,
                replicates=replicates,
                threshold=thresholds[guard],
                failures=failures,
            )

        rows_valid = len(failures) == row_failure_count
        expected_evidence_digest = _canonical_guard_evidence_digest(payload)
        if payload["evidence_sha256"] != expected_evidence_digest:
            failures.append(
                "guard-validation evidence_sha256 does not match JSON evidence."
            )
        if self.markdown_bytes is not None:
            if _sha256_bytes(self.markdown_bytes) != payload["markdown_sha256"]:
                failures.append(
                    "guard-validation markdown bytes do not match markdown_sha256."
                )
        if rows_valid:
            expected_markdown = _render_guard_markdown(payload).encode("utf-8")
            if payload["markdown_sha256"] != _sha256_bytes(expected_markdown):
                failures.append(
                    "guard-validation markdown_sha256 does not match canonical markdown."
                )
            if (
                self.markdown_bytes is not None
                and self.markdown_bytes != expected_markdown
            ):
                failures.append(
                    "guard-validation markdown does not render the bound JSON evidence."
                )

    def _validate_source_identity(self, failures: list[str]) -> None:
        identity = self.payload["source_identity"] if self.payload is not None else None
        if not isinstance(identity, dict) or frozenset(identity) != {
            "policy",
            "primitives",
            "producer",
        }:
            failures.append("guard-validation source_identity shape is invalid.")
            return
        primitives = identity["primitives"]
        if not isinstance(primitives, dict) or frozenset(primitives) != REQUIRED_GUARDS:
            failures.append("guard-validation primitive source identity is incomplete.")
            return
        repo_root = Path(__file__).resolve().parents[2]
        identities = {
            "producer": identity["producer"],
            "policy": identity["policy"],
            **{guard: primitives[guard] for guard in REQUIRED_GUARDS},
        }
        for name, item in identities.items():
            expected_path = GUARD_VALIDATION_SOURCE_FILES[name]
            if not isinstance(item, dict) or frozenset(item) != {"path", "sha256"}:
                failures.append(
                    f"guard-validation {name} source identity shape is invalid."
                )
                continue
            if item["path"] != expected_path:
                failures.append(
                    f"guard-validation {name} source path is not canonical."
                )
                continue
            source_failures: list[str] = []
            raw = _read_regular_snapshot(
                repo_root / expected_path,
                label=f"guard-validation {name} source",
                failures=source_failures,
            )
            failures.extend(source_failures)
            if raw is not None and item["sha256"] != _sha256_bytes(raw):
                failures.append(
                    f"guard-validation {name} source digest does not match."
                )

    @staticmethod
    def _validate_row(
        *,
        index: int,
        row: Any,
        guard: str,
        windows: int,
        seed: int,
        replicates: int,
        threshold: float,
        failures: list[str],
    ) -> None:
        prefix = f"guard-validation rate_rows[{index}]"
        if not isinstance(row, dict):
            failures.append(f"{prefix} must be an object.")
            return
        if frozenset(row) != GUARD_VALIDATION_ROW_FIELDS:
            failures.append(f"{prefix} fields must match v1 exactly.")
            return
        if (
            not isinstance(row["guard"], str)
            or isinstance(row["calibration_windows"], bool)
            or not isinstance(row["calibration_windows"], int)
            or isinstance(row["derived_seed"], bool)
            or not isinstance(row["derived_seed"], int)
            or _finite_number(row["threshold"]) is None
            or not isinstance(row["production_entrypoint"], str)
            or not isinstance(row["primitive_role"], str)
            or isinstance(row["null_trigger_count"], bool)
            or not isinstance(row["null_trigger_count"], int)
            or isinstance(row["shifted_trigger_count"], bool)
            or not isinstance(row["shifted_trigger_count"], int)
            or _finite_number(row["null_trigger_rate"]) is None
            or _finite_number(row["shifted_trigger_rate"]) is None
        ):
            failures.append(f"{prefix} scalar field types are invalid.")
            return
        expected_primitive = GUARD_VALIDATION_PRODUCTION_PRIMITIVES[guard]
        expected_seed = seed + (index // 4) * 1000 + (index % 4)
        if (
            row["guard"] != guard
            or row["calibration_windows"] != windows
            or row["derived_seed"] != expected_seed
            or row["threshold"] != threshold
            or row["production_entrypoint"] != expected_primitive["entrypoint"]
            or row["primitive_role"] != expected_primitive["role"]
        ):
            failures.append(f"{prefix} identity or production binding does not match.")
            return
        null_outcomes = row["null_outcomes"]
        shifted_outcomes = row["shifted_outcomes"]
        if (
            not isinstance(null_outcomes, list)
            or not isinstance(shifted_outcomes, list)
            or len(null_outcomes) != replicates
            or len(shifted_outcomes) != replicates
            or any(not isinstance(value, bool) for value in null_outcomes)
            or any(not isinstance(value, bool) for value in shifted_outcomes)
        ):
            failures.append(f"{prefix} raw outcomes must be boolean replicate arrays.")
            return
        expected_null, expected_shifted = _replay_guard_outcomes(
            guard=guard,
            windows=windows,
            replicates=replicates,
            seed=expected_seed,
            threshold=threshold,
        )
        if null_outcomes != expected_null or shifted_outcomes != expected_shifted:
            failures.append(f"{prefix} raw outcomes do not match deterministic replay.")
        null_count = sum(null_outcomes)
        shifted_count = sum(shifted_outcomes)
        if row["null_trigger_count"] != null_count:
            failures.append(f"{prefix} null_trigger_count does not match outcomes.")
        if row["shifted_trigger_count"] != shifted_count:
            failures.append(f"{prefix} shifted_trigger_count does not match outcomes.")
        if row["null_trigger_rate"] != null_count / float(replicates):
            failures.append(f"{prefix} null_trigger_rate does not match outcomes.")
        if row["shifted_trigger_rate"] != shifted_count / float(replicates):
            failures.append(f"{prefix} shifted_trigger_rate does not match outcomes.")


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
            "check_scope": "local artifact shape only",
            "authoritative_release_approval": False,
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


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
