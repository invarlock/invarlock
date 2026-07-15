"""Validation for the current, immutable negative-evidence consumer index.

The repository may retain obsolete static examples for historical inspection.
This checker only accepts a current negative-evidence claim through the typed
pointer published by :mod:`scripts.model_evidence.negative_fixture_generation`.
It rehashes the immutable bundle and replays the strict verifier for each named
failure, so a self-authored pointer or receipt cannot turn an invalid fixture
into current evidence.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    read_regular_file_bytes,
    sha256_prefixed,
)
from invarlock.reporting.report_schema import validate_report
from invarlock.reporting.verify_contract import run_verify_reports
from invarlock.reporting.verify_contract_types import VerifyOutcome
from scripts.model_evidence.negative_fixture_generation import (
    ALL_SCENARIOS,
    CURRENT_NEGATIVE_EVIDENCE_STATUS,
    CURRENT_NEGATIVE_INDEX_SCHEMA,
    CURRENT_NEGATIVE_POINTER_FILENAME,
    EXPECTED_FAILURE_TEXT,
    GENUINE_SCENARIOS,
    INDEXED_ARTIFACTS,
    NEGATIVE_EVIDENCE_ONLY_RELEASE_STATUS,
    SIMULATED_SCENARIO,
    STRUCTURAL_FAILURE_TEXT,
    NegativeFixtureError,
    _require_release_guard_failure,
)

CURRENT_POINTER_FILENAME = CURRENT_NEGATIVE_POINTER_FILENAME
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_INVENTORIED_ARTIFACTS = {
    key: filename
    for key, filename in INDEXED_ARTIFACTS.items()
    if key != "hash_inventory"
}
_METADATA_ARTIFACT_PATHS = {
    key: filename for key, filename in INDEXED_ARTIFACTS.items() if key != "metadata"
}
_CURRENT_POINTER_FIELDS = frozenset(
    {
        "schema",
        "evidence_kind",
        "current_contract_status",
        "release_status",
        "strict_contract",
        "bundle",
        "bundle_sha256",
        "scenario_count",
        "scenarios",
    }
)
_SCENARIO_FIELDS = frozenset(
    {
        "scenario",
        "category",
        "path",
        "simulation",
        "expected_failure_text",
        "expected_runtime_image_digest",
        "artifacts",
    }
)
_PUBLISHED_BINDING_FIELDS = frozenset(
    {
        "report_sha256",
        "runtime_manifest_sha256",
        "baseline_report_sha256",
        "acceptance_policy_pack_sha256",
    }
)
_GENUINE_SOURCE_FIELDS = _PUBLISHED_BINDING_FIELDS | {"execution_receipt_sha256"}
_SIMULATED_SOURCE_FIELDS = _PUBLISHED_BINDING_FIELDS
_GENUINE_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "scenario",
        "simulation",
        "source",
        "published",
        "execution_receipt_role",
        "contract_checks",
        "release_verifier",
        "expected_runtime_image_digest",
        "authority",
    }
)
_SIMULATED_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "scenario",
        "simulation",
        "simulation_scope",
        "source",
        "published",
        "mutations",
        "contract_checks",
        "release_verifier",
        "expected_runtime_image_digest",
        "authority",
    }
)
_AUTHORITY_FIELDS = frozenset({"filename", "schema", "evidence_kind"})
_RELEASE_VERIFIER_FIELDS = frozenset(
    {
        "outcome",
        "expected_failure_text",
        "structural_contract",
        "diagnostics_sha256",
    }
)


def _label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    return sha256_prefixed(read_regular_file_bytes(path, label="negative fixture"))


def _bundle_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path).removeprefix("sha256:")))
    return digest.hexdigest()


def _load_object(errors: list[str], path: Path, root: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        errors.append(f"{_label(path, root)}: required regular JSON file is missing")
        return None
    try:
        _, payload = read_json_object_snapshot(path, label=_label(path, root))
    except StrictJsonError as exc:
        errors.append(f"{_label(path, root)}: invalid JSON: {exc}")
        return None
    return payload


def _safe_relative_path(
    errors: list[str],
    root: Path,
    raw_path: object,
    *,
    label: str,
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"{label}: path must be a non-empty relative string")
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{label}: path must stay within the public evidence root")
        return None
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        errors.append(f"{label}: path resolves outside the public evidence root")
        return None
    return candidate


def _expected_category(scenario: str) -> str:
    if scenario in GENUINE_SCENARIOS:
        return GENUINE_SCENARIOS[scenario][0]
    return "policy_failures"


def _expected_metadata_class(scenario: str) -> str:
    return (
        "caught_regression_fixture"
        if _expected_category(scenario) == "caught_regressions"
        else "policy_failure_fixture"
    )


def _expected_bundle_files() -> set[Path]:
    return {
        Path(_expected_category(scenario)) / scenario / filename
        for scenario in ALL_SCENARIOS
        for filename in INDEXED_ARTIFACTS.values()
    }


def _check_artifact_summaries(
    errors: list[str],
    *,
    entry: dict[str, Any],
    scenario_dir: Path,
    root: Path,
) -> dict[str, Path] | None:
    summaries = entry.get("artifacts")
    if not isinstance(summaries, dict) or set(summaries) != set(INDEXED_ARTIFACTS):
        errors.append(
            f"{_label(scenario_dir, root)}: current index artifacts must name "
            "the complete canonical artifact set"
        )
        return None
    paths: dict[str, Path] = {}
    for key, filename in INDEXED_ARTIFACTS.items():
        summary = summaries.get(key)
        path = scenario_dir / filename
        if not isinstance(summary, dict) or set(summary) != {"sha256", "size_bytes"}:
            errors.append(
                f"{_label(scenario_dir, root)}: index artifact {key} has invalid summary"
            )
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(
                f"{_label(path, root)}: indexed artifact must be a regular file"
            )
            continue
        try:
            artifact_bytes = read_regular_file_bytes(
                path, label=f"indexed negative fixture artifact {filename}"
            )
        except StrictJsonError as exc:
            errors.append(f"{_label(path, root)}: indexed artifact is unsafe: {exc}")
            continue
        if summary.get("size_bytes") != len(artifact_bytes):
            errors.append(f"{_label(path, root)}: indexed size does not match")
        if summary.get("sha256") != sha256_prefixed(artifact_bytes):
            errors.append(f"{_label(path, root)}: indexed sha256 does not match")
        paths[key] = path
    return paths if len(paths) == len(INDEXED_ARTIFACTS) else None


def _check_hash_inventory(
    errors: list[str],
    *,
    inventory_path: Path,
    artifact_paths: dict[str, Path],
    root: Path,
) -> None:
    inventory = _load_object(errors, inventory_path, root)
    if inventory is None:
        return
    if set(inventory) != {"schema", "artifacts"}:
        errors.append(f"{_label(inventory_path, root)}: inventory shape is invalid")
    if inventory.get("schema") != "invarlock.negative_fixture.hash_inventory.v1":
        errors.append(f"{_label(inventory_path, root)}: unsupported inventory schema")
    entries = inventory.get("artifacts")
    if not isinstance(entries, list):
        errors.append(f"{_label(inventory_path, root)}: artifacts must be a list")
        return
    inventory_by_path = {
        item.get("path"): item for item in entries if isinstance(item, dict)
    }
    expected_paths = set(_INVENTORIED_ARTIFACTS.values())
    if set(inventory_by_path) != expected_paths or len(entries) != len(expected_paths):
        errors.append(
            f"{_label(inventory_path, root)}: inventory must cover exactly the "
            "canonical non-self artifact set"
        )
    for key, filename in _INVENTORIED_ARTIFACTS.items():
        entry = inventory_by_path.get(filename)
        path = artifact_paths[key]
        if not isinstance(entry, dict):
            continue
        if set(entry) != {"path", "bytes", "sha256"}:
            errors.append(
                f"{_label(inventory_path, root)}: {filename} entry shape is invalid"
            )
        artifact_bytes = read_regular_file_bytes(
            path, label=f"inventoried negative fixture artifact {filename}"
        )
        if entry.get("bytes") != len(artifact_bytes):
            errors.append(
                f"{_label(inventory_path, root)}: {filename} byte count mismatch"
            )
        if entry.get("sha256") != sha256_prefixed(artifact_bytes):
            errors.append(f"{_label(inventory_path, root)}: {filename} sha256 mismatch")


def _check_metadata(
    errors: list[str],
    *,
    metadata_path: Path,
    scenario: str,
    root: Path,
) -> None:
    metadata = _load_object(errors, metadata_path, root)
    if metadata is None:
        return
    if metadata.get("schema") != "invarlock.public_evidence.meta.v1":
        errors.append(f"{_label(metadata_path, root)}: unsupported metadata schema")
    if metadata.get("evidence_class") != _expected_metadata_class(scenario):
        errors.append(
            f"{_label(metadata_path, root)}: evidence_class does not match {scenario}"
        )
    artifact_paths = metadata.get("artifact_paths")
    if artifact_paths != _METADATA_ARTIFACT_PATHS:
        errors.append(
            f"{_label(metadata_path, root)}: metadata artifact_paths must be canonical"
        )
    commands = metadata.get("verifier_commands")
    if not isinstance(commands, list) or not all(
        isinstance(command, str) and command.strip() for command in commands
    ):
        errors.append(
            f"{_label(metadata_path, root)}: verifier_commands must be non-empty strings"
        )


def _check_generation_receipt(
    errors: list[str],
    *,
    receipt_path: Path,
    artifact_paths: dict[str, Path],
    scenario: str,
    simulation: bool,
    expected_digest: str,
    root: Path,
) -> None:
    receipt = _load_object(errors, receipt_path, root)
    if receipt is None:
        return
    expected_receipt_fields = (
        _SIMULATED_RECEIPT_FIELDS if simulation else _GENUINE_RECEIPT_FIELDS
    )
    if set(receipt) != expected_receipt_fields:
        errors.append(
            f"{_label(receipt_path, root)}: generation receipt must have "
            "the exact v1 shape"
        )
    expected = {
        "schema": "invarlock.negative_fixture.generation_receipt.v1",
        "scenario": scenario,
        "simulation": simulation,
        "expected_runtime_image_digest": expected_digest,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            errors.append(
                f"{_label(receipt_path, root)}: {field} does not bind {scenario}"
            )
    authority = receipt.get("authority")
    if not isinstance(authority, dict) or set(authority) != _AUTHORITY_FIELDS:
        errors.append(
            f"{_label(receipt_path, root)}: authority binding must have "
            "the exact v1 shape"
        )
    elif authority != {
        "filename": CURRENT_POINTER_FILENAME,
        "schema": CURRENT_NEGATIVE_INDEX_SCHEMA,
        "evidence_kind": "negative_fixture",
    }:
        errors.append(
            f"{_label(receipt_path, root)}: authority binding does not match "
            "the current pointer"
        )
    source = receipt.get("source")
    if not isinstance(source, dict):
        errors.append(f"{_label(receipt_path, root)}: source binding must be an object")
    else:
        expected_source_fields = (
            _SIMULATED_SOURCE_FIELDS if simulation else _GENUINE_SOURCE_FIELDS
        )
        if set(source) != expected_source_fields:
            errors.append(
                f"{_label(receipt_path, root)}: source binding must have "
                "the exact v1 shape"
            )
        expected_source = {
            "report_sha256": _sha256_file(artifact_paths["evaluation_report"]),
            "baseline_report_sha256": _sha256_file(artifact_paths["baseline_report"]),
            "acceptance_policy_pack_sha256": _sha256_file(
                artifact_paths["acceptance_policy_pack"]
            ),
        }
        for field, value in expected_source.items():
            if source.get(field) != value:
                errors.append(
                    f"{_label(receipt_path, root)}: source.{field} does not bind "
                    "the published artifact"
                )
        source_manifest = source.get("runtime_manifest_sha256")
        if (
            not isinstance(source_manifest, str)
            or SHA256_RE.fullmatch(source_manifest) is None
        ):
            errors.append(
                f"{_label(receipt_path, root)}: source runtime manifest digest is invalid"
            )
        execution_receipt = source.get("execution_receipt_sha256")
        if simulation:
            if execution_receipt is not None:
                errors.append(
                    f"{_label(receipt_path, root)}: simulated scenario must not "
                    "claim an execution receipt"
                )
        elif (
            not isinstance(execution_receipt, str)
            or SHA256_RE.fullmatch(execution_receipt) is None
        ):
            errors.append(
                f"{_label(receipt_path, root)}: genuine scenario must bind a "
                "producer execution receipt digest"
            )
    published = receipt.get("published")
    if not isinstance(published, dict):
        errors.append(
            f"{_label(receipt_path, root)}: published binding must be an object"
        )
    else:
        if set(published) != _PUBLISHED_BINDING_FIELDS:
            errors.append(
                f"{_label(receipt_path, root)}: published binding must have "
                "the exact v1 shape"
            )
        expected_published = {
            "report_sha256": _sha256_file(artifact_paths["evaluation_report"]),
            "runtime_manifest_sha256": _sha256_file(artifact_paths["runtime_manifest"]),
            "baseline_report_sha256": _sha256_file(artifact_paths["baseline_report"]),
            "acceptance_policy_pack_sha256": _sha256_file(
                artifact_paths["acceptance_policy_pack"]
            ),
        }
        for field, value in expected_published.items():
            if published.get(field) != value:
                errors.append(
                    f"{_label(receipt_path, root)}: published.{field} does not bind "
                    "the public artifact"
                )
    release_verifier = receipt.get("release_verifier")
    if not isinstance(release_verifier, dict):
        errors.append(
            f"{_label(receipt_path, root)}: release_verifier must be an object"
        )
        return
    if set(release_verifier) != _RELEASE_VERIFIER_FIELDS:
        errors.append(
            f"{_label(receipt_path, root)}: release_verifier must have "
            "the exact v1 shape"
        )
    if release_verifier.get("outcome") != str(VerifyOutcome.POLICY_FAIL):
        errors.append(
            f"{_label(receipt_path, root)}: recorded verifier outcome is invalid"
        )
    if release_verifier.get("expected_failure_text") != EXPECTED_FAILURE_TEXT[scenario]:
        errors.append(f"{_label(receipt_path, root)}: recorded failure text is invalid")
    if release_verifier.get("structural_contract") != "clean":
        errors.append(
            f"{_label(receipt_path, root)}: recorded verifier did not clear "
            "structural contracts"
        )
    diagnostics_digest = release_verifier.get("diagnostics_sha256")
    if (
        not isinstance(diagnostics_digest, str)
        or SHA256_RE.fullmatch(diagnostics_digest) is None
    ):
        errors.append(
            f"{_label(receipt_path, root)}: recorded diagnostics digest is invalid"
        )


def _check_runtime_binding(
    errors: list[str],
    *,
    report_path: Path,
    manifest_path: Path,
    root: Path,
) -> None:
    manifest = _load_object(errors, manifest_path, root)
    if manifest is None:
        return
    report = manifest.get("report")
    if not isinstance(report, dict):
        errors.append(
            f"{_label(manifest_path, root)}: report binding must be an object"
        )
        return
    if report.get("filename") != report_path.name:
        errors.append(
            f"{_label(manifest_path, root)}: report filename binding mismatch"
        )
    if report.get("sha256") != _sha256_file(report_path).removeprefix("sha256:"):
        errors.append(f"{_label(manifest_path, root)}: report sha256 binding mismatch")
    if report.get("path") != report_path.name:
        errors.append(f"{_label(manifest_path, root)}: report path binding mismatch")


def _check_scenario_shape(
    errors: list[str],
    *,
    report: dict[str, Any],
    manifest_path: Path,
    scenario: str,
    root: Path,
) -> None:
    if scenario in GENUINE_SCENARIOS:
        try:
            _require_release_guard_failure(scenario, report)
        except NegativeFixtureError as exc:
            errors.append(
                f"{_label(manifest_path, root)}: current negative scenario shape "
                f"is invalid: {exc}"
            )
        return
    validation = report.get("validation")
    if not isinstance(validation, dict) or any(
        validation.get(field) is not True
        for field in (
            "primary_metric_acceptable",
            "invariants_pass",
            "spectral_stable",
            "rmt_stable",
        )
    ):
        errors.append(
            f"{_label(manifest_path, root)}: runtime simulation source report must "
            "pass all non-runtime release predicates"
        )
    manifest = _load_object(errors, manifest_path, root)
    if manifest is None:
        return
    runtime = manifest.get("runtime")
    if (
        manifest.get("execution_mode") != "host-bypass"
        or not isinstance(runtime, dict)
        or runtime.get("container_execution") is not False
        or runtime.get("image_digest") is not None
    ):
        errors.append(
            f"{_label(manifest_path, root)}: runtime simulation must be limited "
            "to the declared provenance mutation"
        )


def _check_strict_replay(
    errors: list[str],
    *,
    artifact_paths: dict[str, Path],
    scenario: str,
    expected_digest: str,
    root: Path,
) -> None:
    report_path = artifact_paths["evaluation_report"]
    report = _load_object(errors, report_path, root)
    if report is None:
        return
    if not validate_report(report):
        errors.append(
            f"{_label(report_path, root)}: current negative report is not valid "
            "under the current schema"
        )
        return
    _check_scenario_shape(
        errors,
        report=report,
        manifest_path=artifact_paths["runtime_manifest"],
        scenario=scenario,
        root=root,
    )
    try:
        result = run_verify_reports(
            [report_path],
            baseline=artifact_paths["baseline_report"],
            policy_pack=artifact_paths["acceptance_policy_pack"],
            profile="release",
            assurance_mode="strict",
            expected_runtime_image_digest=expected_digest,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary for audit CLI
        errors.append(
            f"{_label(report_path, root)}: strict negative replay raised "
            f"{type(exc).__name__}: {exc}"
        )
        return
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    if result.outcome != VerifyOutcome.POLICY_FAIL:
        errors.append(
            f"{_label(report_path, root)}: strict negative replay must be policy_fail"
        )
    if EXPECTED_FAILURE_TEXT[scenario] not in diagnostics:
        errors.append(
            f"{_label(report_path, root)}: strict negative replay did not reproduce "
            f"{scenario}"
        )
    diagnostics_lower = diagnostics.lower()
    structural_failure = next(
        (marker for marker in STRUCTURAL_FAILURE_TEXT if marker in diagnostics_lower),
        None,
    )
    if structural_failure is not None:
        errors.append(
            f"{_label(report_path, root)}: strict negative replay hit unrelated "
            f"structural failure {structural_failure!r}"
        )
    if (
        scenario != SIMULATED_SCENARIO
        and "report/manifest binding" in diagnostics_lower
    ):
        errors.append(
            f"{_label(report_path, root)}: strict negative replay hit unrelated "
            "report/manifest binding failure"
        )


def check_current_negative_fixture_index(errors: list[str], root: Path) -> bool:
    """Validate the optional current negative-evidence index.

    Returns ``True`` only when a complete current index exists and every
    scenario independently replays as the named strict policy failure.  A
    missing index is valid for ordinary historical audits but false for callers
    that require release-closure negative evidence.
    """

    pointer_path = root / CURRENT_POINTER_FILENAME
    if not pointer_path.exists():
        return False
    initial_errors = len(errors)
    pointer = _load_object(errors, pointer_path, root)
    if pointer is None:
        return False
    if set(pointer) != _CURRENT_POINTER_FIELDS:
        errors.append(
            f"{_label(pointer_path, root)}: current index must have the exact v1 shape"
        )
    if pointer.get("schema") != CURRENT_NEGATIVE_INDEX_SCHEMA:
        errors.append(
            f"{_label(pointer_path, root)}: current index requires "
            f"{CURRENT_NEGATIVE_INDEX_SCHEMA}"
        )
    if pointer.get("evidence_kind") != "negative_fixture":
        errors.append(
            f"{_label(pointer_path, root)}: evidence_kind must be negative_fixture"
        )
    if pointer.get("current_contract_status") != CURRENT_NEGATIVE_EVIDENCE_STATUS:
        errors.append(
            f"{_label(pointer_path, root)}: current_contract_status is invalid"
        )
    if pointer.get("release_status") != NEGATIVE_EVIDENCE_ONLY_RELEASE_STATUS:
        errors.append(f"{_label(pointer_path, root)}: release_status is invalid")
    if pointer.get("strict_contract") != {
        "profile": "release",
        "assurance_mode": "strict",
        "expected_outcome": "policy_fail",
    }:
        errors.append(f"{_label(pointer_path, root)}: strict_contract is invalid")

    bundle_path = _safe_relative_path(
        errors,
        root,
        pointer.get("bundle"),
        label=f"{_label(pointer_path, root)}.bundle",
    )
    bundle_digest = pointer.get("bundle_sha256")
    if not isinstance(bundle_digest, str) or SHA256_RE.fullmatch(bundle_digest) is None:
        errors.append(f"{_label(pointer_path, root)}: bundle_sha256 is invalid")
    if bundle_path is None:
        return False
    relative_bundle = bundle_path.relative_to(root)
    if (
        len(relative_bundle.parts) != 2
        or relative_bundle.parts[0] != "negative_fixture_bundles"
        or HEX_DIGEST_RE.fullmatch(relative_bundle.parts[1]) is None
    ):
        errors.append(
            f"{_label(pointer_path, root)}: bundle must name one immutable bundle"
        )
    if not bundle_path.is_dir() or bundle_path.is_symlink():
        errors.append(f"{_label(bundle_path, root)}: immutable bundle is missing")
        return False
    symlinks = [path for path in bundle_path.rglob("*") if path.is_symlink()]
    if symlinks:
        errors.append(
            f"{_label(bundle_path, root)}: immutable bundle must not contain symlinks"
        )
        return False
    actual_files = {
        path.relative_to(bundle_path)
        for path in bundle_path.rglob("*")
        if path.is_file()
    }
    if actual_files != _expected_bundle_files():
        errors.append(
            f"{_label(bundle_path, root)}: immutable bundle file set is not canonical"
        )
    actual_bundle_digest = "sha256:" + _bundle_digest(bundle_path)
    if bundle_digest != actual_bundle_digest:
        errors.append(f"{_label(pointer_path, root)}: bundle_sha256 does not match")
    if relative_bundle.parts[-1] != actual_bundle_digest.removeprefix("sha256:"):
        errors.append(
            f"{_label(pointer_path, root)}: bundle path digest does not match"
        )

    entries = pointer.get("scenarios")
    if (
        not isinstance(entries, list)
        or pointer.get("scenario_count") != len(ALL_SCENARIOS)
        or len(entries) != len(ALL_SCENARIOS)
    ):
        errors.append(
            f"{_label(pointer_path, root)}: current index must contain all six scenarios"
        )
        return False
    seen: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append(
                f"{_label(pointer_path, root)}: scenario entry must be an object"
            )
            continue
        if set(entry) != _SCENARIO_FIELDS:
            errors.append(
                f"{_label(pointer_path, root)}: scenario entry must have "
                "the exact v1 shape"
            )
        scenario = entry.get("scenario")
        if not isinstance(scenario, str) or scenario not in ALL_SCENARIOS:
            errors.append(f"{_label(pointer_path, root)}: scenario name is invalid")
            continue
        seen.append(scenario)
        category = _expected_category(scenario)
        simulation = scenario == SIMULATED_SCENARIO
        expected_path = (relative_bundle / category / scenario).as_posix()
        if entry.get("category") != category:
            errors.append(
                f"{_label(pointer_path, root)}: {scenario} category is invalid"
            )
        if entry.get("path") != expected_path:
            errors.append(f"{_label(pointer_path, root)}: {scenario} path is invalid")
        if entry.get("simulation") is not simulation:
            errors.append(
                f"{_label(pointer_path, root)}: {scenario} simulation flag is invalid"
            )
        if entry.get("expected_failure_text") != EXPECTED_FAILURE_TEXT[scenario]:
            errors.append(
                f"{_label(pointer_path, root)}: {scenario} expected failure text is invalid"
            )
        expected_digest = entry.get("expected_runtime_image_digest")
        if (
            not isinstance(expected_digest, str)
            or SHA256_RE.fullmatch(expected_digest) is None
        ):
            errors.append(
                f"{_label(pointer_path, root)}: {scenario} runtime image digest is invalid"
            )
            continue
        scenario_dir = bundle_path / category / scenario
        artifact_paths = _check_artifact_summaries(
            errors,
            entry=entry,
            scenario_dir=scenario_dir,
            root=root,
        )
        if artifact_paths is None:
            continue
        _check_hash_inventory(
            errors,
            inventory_path=artifact_paths["hash_inventory"],
            artifact_paths=artifact_paths,
            root=root,
        )
        _check_metadata(
            errors,
            metadata_path=artifact_paths["metadata"],
            scenario=scenario,
            root=root,
        )
        _check_generation_receipt(
            errors,
            receipt_path=artifact_paths["generation_receipt"],
            artifact_paths=artifact_paths,
            scenario=scenario,
            simulation=simulation,
            expected_digest=expected_digest,
            root=root,
        )
        _check_runtime_binding(
            errors,
            report_path=artifact_paths["evaluation_report"],
            manifest_path=artifact_paths["runtime_manifest"],
            root=root,
        )
        _check_strict_replay(
            errors,
            artifact_paths=artifact_paths,
            scenario=scenario,
            expected_digest=expected_digest,
            root=root,
        )
    if seen != list(ALL_SCENARIOS):
        errors.append(
            f"{_label(pointer_path, root)}: scenarios must appear once in canonical order"
        )
    return len(errors) == initial_errors


__all__ = [
    "CURRENT_POINTER_FILENAME",
    "check_current_negative_fixture_index",
]
