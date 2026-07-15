#!/usr/bin/env python3
"""Build reproducible public fixtures from evaluated negative scenarios.

Five scenarios must come from an executed model edit or fault-injection run.  The
runtime-provenance scenario is the sole simulation: it deterministically mutates
an otherwise passing report's runtime manifest and records exactly what changed.
Execution receipts are hash-bound producer provenance, not cryptographic proof
that the recorded command ran.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invarlock.evidence_pack_json import (  # noqa: E402
    StrictJsonError,
    parse_json_bytes,
    read_json_object_snapshot,
    read_regular_file_bytes,
    sha256_prefixed,
)
from invarlock.reporting.report_schema import validate_report  # noqa: E402
from invarlock.reporting.verify_contract import run_verify_reports  # noqa: E402
from invarlock.reporting.verify_contract_types import VerifyOutcome  # noqa: E402
from invarlock.strict_yaml import (  # noqa: E402
    StrictYamlError,
    parse_yaml_documents_bytes,
)

GENUINE_SCENARIOS = {
    "spectral_guard_failure": ("caught_regressions", "model_edit"),
    "rmt_guard_failure": ("caught_regressions", "model_edit"),
    "variance_guard_failure": ("caught_regressions", "model_edit"),
    "invariants_failure": ("policy_failures", "fault_injection"),
    "primary_metric_failure": ("policy_failures", "model_edit"),
}
SIMULATED_SCENARIO = "runtime_provenance_failure"
ALL_SCENARIOS = (*GENUINE_SCENARIOS, SIMULATED_SCENARIO)
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PRIVATE_TEXT_PATTERNS = (
    re.compile(r"(?:^|\s)(?:root|[A-Za-z0-9._-]+)@[A-Za-z0-9._-]+"),
    re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"),
    re.compile(r"/(?:Users|home|root|private|tmp|var/folders)/"),
)
EXPECTED_FAILURE_TEXT = {
    "spectral_guard_failure": "validation.spectral_stable == true",
    "rmt_guard_failure": "validation.rmt_stable == true",
    "variance_guard_failure": "variance.predictive_gate.passed is false",
    "invariants_failure": "validation.invariants_pass == true",
    "primary_metric_failure": "Primary metric policy gate failed",
    "runtime_provenance_failure": "host-bypass",
}
CURRENT_NEGATIVE_INDEX_SCHEMA = "invarlock.negative_fixture.current.v1"
CURRENT_NEGATIVE_POINTER_FILENAME = "negative_fixtures.current.json"
CURRENT_NEGATIVE_EVIDENCE_STATUS = "current_strict_negative_evidence"
NEGATIVE_EVIDENCE_ONLY_RELEASE_STATUS = "negative_evidence_only_not_release_ready"
INDEXED_ARTIFACTS = {
    "evaluation_report": "evaluation.report.json",
    "runtime_manifest": "runtime.manifest.json",
    "baseline_report": "baseline.report.json",
    "acceptance_policy_pack": "acceptance_policy_pack.json",
    "generation_receipt": "generation.receipt.json",
    "hash_inventory": "hash_inventory.json",
    "metadata": "evidence.meta.json",
}
STRUCTURAL_FAILURE_TEXT = (
    "schema validation failed",
    "requires report assurance.mode=strict",
    "strict assurance report missing",
    "requires runtime-bound policy provenance",
    "guard chain evidence",
    "stage must be",
    "is required for strict assurance",
    "requires plugins.",
    "runtime policy receipt",
    "strict baseline provenance",
    "strict baseline profile mismatch",
    "must be a signed 64-bit json integer",
    "bootstrap replay requires",
    "strict ppl evidence requires",
    "provider-digest-missing",
    "invalid policy-pack-v1",
)


class NegativeFixtureError(ValueError):
    """Raised when an input cannot substantiate its named negative fixture."""


class _PointerCommittedError(RuntimeError):
    """The pointer switched, but its parent-directory fsync did not complete."""


def _structured_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key, item in value.items():
            values.extend(_structured_string_values(key))
            values.extend(_structured_string_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_structured_string_values(item))
        return values
    return []


def _decoded_structured_strings(path: Path, payload: bytes) -> list[str]:
    try:
        if path.suffix == ".json":
            documents = [
                parse_json_bytes(payload, label=f"publication JSON {path.name}")
            ]
        elif path.suffix == ".jsonl":
            lines = payload.splitlines()
            if not lines or any(not line.strip() for line in lines):
                raise NegativeFixtureError(
                    f"publication JSONL is empty or ambiguous: {path}"
                )
            documents = [
                parse_json_bytes(line, label=f"publication JSONL {path.name}")
                for line in lines
            ]
        elif path.suffix in {".yaml", ".yml"}:
            documents = parse_yaml_documents_bytes(
                payload, label=f"publication YAML {path.name}"
            )
        else:
            return []
    except (StrictJsonError, StrictYamlError) as exc:
        raise NegativeFixtureError(
            f"publication structured text is invalid: {path}"
        ) from exc
    values: list[str] = []
    for document in documents:
        values.extend(_structured_string_values(document))
    return values


def _validate_publication_tree(path: Path, *, private_roots: tuple[Path, ...]) -> None:
    if not path.is_dir() or path.is_symlink():
        raise NegativeFixtureError(f"publication bundle is missing or unsafe: {path}")
    private_tokens = tuple(str(item.resolve(strict=False)) for item in private_roots)
    for item in path.rglob("*"):
        if item.is_symlink():
            raise NegativeFixtureError(f"publication bundle contains a symlink: {item}")
        if not item.is_file():
            continue
        try:
            payload = read_regular_file_bytes(item, label="publication file")
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        except StrictJsonError as exc:
            raise NegativeFixtureError(f"publication file is unsafe: {item}") from exc
        candidates = [text, *_decoded_structured_strings(item, payload)]
        if any(
            token and token in candidate
            for candidate in candidates
            for token in private_tokens
        ) or any(
            pattern.search(candidate)
            for candidate in candidates
            for pattern in _PRIVATE_TEXT_PATTERNS
        ):
            raise NegativeFixtureError(
                f"publication bundle contains private path or host text: {item}"
            )


def _load_object_snapshot(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        return read_json_object_snapshot(path, label=label)
    except StrictJsonError as exc:
        raise NegativeFixtureError(f"cannot load {label}: {path}") from exc


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    return _load_object_snapshot(path, label=label)[1]


def _canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_canonical_bytes(payload))


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return sha256_prefixed(read_regular_file_bytes(path, label="fixture artifact"))
    except StrictJsonError as exc:
        raise NegativeFixtureError(f"cannot hash fixture artifact: {path}") from exc


def _validation(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("validation")
    if not isinstance(value, dict):
        raise NegativeFixtureError("source report lacks a validation object")
    return value


def _require_bool(value: Any, expected: bool, *, field: str) -> None:
    if value is not expected:
        raise NegativeFixtureError(f"{field} must be {str(expected).lower()}")


def _require_pm_pass(report: dict[str, Any]) -> None:
    _require_bool(
        _validation(report).get("primary_metric_acceptable"),
        True,
        field="validation.primary_metric_acceptable",
    )


def _require_release_guard_failure(
    scenario: str, report: dict[str, Any]
) -> dict[str, Any]:
    """Return the explicit contract checks proven by an evaluated report."""

    validation = _validation(report)
    checks: dict[str, Any] = {"primary_metric_acceptable": True}
    if scenario == "spectral_guard_failure":
        _require_pm_pass(report)
        _require_bool(
            validation.get("spectral_stable"),
            False,
            field="validation.spectral_stable",
        )
        spectral = report.get("spectral")
        if not isinstance(spectral, dict) or not (
            spectral.get("caps_exceeded") is True or bool(spectral.get("violations"))
        ):
            raise NegativeFixtureError(
                "spectral failure needs measured caps_exceeded or violations"
            )
        checks["release_guard_failure"] = "spectral_stable"
    elif scenario == "rmt_guard_failure":
        _require_pm_pass(report)
        _require_bool(
            validation.get("rmt_stable"), False, field="validation.rmt_stable"
        )
        rmt = report.get("rmt")
        if not isinstance(rmt, dict) or not (
            rmt.get("stable") is False or bool(rmt.get("epsilon_violations"))
        ):
            raise NegativeFixtureError(
                "RMT failure needs stable=false or epsilon_violations"
            )
        checks["release_guard_failure"] = "rmt_stable"
    elif scenario == "variance_guard_failure":
        _require_pm_pass(report)
        variance = report.get("variance")
        predictive = (
            variance.get("predictive_gate") if isinstance(variance, dict) else None
        )
        if (
            not isinstance(variance, dict)
            or variance.get("passed") is not False
            or not isinstance(predictive, dict)
            or predictive.get("evaluated") is not True
            or predictive.get("passed") is not False
        ):
            raise NegativeFixtureError(
                "variance failure needs an evaluated, failed predictive gate"
            )
        checks["release_guard_failure"] = "variance_predictive_gate"
    elif scenario == "invariants_failure":
        _require_bool(
            validation.get("invariants_pass"),
            False,
            field="validation.invariants_pass",
        )
        invariants = report.get("invariants")
        if not isinstance(invariants, dict) or not (
            invariants.get("failures") or invariants.get("violations")
        ):
            raise NegativeFixtureError(
                "invariants failure needs retained failure observations"
            )
        checks = {"policy_failure": "invariants_pass"}
    elif scenario == "primary_metric_failure":
        _require_bool(
            validation.get("primary_metric_acceptable"),
            False,
            field="validation.primary_metric_acceptable",
        )
        checks = {"policy_failure": "primary_metric_acceptable"}
    else:  # pragma: no cover - guarded by caller
        raise NegativeFixtureError(f"unknown genuine scenario: {scenario}")
    return checks


def _require_source_binding(
    report_path: Path, manifest_path: Path, manifest: dict[str, Any]
) -> None:
    binding = manifest.get("report")
    if not isinstance(binding, dict):
        raise NegativeFixtureError("source runtime manifest lacks report binding")
    expected = _sha256_file(report_path).removeprefix("sha256:")
    if binding.get("sha256") != expected:
        raise NegativeFixtureError("source runtime manifest report digest mismatch")
    if binding.get("filename") != report_path.name:
        raise NegativeFixtureError("source runtime manifest report filename mismatch")
    if manifest_path.parent != report_path.parent:
        raise NegativeFixtureError(
            "source report and runtime manifest must be sidecars"
        )


def _require_execution_receipt(
    scenario: str,
    receipt_path: Path,
    receipt: dict[str, Any],
    *,
    report_path: Path,
    manifest_path: Path,
    expected_kind: str,
) -> None:
    expected = {
        "schema": "invarlock.negative_fixture.execution_receipt.v1",
        "scenario": scenario,
        "execution_kind": expected_kind,
        "simulation": False,
        "report_sha256": _sha256_file(report_path),
        "runtime_manifest_sha256": _sha256_file(manifest_path),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise NegativeFixtureError(
                f"execution receipt {field} does not bind the source execution"
            )
    command = receipt.get("command")
    if not isinstance(command, str) or not command.strip():
        raise NegativeFixtureError(
            "execution receipt must retain the evaluated command"
        )
    if receipt_path.parent != report_path.parent:
        raise NegativeFixtureError("execution receipt must be a source sidecar")


def _public_manifest(
    manifest: dict[str, Any], *, report_bytes: bytes, relative_report: str
) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    result["report"] = {
        "filename": "evaluation.report.json",
        "path": relative_report,
        "sha256": _sha256_bytes(report_bytes).removeprefix("sha256:"),
    }
    return result


def _metadata(
    scenario: str, *, simulated: bool, expected_runtime_image_digest: str
) -> dict[str, Any]:
    caught = scenario in GENUINE_SCENARIOS and GENUINE_SCENARIOS[scenario][0] == (
        "caught_regressions"
    )
    label = scenario.removesuffix("_failure").replace("_", " ")
    if caught:
        summary = (
            f"Caught-regression fixture from an evaluated {label} failure where "
            "the primary metric passes."
        )
        evidence_class = "caught_regression_fixture"
    elif simulated:
        summary = (
            "Policy-failure fixture produced by a deterministic runtime-provenance "
            "manifest mutation; it is not a runtime execution claim."
        )
        evidence_class = "policy_failure_fixture"
    else:
        summary = f"Policy-failure fixture from an evaluated {label} fault."
        evidence_class = "policy_failure_fixture"
    return {
        "schema": "invarlock.public_evidence.meta.v1",
        "evidence_class": evidence_class,
        "summary": summary,
        "generated_by": "scripts/model_evidence/negative_fixture_generation.py",
        "artifact_paths": {
            "evaluation_report": "evaluation.report.json",
            "runtime_manifest": "runtime.manifest.json",
            "baseline_report": "baseline.report.json",
            "acceptance_policy_pack": "acceptance_policy_pack.json",
            "generation_receipt": "generation.receipt.json",
            "hash_inventory": "hash_inventory.json",
        },
        "non_goals": [
            "This fixture does not estimate production failure rates.",
            "This fixture does not establish universal guard efficacy.",
            "Producer execution receipts are not cryptographic attestations.",
        ],
        "verifier_commands": [
            "invarlock verify evaluation.report.json "
            "--baseline baseline.report.json "
            "--policy-pack acceptance_policy_pack.json "
            f"--expected-runtime-image-digest {expected_runtime_image_digest} "
            "--profile release --assurance strict"
        ],
    }


def _inventory(directory: Path) -> dict[str, Any]:
    artifacts = []
    for name in (
        "evaluation.report.json",
        "runtime.manifest.json",
        "baseline.report.json",
        "acceptance_policy_pack.json",
        "generation.receipt.json",
        "evidence.meta.json",
    ):
        path = directory / name
        artifact_bytes = read_regular_file_bytes(
            path, label=f"negative fixture inventory artifact {name}"
        )
        artifacts.append(
            {
                "path": name,
                "bytes": len(artifact_bytes),
                "sha256": sha256_prefixed(artifact_bytes),
            }
        )
    return {
        "schema": "invarlock.negative_fixture.hash_inventory.v1",
        "artifacts": artifacts,
    }


def _verification_inputs(
    entry: dict[str, Any], *, report_path: Path
) -> tuple[Path, Path, str, bytes, bytes]:
    baseline_path = Path(str(entry.get("baseline_report", ""))).resolve(strict=True)
    policy_path = Path(str(entry.get("policy_pack", ""))).resolve(strict=True)
    digest = entry.get("expected_runtime_image_digest")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise NegativeFixtureError(
            "expected_runtime_image_digest must be an explicit sha256 digest"
        )
    if baseline_path == report_path:
        raise NegativeFixtureError("negative report cannot be its own trusted baseline")
    baseline_bytes, _ = _load_object_snapshot(
        baseline_path, label="trusted baseline report"
    )
    policy_bytes, _ = _load_object_snapshot(policy_path, label="acceptance policy pack")
    return baseline_path, policy_path, digest, baseline_bytes, policy_bytes


def _require_unchanged_snapshot(path: Path, expected: bytes, *, label: str) -> None:
    try:
        observed = read_regular_file_bytes(path, label=label)
    except StrictJsonError as exc:
        raise NegativeFixtureError(f"{label} became unsafe during generation") from exc
    if observed != expected:
        raise NegativeFixtureError(f"{label} changed during generation")


def _require_verifier_failure(
    scenario: str,
    *,
    report_path: Path,
    baseline_path: Path,
    policy_path: Path,
    expected_runtime_image_digest: str,
) -> dict[str, Any]:
    result = run_verify_reports(
        [report_path],
        baseline=baseline_path,
        policy_pack=policy_path,
        profile="release",
        assurance_mode="strict",
        expected_runtime_image_digest=expected_runtime_image_digest,
    )
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    expected = EXPECTED_FAILURE_TEXT[scenario]
    if result.outcome != VerifyOutcome.POLICY_FAIL or expected not in diagnostics:
        raise NegativeFixtureError(
            f"release verifier did not reproduce {scenario}: expected {expected!r}; "
            f"diagnostics={diagnostics!r}"
        )
    diagnostics_lower = diagnostics.lower()
    structural_failure = next(
        (marker for marker in STRUCTURAL_FAILURE_TEXT if marker in diagnostics_lower),
        None,
    )
    if structural_failure is not None:
        raise NegativeFixtureError(
            f"release verifier found unrelated structural failure for {scenario}: "
            f"{structural_failure}"
        )
    if (
        scenario != SIMULATED_SCENARIO
        and "report/manifest binding" in diagnostics_lower
    ):
        raise NegativeFixtureError(
            f"release verifier found unrelated provenance failure for {scenario}"
        )
    return {
        "outcome": str(result.outcome),
        "expected_failure_text": expected,
        "structural_contract": "clean",
        "diagnostics_sha256": _sha256_bytes(diagnostics.encode()),
    }


def _stage_genuine(scenario: str, entry: dict[str, Any], stage: Path) -> str:
    _, expected_kind = GENUINE_SCENARIOS[scenario]
    report_path = Path(str(entry.get("report", ""))).resolve(strict=True)
    manifest_path = Path(str(entry.get("runtime_manifest", ""))).resolve(strict=True)
    receipt_path = Path(str(entry.get("execution_receipt", ""))).resolve(strict=True)
    report_bytes, report = _load_object_snapshot(report_path, label="source report")
    manifest_bytes, manifest = _load_object_snapshot(
        manifest_path, label="source runtime manifest"
    )
    receipt_bytes, receipt = _load_object_snapshot(
        receipt_path, label="source execution receipt"
    )
    if not validate_report(report):
        raise NegativeFixtureError(f"source report is not schema-valid: {report_path}")
    checks = _require_release_guard_failure(scenario, report)
    _require_source_binding(report_path, manifest_path, manifest)
    (
        baseline_path,
        policy_path,
        expected_digest,
        baseline_bytes,
        policy_bytes,
    ) = _verification_inputs(entry, report_path=report_path)
    _require_execution_receipt(
        scenario,
        receipt_path,
        receipt,
        report_path=report_path,
        manifest_path=manifest_path,
        expected_kind=expected_kind,
    )
    verifier = _require_verifier_failure(
        scenario,
        report_path=report_path,
        baseline_path=baseline_path,
        policy_path=policy_path,
        expected_runtime_image_digest=expected_digest,
    )
    for path, expected_bytes, label in (
        (report_path, report_bytes, "source report"),
        (manifest_path, manifest_bytes, "source runtime manifest"),
        (receipt_path, receipt_bytes, "source execution receipt"),
        (baseline_path, baseline_bytes, "trusted baseline report"),
        (policy_path, policy_bytes, "acceptance policy pack"),
    ):
        _require_unchanged_snapshot(path, expected_bytes, label=label)
    relative = "evaluation.report.json"
    (stage / "evaluation.report.json").write_bytes(report_bytes)
    (stage / "baseline.report.json").write_bytes(baseline_bytes)
    (stage / "acceptance_policy_pack.json").write_bytes(policy_bytes)
    _write_json(
        stage / "runtime.manifest.json",
        _public_manifest(manifest, report_bytes=report_bytes, relative_report=relative),
    )
    _write_json(
        stage / "generation.receipt.json",
        {
            "schema": "invarlock.negative_fixture.generation_receipt.v1",
            "scenario": scenario,
            "simulation": False,
            "source": {
                "execution_receipt_sha256": sha256_prefixed(receipt_bytes),
                "report_sha256": sha256_prefixed(report_bytes),
                "runtime_manifest_sha256": sha256_prefixed(manifest_bytes),
                "baseline_report_sha256": sha256_prefixed(baseline_bytes),
                "acceptance_policy_pack_sha256": sha256_prefixed(policy_bytes),
            },
            "published": {
                "report_sha256": _sha256_file(stage / "evaluation.report.json"),
                "runtime_manifest_sha256": _sha256_file(
                    stage / "runtime.manifest.json"
                ),
                "baseline_report_sha256": _sha256_file(stage / "baseline.report.json"),
                "acceptance_policy_pack_sha256": _sha256_file(
                    stage / "acceptance_policy_pack.json"
                ),
            },
            "execution_receipt_role": (
                "Hash-bound producer provenance; not cryptographic execution "
                "attestation."
            ),
            "contract_checks": checks,
            "release_verifier": verifier,
            "expected_runtime_image_digest": expected_digest,
            "authority": {
                "filename": CURRENT_NEGATIVE_POINTER_FILENAME,
                "schema": CURRENT_NEGATIVE_INDEX_SCHEMA,
                "evidence_kind": "negative_fixture",
            },
        },
    )
    return expected_digest


def _stage_runtime_simulation(entry: dict[str, Any], stage: Path) -> str:
    report_path = Path(str(entry.get("report", ""))).resolve(strict=True)
    manifest_path = Path(str(entry.get("runtime_manifest", ""))).resolve(strict=True)
    report_bytes, report = _load_object_snapshot(report_path, label="source report")
    manifest_bytes, manifest = _load_object_snapshot(
        manifest_path, label="source runtime manifest"
    )
    if not validate_report(report):
        raise NegativeFixtureError(f"source report is not schema-valid: {report_path}")
    _require_source_binding(report_path, manifest_path, manifest)
    (
        baseline_path,
        policy_path,
        expected_digest,
        baseline_bytes,
        policy_bytes,
    ) = _verification_inputs(entry, report_path=report_path)
    validation = _validation(report)
    for field in (
        "primary_metric_acceptable",
        "invariants_pass",
        "spectral_stable",
        "rmt_stable",
    ):
        _require_bool(validation.get(field), True, field=f"validation.{field}")
    relative = "evaluation.report.json"
    mutated = _public_manifest(
        manifest, report_bytes=report_bytes, relative_report=relative
    )
    mutated["execution_mode"] = "host-bypass"
    runtime = mutated.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        mutated["runtime"] = runtime
    runtime.update(
        {
            "container_execution": False,
            "image_digest": None,
            "image_ref": "host-bypass-public-fixture",
        }
    )
    (stage / "evaluation.report.json").write_bytes(report_bytes)
    (stage / "baseline.report.json").write_bytes(baseline_bytes)
    (stage / "acceptance_policy_pack.json").write_bytes(policy_bytes)
    _write_json(stage / "runtime.manifest.json", mutated)
    verifier = _require_verifier_failure(
        SIMULATED_SCENARIO,
        report_path=stage / "evaluation.report.json",
        baseline_path=stage / "baseline.report.json",
        policy_path=stage / "acceptance_policy_pack.json",
        expected_runtime_image_digest=expected_digest,
    )
    _write_json(
        stage / "generation.receipt.json",
        {
            "schema": "invarlock.negative_fixture.generation_receipt.v1",
            "scenario": SIMULATED_SCENARIO,
            "simulation": True,
            "simulation_scope": "runtime manifest provenance fields only",
            "source": {
                "report_sha256": sha256_prefixed(report_bytes),
                "runtime_manifest_sha256": sha256_prefixed(manifest_bytes),
                "baseline_report_sha256": sha256_prefixed(baseline_bytes),
                "acceptance_policy_pack_sha256": sha256_prefixed(policy_bytes),
            },
            "published": {
                "report_sha256": _sha256_file(stage / "evaluation.report.json"),
                "runtime_manifest_sha256": _sha256_file(
                    stage / "runtime.manifest.json"
                ),
                "baseline_report_sha256": _sha256_file(stage / "baseline.report.json"),
                "acceptance_policy_pack_sha256": _sha256_file(
                    stage / "acceptance_policy_pack.json"
                ),
            },
            "mutations": [
                {"field": "execution_mode", "value": "host-bypass"},
                {"field": "runtime.container_execution", "value": False},
                {"field": "runtime.image_digest", "value": None},
                {
                    "field": "runtime.image_ref",
                    "value": "host-bypass-public-fixture",
                },
            ],
            "contract_checks": {
                "source_release_gates_pass": True,
                "policy_failure": "runtime_provenance",
            },
            "release_verifier": verifier,
            "expected_runtime_image_digest": expected_digest,
            "authority": {
                "filename": CURRENT_NEGATIVE_POINTER_FILENAME,
                "schema": CURRENT_NEGATIVE_INDEX_SCHEMA,
                "evidence_kind": "negative_fixture",
            },
        },
    )
    return expected_digest


def _bundle_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path).removeprefix("sha256:")))
    return digest.hexdigest()


def _current_negative_index(
    bundle: Path,
    *,
    output_root: Path,
    bundle_digest: str,
) -> dict[str, Any]:
    """Build the atomic, typed public consumer index for one immutable bundle."""

    bundle_relative = bundle.relative_to(output_root).as_posix()
    scenarios: list[dict[str, Any]] = []
    for scenario in ALL_SCENARIOS:
        category = (
            GENUINE_SCENARIOS[scenario][0]
            if scenario in GENUINE_SCENARIOS
            else "policy_failures"
        )
        scenario_dir = bundle / category / scenario
        receipt = _load_object(
            scenario_dir / "generation.receipt.json",
            label=f"generated {scenario} receipt",
        )
        digest = receipt.get("expected_runtime_image_digest")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise NegativeFixtureError(
                f"generated {scenario} receipt lacks a runtime image digest"
            )
        artifacts: dict[str, dict[str, Any]] = {}
        for key, filename in INDEXED_ARTIFACTS.items():
            artifact = scenario_dir / filename
            if not artifact.is_file() or artifact.is_symlink():
                raise NegativeFixtureError(
                    f"generated {scenario} artifact missing or not regular: {filename}"
                )
            artifact_bytes = read_regular_file_bytes(
                artifact, label=f"generated {scenario} artifact {filename}"
            )
            artifacts[key] = {
                "sha256": sha256_prefixed(artifact_bytes),
                "size_bytes": len(artifact_bytes),
            }
        scenarios.append(
            {
                "scenario": scenario,
                "category": category,
                "path": (Path(bundle_relative) / category / scenario).as_posix(),
                "simulation": scenario == SIMULATED_SCENARIO,
                "expected_failure_text": EXPECTED_FAILURE_TEXT[scenario],
                "expected_runtime_image_digest": digest,
                "artifacts": artifacts,
            }
        )
    return {
        "schema": CURRENT_NEGATIVE_INDEX_SCHEMA,
        "evidence_kind": "negative_fixture",
        "current_contract_status": CURRENT_NEGATIVE_EVIDENCE_STATUS,
        "release_status": NEGATIVE_EVIDENCE_ONLY_RELEASE_STATUS,
        "strict_contract": {
            "profile": "release",
            "assurance_mode": "strict",
            "expected_outcome": "policy_fail",
        },
        "bundle": bundle_relative,
        "bundle_sha256": f"sha256:{bundle_digest}",
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fsync_tree(root: Path) -> None:
    """Make every staged file and directory durable before publication."""

    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        _fsync_directory(directory)


def _commit_immutable_bundle(stage: Path, output_root: Path) -> tuple[Path, str, bool]:
    """Durably commit bytes without creating or changing an authority pointer."""

    bundle_digest = _bundle_digest(stage)
    bundle_parent = output_root / "negative_fixture_bundles"
    bundle_parent.mkdir(parents=True, exist_ok=True)
    bundle = bundle_parent / bundle_digest
    created_bundle = False
    _fsync_tree(stage)
    if bundle.exists():
        if _bundle_digest(bundle) != bundle_digest:
            raise NegativeFixtureError("existing immutable negative bundle is corrupt")
        _fsync_tree(bundle)
        shutil.rmtree(stage)
    else:
        os.replace(stage, bundle)
        created_bundle = True
        _fsync_directory(bundle_parent)
    return bundle, bundle_digest, created_bundle


def _atomic_replace_json(
    pointer_path: Path, payload: object, *, temporary_prefix: str
) -> None:
    """Durably replace one JSON pointer; the caller defines its authority semantics."""

    pointer_tmp = pointer_path.parent / f".{temporary_prefix}-{uuid.uuid4().hex}.tmp"
    switched = False
    try:
        _write_json(pointer_tmp, payload)
        with pointer_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(pointer_tmp, pointer_path)
        switched = True
        _fsync_directory(pointer_path.parent)
    except BaseException as exc:
        if switched:
            raise _PointerCommittedError(
                "JSON pointer switched but directory durability could not be confirmed"
            ) from exc
        raise
    finally:
        pointer_tmp.unlink(missing_ok=True)


def _replace_output_bundle(stage: Path, output_root: Path) -> Path:
    """Publish one validated bundle through the sole rich v1 authority schema."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".negative-fixtures.publish.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        bundle, bundle_digest, created_bundle = _commit_immutable_bundle(
            stage, output_root
        )
        pointer_path = output_root / CURRENT_NEGATIVE_POINTER_FILENAME
        try:
            pointer = _current_negative_index(
                bundle,
                output_root=output_root,
                bundle_digest=bundle_digest,
            )
            _atomic_replace_json(
                pointer_path,
                pointer,
                temporary_prefix="negative-fixtures-pointer",
            )
        except _PointerCommittedError:
            raise
        except BaseException:
            if created_bundle and bundle.exists():
                shutil.rmtree(bundle)
                _fsync_directory(bundle.parent)
            raise
        return bundle


def generate(spec_path: Path, output_root: Path) -> None:
    spec = _load_object(spec_path.resolve(strict=True), label="generation spec")
    if spec.get("schema") != "invarlock.negative_fixture.generation_spec.v1":
        raise NegativeFixtureError("unsupported generation spec schema")
    scenarios = spec.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != set(ALL_SCENARIOS):
        raise NegativeFixtureError(
            "generation spec must define exactly the six negative scenarios"
        )
    output_root = output_root.resolve()
    build_stage = output_root / f".negative-fixtures-stage-{uuid.uuid4().hex}"
    build_stage.mkdir(parents=True)
    try:
        for scenario in ALL_SCENARIOS:
            entry = scenarios[scenario]
            if not isinstance(entry, dict):
                raise NegativeFixtureError(
                    f"scenario entry must be an object: {scenario}"
                )
            category = (
                GENUINE_SCENARIOS[scenario][0]
                if scenario in GENUINE_SCENARIOS
                else "policy_failures"
            )
            stage = build_stage / category / scenario
            stage.mkdir(parents=True)
            if scenario in GENUINE_SCENARIOS:
                expected_digest = _stage_genuine(scenario, entry, stage)
                simulated = False
            else:
                expected_digest = _stage_runtime_simulation(entry, stage)
                simulated = True
            _write_json(
                stage / "evidence.meta.json",
                _metadata(
                    scenario,
                    simulated=simulated,
                    expected_runtime_image_digest=expected_digest,
                ),
            )
            _write_json(stage / "hash_inventory.json", _inventory(stage))
            _validate_publication_tree(
                stage,
                private_roots=(
                    Path(str(entry["report"])).resolve().parent,
                    REPO_ROOT,
                ),
            )
        _replace_output_bundle(build_stage, output_root)
    finally:
        if build_stage.exists():
            shutil.rmtree(build_stage)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    generate(args.spec, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
