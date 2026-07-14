from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MATRIX = Path("examples/integrations/source_matrix.json")
SOURCE_MATRIX_SCHEMA = "invarlock.integration_source_matrix.v1"
SOURCE_MATRIX_TOP_LEVEL_FIELDS = frozenset({"schema", "description", "entries"})
SOURCE_MATRIX_TARGETS = frozenset(
    {
        "awq",
        "fine_tune",
        "gptqmodel",
        "hf_bnb",
        "hqq",
        "magnitude_prune",
        "peft_lora",
        "quanto",
        "torchao_int8_runtime",
    }
)
SOURCE_MATRIX_ENTRY_FIELDS = frozenset(
    {
        "command_shape",
        "expected",
        "lane",
        "provenance_artifacts",
        "readme",
        "report_path",
        "required_artifacts",
        "runner",
        "runner_enforcement",
        "runtime_image",
        "status_label",
        "strict_claim_phrase",
        "subject_adapter",
        "subject_form",
        "target",
        "verification_profile",
    }
)
SOURCE_MATRIX_TRAINING_FIELDS = frozenset({"training_profile", "training_scope"})
SOURCE_MATRIX_TRAINING_TARGETS = frozenset({"fine_tune", "peft_lora"})
SOURCE_MATRIX_RUNTIME_IMAGE_FIELDS = frozenset(
    {
        "declared_digest_source",
        "expected_digest_source",
        "family",
        "source_command",
    }
)
SOURCE_MATRIX_EXPECTED_FIELDS = frozenset(
    {
        "lane_artifact_label",
        "runtime_expected_digest_matched",
        "runtime_provenance_declared",
        "runtime_provenance_status",
        "runtime_provenance_verified",
        "verify_status",
    }
)
SOURCE_MATRIX_RUNNER_ENFORCEMENT_FIELDS = frozenset(
    {"backend_inventory", "runtime_quantization_proof"}
)
BACKEND_INVENTORY_SCHEMA = "invarlock/backend-inventory-v1"
RUNTIME_QUANTIZATION_PROOF_SCHEMA = "invarlock/runtime-quantization-proof-v1"
RUNTIME_QUANTIZATION_PROOF_KIND = "live_loaded_model_runtime_type_inventory"
RUNTIME_QUANTIZATION_PROOF_FILENAME = "runtime_quantization_proof.json"

# These adapters have a narrowly defined, observable runtime representation.
# A strict lane for one of them is not meaningful unless the producer recorded
# that representation after loading the subject.  Packed storage formats are
# deliberately excluded: a module inventory cannot establish their on-disk
# transformation semantics.
MODULE_BACKED_QUANTIZED_ADAPTER_BACKENDS = {
    "hf_bnb": "bitsandbytes",
    "hf_awq": "gptqmodel",
    "hf_gptq": "gptqmodel",
    "hf_torchao": "torchao",
    "hf_hqq": "hqq",
    "hf_quanto": "optimum-quanto",
}
STRICT_UNSUPPORTED_QUANTIZED_ADAPTERS = frozenset({"hf_ct"})

_BITSANDBYTES_RUNTIME_TYPES = frozenset(
    {
        "bitsandbytes.nn.modules.linear4bit",
        "bitsandbytes.nn.modules.linear8bitlt",
    }
)
_TORCHAO_RUNTIME_TYPES = frozenset(
    {
        "torchao.dtypes.affine_quantized_tensor.affinequantizedtensor",
        "torchao.quantization.int8tensor",
        "torchao.quantization.quantize_.workflows.int8.int8_tensor.int8tensor",
    }
)
_GPTQMODEL_QLINEAR_PREFIX = "gptqmodel.nn_modules.qlinear."
_GPTQMODEL_AWQ_MODULES = frozenset(
    {
        "bitblas_awq",
        "exllamav2_awq",
        "gemm_awq",
        "gemm_awq_triton",
        "gemv_awq",
        "gemv_fast_awq",
        "machete_awq",
        "marlin_awq",
        "torch_aten_kernel_awq",
        "torch_awq",
        "torch_fused_awq",
        "torch_int8_awq",
    }
)
_GPTQMODEL_GPTQ_MODULES = frozenset(
    {
        "bitblas",
        "exllamav2",
        "machete",
        "marlin",
        "torch",
        "torch_aten_kernel",
        "torch_fused",
        "torch_int8",
        "tritonv2",
    }
)
TRAINING_EVIDENCE_ARTIFACTS = frozenset(
    {
        "training_receipt.json",
        "training_binding.json",
        "training_evidence_proof.json",
        "training_profile_snapshot.json",
    }
)
TRAINING_SNAPSHOT_SCOPES = frozenset({"all", "attn", "ffn"})


@dataclass(frozen=True)
class ValidationIssue:
    target: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"target": self.target, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class AcceptanceInputs:
    baseline_report: Path
    policy_pack: Path
    expected_runtime_image_digest: str
    python_bin: str = sys.executable


@dataclass(frozen=True)
class VerificationInputSnapshots:
    report: bytes
    runtime_manifest: bytes
    baseline_report: bytes
    policy_pack: bytes


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite_json_value(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r}")


def _read_regular_snapshot(path: Path, *, label: str) -> bytes:
    """Read one regular file descriptor exactly once without following symlinks.

    Descriptor identity and metadata are checked around the read so replacement
    or in-place mutation cannot produce an accepted mixed snapshot.
    """

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a readable regular file: {exc}") from exc
    try:
        try:
            before = os.fstat(descriptor)
            path_state = os.lstat(path)
            if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(path_state.st_mode):
                raise ValueError(f"{label} must be a regular file")
            if (before.st_dev, before.st_ino) != (path_state.st_dev, path_state.st_ino):
                raise ValueError(f"{label} path changed while it was being opened")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                raw = handle.read()
            after = os.fstat(descriptor)
        except OSError as exc:
            raise ValueError(f"{label} could not be read safely: {exc}") from exc
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"{label} changed while it was being read")
    return raw


def _parse_strict_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_value,
        )
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON: {exc}") from exc


def _read_strict_json_snapshot(path: Path, *, label: str) -> tuple[bytes, Any]:
    raw = _read_regular_snapshot(path, label=label)
    return raw, _parse_strict_json(raw, label=label)


def _read_source_matrix_snapshot(path: Path) -> dict[str, Any]:
    _, payload = _read_strict_json_snapshot(path, label="source matrix")
    if not isinstance(payload, dict):
        raise ValueError("source matrix must contain a JSON object")
    return payload


def _require_exact_fields(
    payload: dict[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(f"{context} fields must match exactly: {', '.join(details)}")


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{context} must be a nonempty canonical string")
    return value


def _validate_source_matrix_shape(payload: dict[str, Any]) -> list[dict[str, Any]]:
    _require_exact_fields(
        payload,
        SOURCE_MATRIX_TOP_LEVEL_FIELDS,
        context="source matrix",
    )
    if payload["schema"] != SOURCE_MATRIX_SCHEMA:
        raise ValueError("unsupported source matrix schema")
    _require_nonempty_string(
        payload["description"], context="source matrix description"
    )
    entries = payload["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("source matrix entries must be a nonempty list")

    validated: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for index, entry in enumerate(entries):
        context = f"source matrix entry {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{context} must be an object")
        target = _require_nonempty_string(
            entry.get("target"), context=f"{context} target"
        )
        if target not in SOURCE_MATRIX_TARGETS:
            raise ValueError(f"{context} target is not canonical: {target!r}")
        if target in seen_targets:
            raise ValueError(f"source matrix target is duplicated: {target!r}")
        seen_targets.add(target)

        expected_entry_fields = SOURCE_MATRIX_ENTRY_FIELDS
        if target in SOURCE_MATRIX_TRAINING_TARGETS:
            expected_entry_fields |= SOURCE_MATRIX_TRAINING_FIELDS
        _require_exact_fields(entry, expected_entry_fields, context=context)

        for field in expected_entry_fields - {
            "expected",
            "provenance_artifacts",
            "required_artifacts",
            "runner_enforcement",
            "runtime_image",
        }:
            _require_nonempty_string(entry[field], context=f"{context} {field}")

        runtime_image = entry["runtime_image"]
        if not isinstance(runtime_image, dict):
            raise ValueError(f"{context} runtime_image must be an object")
        _require_exact_fields(
            runtime_image,
            SOURCE_MATRIX_RUNTIME_IMAGE_FIELDS,
            context=f"{context} runtime_image",
        )
        for field, value in runtime_image.items():
            _require_nonempty_string(value, context=f"{context} runtime_image.{field}")

        expected = entry["expected"]
        if not isinstance(expected, dict):
            raise ValueError(f"{context} expected must be an object")
        _require_exact_fields(
            expected,
            SOURCE_MATRIX_EXPECTED_FIELDS,
            context=f"{context} expected",
        )
        for field in {
            "lane_artifact_label",
            "runtime_provenance_declared",
            "runtime_provenance_status",
            "verify_status",
        }:
            _require_nonempty_string(
                expected[field], context=f"{context} expected.{field}"
            )
        for field in {
            "runtime_expected_digest_matched",
            "runtime_provenance_verified",
        }:
            if not isinstance(expected[field], bool):
                raise ValueError(f"{context} expected.{field} must be a boolean")

        for field in ("required_artifacts", "provenance_artifacts"):
            values = entry[field]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{context} {field} must be a nonempty list")
            canonical_values = [
                _require_nonempty_string(value, context=f"{context} {field} item")
                for value in values
            ]
            if len(set(canonical_values)) != len(canonical_values):
                raise ValueError(f"{context} {field} must not contain duplicates")

        runner_enforcement = entry["runner_enforcement"]
        if not isinstance(runner_enforcement, dict):
            raise ValueError(f"{context} runner_enforcement must be an object")
        expected_runner_fields = (
            SOURCE_MATRIX_RUNNER_ENFORCEMENT_FIELDS
            if entry["subject_adapter"] in MODULE_BACKED_QUANTIZED_ADAPTER_BACKENDS
            else frozenset()
        )
        _require_exact_fields(
            runner_enforcement,
            expected_runner_fields,
            context=f"{context} runner_enforcement",
        )
        for field, value in runner_enforcement.items():
            _require_nonempty_string(
                value,
                context=f"{context} runner_enforcement.{field}",
            )
        validated.append(entry)
    return validated


def _capture_verification_inputs(
    *,
    report_path: Path,
    acceptance_inputs: AcceptanceInputs,
) -> VerificationInputSnapshots:
    return VerificationInputSnapshots(
        report=_read_regular_snapshot(report_path, label="evaluation report"),
        runtime_manifest=_read_regular_snapshot(
            report_path.with_name("runtime.manifest.json"),
            label="runtime manifest",
        ),
        baseline_report=_read_regular_snapshot(
            acceptance_inputs.baseline_report,
            label="acceptance baseline report",
        ),
        policy_pack=_read_regular_snapshot(
            acceptance_inputs.policy_pack,
            label="acceptance policy pack",
        ),
    )


def _first_verification(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return {}
    result = results[0]
    if not isinstance(result, dict):
        return {}
    verification = result.get("verification")
    return verification if isinstance(verification, dict) else {}


def _verification_comparison_surface(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    verification = _first_verification(payload)
    results = payload.get("results")
    result = results[0] if isinstance(results, list) and len(results) == 1 else {}
    if not isinstance(result, dict):
        result = {}
    return {
        "summary": payload.get("summary"),
        "result": {
            "ok": result.get("ok"),
            "reason": result.get("reason"),
            "ratio_vs_baseline": result.get("ratio_vs_baseline"),
        },
        "runtime_provenance": verification.get("runtime_provenance"),
        "receipt": verification.get("receipt"),
    }


def _validate_verification_output_shape(
    *, target: str, path: Path, payload: Any
) -> list[ValidationIssue]:
    def issue(message: str) -> list[ValidationIssue]:
        return [ValidationIssue(target=target, path=str(path), message=message)]

    if not isinstance(payload, dict):
        return issue("verify artifact must contain a JSON object")
    if payload.get("format_version") != "verify-v1":
        return issue("verify artifact format_version must be verify-v1")
    if not isinstance(payload.get("summary"), dict):
        return issue("verify artifact summary must contain an object")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return issue("verify artifact results must contain exactly one result")
    if not isinstance(results[0], dict):
        return issue("verify artifact result must contain an object")
    verification = results[0].get("verification")
    if not isinstance(verification, dict):
        return issue("verify artifact result.verification must contain an object")
    if not isinstance(verification.get("receipt"), dict):
        return issue("verify artifact verification.receipt must contain an object")
    return []


def _validate_verification_receipt(
    *,
    target: str,
    verify_path: Path,
    payload: Any,
    snapshots: VerificationInputSnapshots,
    acceptance_inputs: AcceptanceInputs,
    verification_profile: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    receipt = _first_verification(payload).get("receipt")
    if not isinstance(receipt, dict):
        return [
            ValidationIssue(
                target=target,
                path=str(verify_path),
                message="verify artifact is missing its cryptographic input receipt",
            )
        ]
    expected = {
        "subject_report_sha256": hashlib.sha256(snapshots.report).hexdigest(),
        "baseline_report_sha256": hashlib.sha256(snapshots.baseline_report).hexdigest(),
        "policy_pack_sha256": hashlib.sha256(snapshots.policy_pack).hexdigest(),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=f"verify receipt {key} does not bind the reviewed input",
                )
            )
    receipt_inputs = receipt.get("inputs")
    if not isinstance(receipt_inputs, dict):
        receipt_inputs = {}
    expected_inputs = {
        "profile": verification_profile,
        "assurance_mode": "strict",
        "expected_runtime_image_digest": (
            acceptance_inputs.expected_runtime_image_digest
        ),
    }
    for key, value in expected_inputs.items():
        if receipt_inputs.get(key) != value:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=f"verify receipt input {key} does not match acceptance policy",
                )
            )
    return issues


def _replay_strict_verification(
    *,
    repo_root: Path,
    target: str,
    report_path: Path,
    verify_path: Path,
    stored_payload: Any,
    acceptance_inputs: AcceptanceInputs,
    snapshots: VerificationInputSnapshots,
    verification_profile: str,
) -> list[ValidationIssue]:
    env = os.environ.copy()
    source_path = str(repo_root / "src")
    inherited_pythonpath = [
        str(Path(entry).resolve())
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    env["PYTHONPATH"] = os.pathsep.join([source_path, *inherited_pythonpath])
    with tempfile.TemporaryDirectory(prefix="invarlock-source-matrix-") as temp_dir:
        snapshot_dir = Path(temp_dir)
        snapshot_report = snapshot_dir / "evaluation.report.json"
        snapshot_manifest = snapshot_dir / "runtime.manifest.json"
        snapshot_baseline = snapshot_dir / "acceptance-baseline.json"
        snapshot_policy = snapshot_dir / "acceptance-policy-pack"
        snapshot_report.write_bytes(snapshots.report)
        snapshot_manifest.write_bytes(snapshots.runtime_manifest)
        snapshot_baseline.write_bytes(snapshots.baseline_report)
        snapshot_policy.write_bytes(snapshots.policy_pack)
        command = [
            acceptance_inputs.python_bin,
            "-m",
            "invarlock",
            "verify",
            str(snapshot_report),
            "--json",
            "--profile",
            verification_profile,
            "--assurance",
            "strict",
            "--runtime-provenance",
            "container",
            "--baseline",
            str(snapshot_baseline),
            "--policy-pack",
            str(snapshot_policy),
            "--expected-runtime-image-digest",
            acceptance_inputs.expected_runtime_image_digest,
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        detail = (completed.stdout or completed.stderr).strip().replace("\n", " ")
        return [
            ValidationIssue(
                target=target,
                path=str(report_path),
                message=(
                    "canonical strict verifier replay failed"
                    + (f": {detail[:500]}" if detail else "")
                ),
            )
        ]
    try:
        replay_payload = _parse_strict_json(
            completed.stdout.encode("utf-8"),
            label="canonical verifier replay",
        )
    except ValueError as exc:
        return [
            ValidationIssue(
                target=target,
                path=str(report_path),
                message=f"canonical verifier replay returned invalid JSON: {exc}",
            )
        ]
    issues = _validate_verification_output_shape(
        target=target,
        path=verify_path,
        payload=replay_payload,
    )
    issues.extend(
        _validate_verification_receipt(
            target=target,
            verify_path=verify_path,
            payload=replay_payload,
            snapshots=snapshots,
            acceptance_inputs=acceptance_inputs,
            verification_profile=verification_profile,
        )
    )
    if _verification_comparison_surface(stored_payload) != (
        _verification_comparison_surface(replay_payload)
    ):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(verify_path),
                message="stored verify artifact does not match canonical strict replay",
            )
        )
    return issues


def _report_dir(repo_root: Path, entry: dict[str, Any]) -> Path:
    readme_parent = (repo_root / str(entry["readme"])).parent
    report_path = str(entry["report_path"]).replace(
        "<artifact-lane>", str(entry["lane"])
    )
    return readme_parent / report_path


def _verify_status(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    if summary.get("status"):
        return str(summary["status"])
    if summary.get("reason"):
        return str(summary["reason"])
    if summary.get("ok") is True:
        return "ok"
    return None


def _runtime_provenance(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    for result in results:
        if not isinstance(result, dict):
            continue
        verification = result.get("verification")
        if not isinstance(verification, dict):
            continue
        runtime = verification.get("runtime_provenance")
        if isinstance(runtime, dict):
            return runtime
    return {}


_SUMMARY_FIELDS = frozenset(
    {
        "assurance",
        "baseline",
        "device",
        "execution_mode",
        "html",
        "lane_artifact",
        "lane_artifact_label",
        "report",
        "run_command",
        "runtime_provenance",
        "runtime_quantization_proof",
        "status",
        "subject",
        "summary",
        "training_binding",
        "training_binding_status",
        "verify",
        "verify_error",
        "verify_reason",
        "verify_runtime_provenance_declared",
        "verify_runtime_provenance_issues",
        "verify_runtime_provenance_status",
        "verify_runtime_provenance_verified",
        "verify_status",
    }
)


def _summary_fields(path: Path) -> dict[str, str]:
    raw = _read_regular_snapshot(path, label="run summary")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"run summary is not UTF-8: {exc}") from exc
    fields: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or ":" not in line:
            raise ValueError(
                f"run summary line {line_number} must be one nonempty key/value field"
            )
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in _SUMMARY_FIELDS:
            raise ValueError(
                f"run summary line {line_number} has unknown field {key!r}"
            )
        if key in fields:
            raise ValueError(f"run summary duplicates field {key!r}")
        if not value:
            raise ValueError(f"run summary field {key!r} must not be empty")
        fields[key] = value
    return fields


def _command_fields(path: Path) -> dict[str, tuple[str, ...]]:
    raw = _read_regular_snapshot(path, label="run command")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"run command is not UTF-8: {exc}") from exc
    fields: dict[str, tuple[str, ...]] = {}
    allowed = {"wrapper", "evaluate", "verify", "html"}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or ":" not in line:
            raise ValueError(
                f"run command line {line_number} must be one nonempty command field"
            )
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in allowed:
            raise ValueError(
                f"run command line {line_number} has unknown field {key!r}"
            )
        if key in fields:
            raise ValueError(f"run command duplicates field {key!r}")
        try:
            tokens = tuple(shlex.split(value.strip(), posix=True))
        except ValueError as exc:
            raise ValueError(
                f"run command field {key!r} is not valid shell argv"
            ) from exc
        if not tokens:
            raise ValueError(f"run command field {key!r} must not be empty")
        fields[key] = tokens
    if "wrapper" not in fields or "evaluate" not in fields:
        raise ValueError(
            "run command must contain exactly one wrapper and evaluate command"
        )
    return fields


def _read_required_json_object(
    *,
    target: str,
    path: Path,
    artifact_name: str,
) -> tuple[dict[str, Any] | None, ValidationIssue | None]:
    return _read_strict_json_object(
        target=target,
        path=path,
        artifact_name=artifact_name,
    )


def _read_strict_json_object(
    *,
    target: str,
    path: Path,
    artifact_name: str,
) -> tuple[dict[str, Any] | None, ValidationIssue | None]:
    """Read one sidecar without accepting duplicate keys or non-finite values.

    The source-matrix validator is a consumer boundary.  Its acceptance must
    not depend on Python's default last-key-wins JSON behavior, especially for
    a sidecar whose ``ok`` value controls strict-lane eligibility.
    """

    try:
        _, payload = _read_strict_json_snapshot(path, label=artifact_name)
    except (OSError, UnicodeError, ValueError) as exc:
        return None, ValidationIssue(
            target=target,
            path=str(path),
            message=f"{artifact_name} is not strict JSON: {exc}",
        )
    if not isinstance(payload, dict):
        return None, ValidationIssue(
            target=target,
            path=str(path),
            message=f"{artifact_name} must contain a JSON object",
        )
    return payload, None


def _runtime_type_matches_adapter(
    *,
    adapter: str,
    type_name: str,
    quantization_method: str | None,
) -> bool:
    """Independently recognize the narrow live type families a strict lane needs.

    This deliberately does not call the producer-side recognizer.  The
    validator should catch a producer regression that made a dense or
    cross-family runtime appear quantized.
    """

    normalized = type_name.casefold()
    if adapter == "hf_bnb":
        return normalized in _BITSANDBYTES_RUNTIME_TYPES
    if adapter == "hf_torchao":
        return normalized in _TORCHAO_RUNTIME_TYPES
    if adapter == "hf_hqq":
        return normalized.startswith("hqq.") and "hqqlinear" in normalized
    if adapter == "hf_quanto":
        return normalized == "optimum.quanto.nn.qlinear.qlinear"
    if adapter not in {"hf_awq", "hf_gptq"}:
        return False
    if not normalized.startswith(_GPTQMODEL_QLINEAR_PREFIX):
        return False
    remainder = normalized.removeprefix(_GPTQMODEL_QLINEAR_PREFIX)
    module_name, separator, _class_name = remainder.partition(".")
    if not separator:
        # A generic base QLinear has no family in its qualified type.  The
        # explicit live quantization method is necessary to make it useful.
        return "linear" in normalized and quantization_method == (
            "awq" if adapter == "hf_awq" else "gptq"
        )
    family_modules = (
        _GPTQMODEL_AWQ_MODULES if adapter == "hf_awq" else _GPTQMODEL_GPTQ_MODULES
    )
    return module_name in family_modules


def _validate_runtime_quantization_proof(
    *,
    target: str,
    path: Path,
    payload: dict[str, Any],
    expected_adapter: str,
    expected_backend: str,
    backend_inventory: dict[str, Any] | None,
) -> list[ValidationIssue]:
    """Validate an authoritative live-runtime representation sidecar.

    This proof says only that the evaluated process exposed recognized,
    backend-specific runtime types.  It is intentionally not treated as a
    packed-storage or checkpoint-artifact proof; those have separate contracts.
    """

    issues: list[ValidationIssue] = []

    def add(message: str) -> None:
        issues.append(ValidationIssue(target=target, path=str(path), message=message))

    required_fields = {
        "schema",
        "proof_kind",
        "adapter",
        "backend",
        "backend_version",
        "ok",
        "status",
        "reason",
        "live_model_observed",
        "module_inventory_observed",
        "recognized_quantized_runtime_type_count",
        "recognized_quantized_runtime_types",
        "recognized_quantized_runtime_observation_kinds",
        "live_model_quantization_method",
        "backend_runtime_importable",
        "backend_runtime_import_error_type",
        "backend_runtime_version",
        "backend_runtime_compatibility_bridge_required",
        "backend_runtime_compatibility_bridge_applied",
        "backend_runtime_compatibility_bridge_error_type",
        "packed_storage_artifact_proof_required",
        "artifact_binding",
    }
    missing_fields = sorted(required_fields.difference(payload))
    unexpected_fields = sorted(set(payload).difference(required_fields))
    if missing_fields:
        add(
            "runtime quantization proof is missing required fields: "
            + ", ".join(missing_fields)
        )
    if unexpected_fields:
        add(
            "runtime quantization proof has unsupported v1 fields: "
            + ", ".join(unexpected_fields)
        )

    if payload.get("schema") != RUNTIME_QUANTIZATION_PROOF_SCHEMA:
        add(
            "runtime quantization proof schema mismatch: "
            f"expected {RUNTIME_QUANTIZATION_PROOF_SCHEMA!r}, "
            f"got {payload.get('schema')!r}"
        )
    if payload.get("proof_kind") != RUNTIME_QUANTIZATION_PROOF_KIND:
        add("runtime quantization proof kind is not a live runtime type inventory")
    if payload.get("adapter") != expected_adapter:
        add(
            "runtime quantization proof adapter mismatch: "
            f"expected {expected_adapter!r}, got {payload.get('adapter')!r}"
        )
    if payload.get("backend") != expected_backend:
        add(
            "runtime quantization proof backend mismatch: "
            f"expected {expected_backend!r}, got {payload.get('backend')!r}"
        )
    if (
        not isinstance(payload.get("backend_version"), str)
        or not payload["backend_version"].strip()
    ):
        add("runtime quantization proof must record a non-empty backend_version")
    if payload.get("ok") is not True:
        add("runtime quantization proof does not record ok: true")
    if payload.get("status") != "verified_live_runtime_types":
        add("runtime quantization proof status must be verified_live_runtime_types")
    if payload.get("reason") != "recognized_live_quantized_runtime_types":
        add(
            "runtime quantization proof reason must be "
            "recognized_live_quantized_runtime_types"
        )
    if payload.get("live_model_observed") is not True:
        add("runtime quantization proof does not record a live model observation")
    if payload.get("module_inventory_observed") is not True:
        add("runtime quantization proof does not record a module inventory observation")
    if payload.get("packed_storage_artifact_proof_required") is not False:
        add("runtime quantization proof must not stand in for packed-storage proof")
    if payload.get("artifact_binding") != "not_attempted":
        add(
            "runtime quantization proof artifact_binding must be not_attempted; "
            "it is live-runtime evidence, not an artifact proof"
        )

    count = payload.get("recognized_quantized_runtime_type_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        add(
            "runtime quantization proof recognized_quantized_runtime_type_count "
            "must be a positive integer"
        )
    runtime_types = payload.get("recognized_quantized_runtime_types")
    valid_runtime_types = isinstance(runtime_types, list) and all(
        isinstance(value, str) and value.strip() == value and value
        for value in runtime_types
    )
    if not valid_runtime_types or not runtime_types:
        add(
            "runtime quantization proof must contain non-empty recognized "
            "runtime type names"
        )
        runtime_types = []
    elif runtime_types != sorted(set(runtime_types)):
        add("runtime quantization proof runtime type names must be sorted and unique")
    if isinstance(count, int) and not isinstance(count, bool) and runtime_types:
        if count < len(runtime_types):
            add(
                "runtime quantization proof type count is smaller than its unique "
                "runtime type inventory"
            )
    observation_kinds = payload.get("recognized_quantized_runtime_observation_kinds")
    if (
        not isinstance(observation_kinds, list)
        or not observation_kinds
        or not all(kind in {"module", "direct_weight"} for kind in observation_kinds)
    ):
        add(
            "runtime quantization proof observation kinds must be a non-empty "
            "supported list"
        )
        observation_kinds = []
    elif observation_kinds != sorted(set(observation_kinds)):
        add("runtime quantization proof observation kinds must be sorted and unique")

    raw_quantization_method = payload.get("live_model_quantization_method")
    quantization_method = (
        raw_quantization_method if isinstance(raw_quantization_method, str) else None
    )
    if raw_quantization_method is not None and quantization_method not in {
        "awq",
        "gptq",
    }:
        add(
            "runtime quantization proof live_model_quantization_method must be "
            "awq, gptq, or null"
        )
    expected_method = (
        "awq"
        if expected_adapter == "hf_awq"
        else "gptq"
        if expected_adapter == "hf_gptq"
        else None
    )
    if expected_method is not None and quantization_method not in {
        None,
        expected_method,
    }:
        add(
            "runtime quantization proof live quantization method does not match "
            f"{expected_adapter!r}"
        )
    for type_name in runtime_types:
        if not _runtime_type_matches_adapter(
            adapter=expected_adapter,
            type_name=type_name,
            quantization_method=quantization_method,
        ):
            add(
                "runtime quantization proof contains an unrecognized or "
                f"cross-family runtime type for {expected_adapter!r}: {type_name!r}"
            )

    if expected_adapter in {"hf_awq", "hf_gptq"}:
        if payload.get("backend_runtime_importable") is not True:
            add(
                "GPTQModel runtime proof does not record backend_runtime_importable: true"
            )
        if payload.get("backend_runtime_import_error_type") is not None:
            add("GPTQModel runtime proof records a backend runtime import error")
        if (
            not isinstance(payload.get("backend_runtime_version"), str)
            or not payload["backend_runtime_version"].strip()
        ):
            add("GPTQModel runtime proof must record a non-empty runtime version")
        bridge_required = payload.get("backend_runtime_compatibility_bridge_required")
        bridge_applied = payload.get("backend_runtime_compatibility_bridge_applied")
        bridge_error = payload.get("backend_runtime_compatibility_bridge_error_type")
        if not isinstance(bridge_required, bool):
            add("GPTQModel runtime proof bridge-required field must be boolean")
        if not isinstance(bridge_applied, bool):
            add("GPTQModel runtime proof bridge-applied field must be boolean")
        if bridge_required is True and bridge_applied is not True:
            add("GPTQModel runtime proof required compatibility bridge was not applied")
        if bridge_error is not None:
            add("GPTQModel runtime proof records a compatibility bridge error")
    else:
        for field in (
            "backend_runtime_importable",
            "backend_runtime_import_error_type",
            "backend_runtime_version",
            "backend_runtime_compatibility_bridge_required",
            "backend_runtime_compatibility_bridge_applied",
            "backend_runtime_compatibility_bridge_error_type",
        ):
            if payload.get(field) is not None:
                add(f"non-GPTQModel runtime proof must record {field} as null")

    if backend_inventory is not None:
        if backend_inventory.get("adapter") != expected_adapter:
            add(
                "runtime quantization proof cannot bind a mismatched backend inventory adapter"
            )
        if backend_inventory.get("backend") != expected_backend:
            add(
                "runtime quantization proof cannot bind a mismatched backend inventory backend"
            )
        if payload.get("adapter") != backend_inventory.get("adapter"):
            add("runtime quantization proof adapter does not match backend inventory")
        if payload.get("backend") != backend_inventory.get("backend"):
            add("runtime quantization proof backend does not match backend inventory")
        if payload.get("backend_version") != backend_inventory.get("backend_version"):
            add(
                "runtime quantization proof backend_version does not match "
                "backend inventory"
            )
        if count != backend_inventory.get("quantized_module_count"):
            add(
                "runtime quantization proof observation count does not match "
                "backend inventory"
            )
        if runtime_types != backend_inventory.get("quantized_module_types"):
            add(
                "runtime quantization proof runtime types do not exactly match "
                "backend inventory"
            )
        if observation_kinds != backend_inventory.get("quantized_observation_kinds"):
            add(
                "runtime quantization proof observation kinds do not match "
                "backend inventory"
            )

    return issues


def _validate_backend_inventory(
    *,
    target: str,
    path: Path,
    payload: dict[str, Any],
    expected_adapter: str | None,
    expected_backend: str | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_fields = {
        "schema",
        "adapter",
        "backend",
        "backend_version",
        "transformers_version",
        "quantization_config",
        "quantized_module_count",
        "quantized_module_types",
        "quantized_observation_kinds",
        "device_map",
        "memory_footprint",
        "load_smoke",
        "inference_smoke",
    }
    missing_fields = sorted(required_fields.difference(payload))
    unexpected_fields = sorted(set(payload).difference(required_fields))
    if missing_fields:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory is missing required fields: "
                    + ", ".join(missing_fields)
                ),
            )
        )
    if unexpected_fields:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory has unsupported v1 fields: "
                    + ", ".join(unexpected_fields)
                ),
            )
        )
    if payload.get("schema") != BACKEND_INVENTORY_SCHEMA:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory schema mismatch: "
                    f"expected {BACKEND_INVENTORY_SCHEMA!r}, "
                    f"got {payload.get('schema')!r}"
                ),
            )
        )
    if expected_adapter and payload.get("adapter") != expected_adapter:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory adapter mismatch: "
                    f"expected {expected_adapter!r}, got {payload.get('adapter')!r}"
                ),
            )
        )
    if expected_backend and payload.get("backend") != expected_backend:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory backend mismatch: "
                    f"expected {expected_backend!r}, got {payload.get('backend')!r}"
                ),
            )
        )
    if (
        not isinstance(payload.get("backend_version"), str)
        or not payload["backend_version"].strip()
    ):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message="backend inventory backend_version must be non-empty",
            )
        )
    quantized_count = payload.get("quantized_module_count")
    if (
        isinstance(quantized_count, bool)
        or not isinstance(quantized_count, int)
        or quantized_count < 0
    ):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory quantized_module_count must be a "
                    "non-negative integer"
                ),
            )
        )
    for field in ("load_smoke", "inference_smoke"):
        if payload.get(field) is not True:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(path),
                    message=f"backend inventory must record {field}: true",
                )
            )
    module_types = payload.get("quantized_module_types")
    if not isinstance(module_types, list) or not all(
        isinstance(value, str) and value and value.strip() == value
        for value in module_types
    ):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message="backend inventory quantized_module_types must be a string list",
            )
        )
        module_types = []
    elif module_types != sorted(set(module_types)):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory quantized_module_types must be sorted and unique"
                ),
            )
        )
    if isinstance(quantized_count, int) and quantized_count < len(module_types):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message="backend inventory count is smaller than its type inventory",
            )
        )
    observation_kinds = payload.get("quantized_observation_kinds")
    if (
        not isinstance(observation_kinds, list)
        or not observation_kinds
        or not all(kind in {"module", "direct_weight"} for kind in observation_kinds)
    ):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory observation kinds must be a non-empty "
                    "supported list"
                ),
            )
        )
    elif observation_kinds != sorted(set(observation_kinds)):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message="backend inventory observation kinds must be sorted and unique",
            )
        )
    return issues


def _validate_runtime_manifest(
    *,
    target: str,
    path: Path,
    payload: dict[str, Any],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def require_fields(
        value: Any,
        *,
        label: str,
        required: set[str],
        optional: set[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(path),
                    message=f"runtime manifest {label} must contain an object",
                )
            )
            return {}
        allowed = required | (optional or set())
        missing = sorted(required.difference(value))
        unexpected = sorted(set(value).difference(allowed))
        if missing or unexpected:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(path),
                    message=(
                        f"runtime manifest {label} fields do not match v1 "
                        f"(missing={missing}, extra={unexpected})"
                    ),
                )
            )
        return value

    require_fields(
        payload,
        label="root",
        required={
            "manifest_version",
            "generated_at_utc",
            "verifier_contract_version",
            "report",
            "config",
            "execution_mode",
            "runtime",
        },
        optional={"context"},
    )
    require_fields(
        payload.get("report"),
        label="report",
        required={"path", "filename", "sha256"},
    )
    require_fields(
        payload.get("config"),
        label="config",
        required={"path", "sha256", "source"},
    )
    runtime = payload.get("runtime")
    runtime = require_fields(
        runtime,
        label="runtime",
        required={
            "image_ref",
            "image_digest",
            "container_execution",
            "allow_network",
            "allow_remote_code",
            "allow_third_party_plugins",
        },
    )
    if not str(runtime.get("image_digest") or "").strip():
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message="runtime manifest runtime.image_digest must be present",
            )
        )
    if not str(runtime.get("image_ref") or "").strip():
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message="runtime manifest runtime.image_ref must be present",
            )
        )
    return issues


def _validate_training_binding(
    *,
    repo_root: Path,
    target: str,
    report_dir: Path,
    expected_training_profile: Any,
) -> list[ValidationIssue]:
    binding_path = report_dir / "training_binding.json"
    receipt_path = report_dir / "training_receipt.json"
    if not binding_path.is_file() or not receipt_path.is_file():
        return []
    try:
        _, binding_value = _read_strict_json_snapshot(
            binding_path,
            label="training binding",
        )
        receipt_bytes, receipt_value = _read_strict_json_snapshot(
            receipt_path,
            label="training receipt",
        )
        report_bytes = _read_regular_snapshot(
            report_dir / "evaluation.report.json",
            label="training evaluation report",
        )
        verify_bytes = _read_regular_snapshot(
            report_dir / "verify.json",
            label="training verify artifact",
        )
    except ValueError as exc:
        return [
            ValidationIssue(
                target=target,
                path=str(binding_path),
                message=f"training binding inputs are invalid: {exc}",
            )
        ]
    if not isinstance(binding_value, dict):
        return [
            ValidationIssue(
                target=target,
                path=str(binding_path),
                message="training binding must contain a JSON object",
            )
        ]
    binding = binding_value
    issues: list[ValidationIssue] = []
    if not isinstance(expected_training_profile, str) or not expected_training_profile:
        return [
            ValidationIssue(
                target=target,
                path=str(binding_path),
                message="source matrix training target is missing training_profile",
            )
        ]
    editing_path = repo_root / "scripts" / "evidence_packs" / "python"
    if str(editing_path) not in sys.path:
        sys.path.insert(0, str(editing_path))
    try:
        from editing.training_contract import (
            TrainingProfileError,
            load_training_profile,
        )
        from editing.training_receipt import (
            TrainingReceiptError,
            require_valid_training_receipt,
        )

        if not isinstance(receipt_value, dict):
            raise TrainingReceiptError("training receipt JSON root must be an object")
        receipt_payload = receipt_value
        profile = load_training_profile(
            expected_training_profile,
            profiles_path=(
                repo_root / "scripts" / "evidence_packs" / "training_profiles.json"
            ),
            repo_root=repo_root,
        )
        receipt = require_valid_training_receipt(receipt_payload, profile=profile)
    except (
        OSError,
        UnicodeError,
        ValueError,
        TrainingProfileError,
        TrainingReceiptError,
    ) as exc:
        return [
            ValidationIssue(
                target=target,
                path=str(receipt_path),
                message=f"training receipt/profile contract failed: {exc}",
            )
        ]
    expected_binding_fields = {
        "evaluation_report_sha256",
        "receipt_sha256",
        "schema",
        "subject_tree_sha256",
        "training_receipt_file_sha256",
        "verified",
        "verify_artifact_sha256",
    }
    if set(binding) != expected_binding_fields:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(binding_path),
                message="training binding fields must match v1 exactly",
            )
        )
    if binding.get("schema") != "invarlock.integration_training_binding.v1":
        issues.append(
            ValidationIssue(
                target=target,
                path=str(binding_path),
                message="training binding schema mismatch",
            )
        )
    if binding.get("verified") is not True:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(binding_path),
                message="training binding does not record verified: true",
            )
        )
    if (
        binding.get("training_receipt_file_sha256")
        != hashlib.sha256(receipt_bytes).hexdigest()
    ):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(binding_path),
                message="training binding does not match copied training receipt",
            )
        )
    if binding.get("receipt_sha256") != receipt.get("receipt_sha256"):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(binding_path),
                message="training binding receipt_sha256 does not match canonical receipt",
            )
        )
    subject_tree_sha256 = receipt.get("hashes", {}).get("subject_tree_sha256")
    if binding.get("subject_tree_sha256") != subject_tree_sha256:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(binding_path),
                message=(
                    "training binding subject_tree_sha256 does not match "
                    "canonical receipt"
                ),
            )
        )
    bound_artifacts = {
        "evaluation_report_sha256": ("evaluation.report.json", report_bytes),
        "verify_artifact_sha256": ("verify.json", verify_bytes),
    }
    for key, (artifact_name, artifact_bytes) in bound_artifacts.items():
        if binding.get(key) != hashlib.sha256(artifact_bytes).hexdigest():
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(binding_path),
                    message=f"training binding does not match {artifact_name}",
                )
            )
    return issues


def _validate_training_evidence(
    *,
    repo_root: Path,
    target: str,
    report_dir: Path,
    expected_training_profile: Any,
    expected_training_scope: Any,
) -> list[ValidationIssue]:
    artifact_paths = {
        "training receipt": report_dir / "training_receipt.json",
        "training binding": report_dir / "training_binding.json",
        "training evidence proof": report_dir / "training_evidence_proof.json",
        "training profile snapshot": report_dir / "training_profile_snapshot.json",
        "evaluation report": report_dir / "evaluation.report.json",
    }
    issues: list[ValidationIssue] = []
    for artifact_name, path in artifact_paths.items():
        if not path.is_file():
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(path),
                    message=f"{artifact_name} is missing",
                )
            )
    if issues:
        return issues

    if not isinstance(expected_training_profile, str) or not expected_training_profile:
        return [
            ValidationIssue(
                target=target,
                path=str(report_dir / "training_profile_snapshot.json"),
                message="source matrix training target is missing training_profile",
            )
        ]
    if expected_training_scope not in TRAINING_SNAPSHOT_SCOPES:
        return [
            ValidationIssue(
                target=target,
                path=str(report_dir / "training_profile_snapshot.json"),
                message=(
                    "source matrix training target has invalid training_scope; "
                    "expected all, attn, or ffn"
                ),
            )
        ]

    src_path = repo_root / "src"
    editing_path = repo_root / "scripts" / "evidence_packs" / "python"
    for import_path in (src_path, editing_path):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    try:
        from editing.training_contract import (
            TRAINING_PROFILES_SCHEMA,
            TrainingProfileError,
            load_training_profile,
        )
        from editing.training_profile_snapshot import TRAINING_PROFILE_SNAPSHOT_SCHEMA
        from editing.training_receipt import (
            TrainingReceiptError,
            require_valid_training_receipt,
        )

        from invarlock.evidence_pack_json import (
            StrictJsonError,
            read_json_object_snapshot,
        )
        from invarlock.training_evidence import (
            TrainingEvidenceProofError,
            require_valid_training_evidence_proof,
        )
    except ImportError as exc:
        return [
            ValidationIssue(
                target=target,
                path=str(report_dir),
                message=f"training evidence validation dependencies are unavailable: {exc}",
            )
        ]

    def read_strict_object(
        path: Path,
        artifact_name: str,
    ) -> tuple[bytes, dict[str, Any]] | None:
        try:
            return read_json_object_snapshot(path, label=artifact_name)
        except (OSError, UnicodeError, StrictJsonError) as exc:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(path),
                    message=f"{artifact_name} is not strict JSON: {exc}",
                )
            )
            return None

    profiles_path = repo_root / "scripts" / "evidence_packs" / "training_profiles.json"
    profiles_snapshot = read_strict_object(profiles_path, "immutable training profiles")
    receipt_snapshot = read_strict_object(
        artifact_paths["training receipt"], "training receipt"
    )
    # Semantic binding checks run in _validate_training_binding. Read the same
    # sidecar through the strict boundary here so a duplicate-key document
    # cannot become part of an otherwise accepted training evidence set.
    binding_snapshot = read_strict_object(
        artifact_paths["training binding"], "training binding"
    )
    proof_snapshot = read_strict_object(
        artifact_paths["training evidence proof"], "training evidence proof"
    )
    profile_snapshot = read_strict_object(
        artifact_paths["training profile snapshot"], "training profile snapshot"
    )
    report_snapshot = read_strict_object(
        artifact_paths["evaluation report"], "evaluation report"
    )
    if any(
        snapshot is None
        for snapshot in (
            profiles_snapshot,
            receipt_snapshot,
            binding_snapshot,
            proof_snapshot,
            profile_snapshot,
            report_snapshot,
        )
    ):
        return issues

    assert profiles_snapshot is not None
    assert receipt_snapshot is not None
    assert binding_snapshot is not None
    assert proof_snapshot is not None
    assert profile_snapshot is not None
    assert report_snapshot is not None
    _, profiles_document = profiles_snapshot
    _, receipt_payload = receipt_snapshot
    _, proof_payload = proof_snapshot
    profile_snapshot_bytes, profile_snapshot_payload = profile_snapshot
    _, report_payload = report_snapshot

    if (
        set(profiles_document) != {"schema", "profiles"}
        or profiles_document.get("schema") != TRAINING_PROFILES_SCHEMA
    ):
        return [
            *issues,
            ValidationIssue(
                target=target,
                path=str(profiles_path),
                message="immutable training profiles have an unknown schema",
            ),
        ]
    profiles = profiles_document.get("profiles")
    expected_profile_payload = (
        profiles.get(expected_training_profile) if isinstance(profiles, dict) else None
    )
    if not isinstance(expected_profile_payload, dict):
        return [
            *issues,
            ValidationIssue(
                target=target,
                path=str(profiles_path),
                message="source matrix training_profile is unavailable",
            ),
        ]
    try:
        profile = load_training_profile(
            expected_training_profile,
            profiles_path=profiles_path,
            repo_root=repo_root,
        )
        receipt = require_valid_training_receipt(receipt_payload, profile=profile)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TrainingProfileError,
        TrainingReceiptError,
    ) as exc:
        return [
            *issues,
            ValidationIssue(
                target=target,
                path=str(artifact_paths["training receipt"]),
                message=f"training receipt/profile contract failed: {exc}",
            ),
        ]

    metadata = report_payload.get("meta")
    baseline_ref = report_payload.get("baseline_ref")
    artifact_identity = (
        metadata.get("model_identity") if isinstance(metadata, dict) else None
    )
    baseline_identity = (
        baseline_ref.get("model_identity") if isinstance(baseline_ref, dict) else None
    )
    if not isinstance(artifact_identity, dict) or not isinstance(
        baseline_identity, dict
    ):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(artifact_paths["evaluation report"]),
                message=(
                    "evaluation report must expose baseline and artifact model identities "
                    "for training evidence validation"
                ),
            )
        )
    else:
        try:
            require_valid_training_evidence_proof(
                proof_payload,
                receipt,
                expected_edit_type=profile.edit_type,
                expected_baseline_identity=baseline_identity,
                expected_artifact_identity=artifact_identity,
            )
        except TrainingEvidenceProofError as exc:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(artifact_paths["training evidence proof"]),
                    message=f"training evidence proof contract failed: {exc}",
                )
            )

    expected_snapshot_fields = {
        "schema",
        "profile_id",
        "profile_sha256",
        "scope",
        "profile",
    }
    if set(profile_snapshot_payload) != expected_snapshot_fields:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(artifact_paths["training profile snapshot"]),
                message="training profile snapshot has an invalid field set",
            )
        )
    if profile_snapshot_payload.get("schema") != TRAINING_PROFILE_SNAPSHOT_SCHEMA:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(artifact_paths["training profile snapshot"]),
                message="training profile snapshot schema mismatch",
            )
        )
    if profile_snapshot_payload.get("profile_id") != expected_training_profile:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(artifact_paths["training profile snapshot"]),
                message="training profile snapshot profile_id does not match source matrix",
            )
        )
    if profile_snapshot_payload.get("profile_sha256") != profile.profile_sha256:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(artifact_paths["training profile snapshot"]),
                message="training profile snapshot digest does not match immutable profile",
            )
        )
    if profile_snapshot_payload.get("scope") != expected_training_scope:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(artifact_paths["training profile snapshot"]),
                message="training profile snapshot scope does not match source matrix",
            )
        )
    if profile_snapshot_payload.get("profile") != expected_profile_payload:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(artifact_paths["training profile snapshot"]),
                message="training profile snapshot profile does not match immutable profile",
            )
        )
    try:
        canonical_snapshot = (
            json.dumps(
                profile_snapshot_payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError):
        canonical_snapshot = b""
    if profile_snapshot_bytes != canonical_snapshot:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(artifact_paths["training profile snapshot"]),
                message="training profile snapshot is not canonical",
            )
        )
    return issues


def validate_entry(
    repo_root: Path,
    entry: dict[str, Any],
    *,
    acceptance_inputs: AcceptanceInputs | None,
) -> list[ValidationIssue]:
    target = str(entry["target"])
    issues: list[ValidationIssue] = []
    report_dir = _report_dir(repo_root, entry)
    expected = entry.get("expected", {})
    verification_profile = entry.get("verification_profile")
    if verification_profile not in {"ci", "release"}:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(repo_root / str(entry.get("readme", ""))),
                message="source matrix requires verification_profile ci or release",
            )
        )

    artifacts = entry.get("required_artifacts", [])
    if not isinstance(artifacts, list):
        return [
            ValidationIssue(
                target=target,
                path=str(report_dir),
                message="required_artifacts is not a list",
            )
        ]

    artifact_names = {artifact for artifact in artifacts if isinstance(artifact, str)}
    subject_adapter = entry.get("subject_adapter")
    expected_quantized_backend = (
        MODULE_BACKED_QUANTIZED_ADAPTER_BACKENDS.get(subject_adapter)
        if isinstance(subject_adapter, str)
        else None
    )
    is_strict_lane = entry.get("lane") == "cuda-container-strict"
    if (
        is_strict_lane
        and isinstance(subject_adapter, str)
        and subject_adapter in STRICT_UNSUPPORTED_QUANTIZED_ADAPTERS
    ):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(repo_root / str(entry.get("readme", ""))),
                message=(
                    "compressed-tensors (hf_ct) is not eligible for strict source "
                    "matrix validation until a packed-storage artifact proof exists"
                ),
            )
        )
    if is_strict_lane and expected_quantized_backend is not None:
        runner_enforcement = entry.get("runner_enforcement")
        if not isinstance(runner_enforcement, dict):
            runner_enforcement = {}
        for required_artifact in (
            "backend_inventory.json",
            RUNTIME_QUANTIZATION_PROOF_FILENAME,
        ):
            if required_artifact not in artifact_names:
                issues.append(
                    ValidationIssue(
                        target=target,
                        path=str(repo_root / str(entry.get("readme", ""))),
                        message=(
                            "strict module-backed quantized source matrix requires "
                            f"{required_artifact}"
                        ),
                    )
                )
        for enforcement_name, required_flag in {
            "backend_inventory": "--require-backend-inventory",
            "runtime_quantization_proof": "--require-runtime-quantization-proof",
        }.items():
            if runner_enforcement.get(enforcement_name) != required_flag:
                issues.append(
                    ValidationIssue(
                        target=target,
                        path=str(repo_root / str(entry.get("runner", ""))),
                        message=(
                            "strict module-backed quantized source matrix must bind "
                            f"{enforcement_name} to {required_flag}"
                        ),
                    )
                )
    has_training_contract = bool(
        entry.get("training_profile") is not None
        or artifact_names.intersection(TRAINING_EVIDENCE_ARTIFACTS)
    )
    if has_training_contract:
        missing_training_artifacts = sorted(
            TRAINING_EVIDENCE_ARTIFACTS.difference(artifact_names)
        )
        if missing_training_artifacts:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(repo_root / str(entry.get("readme", ""))),
                    message=(
                        "training source matrix is missing required evidence artifacts: "
                        + ", ".join(missing_training_artifacts)
                    ),
                )
            )
        if entry.get("training_scope") not in TRAINING_SNAPSHOT_SCOPES:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(repo_root / str(entry.get("readme", ""))),
                    message=(
                        "training source matrix requires training_scope all, attn, or ffn"
                    ),
                )
            )

    if not report_dir.is_dir():
        return [
            *issues,
            ValidationIssue(
                target=target,
                path=str(report_dir),
                message="report directory is missing",
            ),
        ]

    for artifact in artifacts:
        artifact_path = report_dir / str(artifact)
        if not artifact_path.is_file():
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(artifact_path),
                    message="required artifact is missing",
                )
            )
            continue
        try:
            artifact_bytes = _read_regular_snapshot(
                artifact_path,
                label=f"required artifact {artifact}",
            )
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(artifact_path),
                    message=str(exc),
                )
            )
            continue
        if not artifact_bytes:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(artifact_path),
                    message="required artifact is empty",
                )
            )
            continue
        if artifact_path.suffix == ".json":
            try:
                payload = _parse_strict_json(
                    artifact_bytes,
                    label=f"required JSON artifact {artifact}",
                )
                if not isinstance(payload, dict):
                    raise ValueError("required JSON artifact must contain an object")
            except (UnicodeError, ValueError) as exc:
                issues.append(
                    ValidationIssue(
                        target=target,
                        path=str(artifact_path),
                        message=f"required JSON artifact is invalid: {exc}",
                    )
                )

    lane_artifact_path = report_dir / "lane_artifact.json"
    if lane_artifact_path.is_file():
        lane_label: str | None = None
        lane_artifact, lane_error = _read_required_json_object(
            target=target,
            path=lane_artifact_path,
            artifact_name="lane artifact",
        )
        if lane_error is not None:
            issues.append(lane_error)
        elif lane_artifact is not None:
            expected_lane_fields = {
                "assurance",
                "device",
                "execution_mode",
                "lane",
                "lane_artifact_label",
                "runtime_provenance",
            }
            if set(lane_artifact) != expected_lane_fields:
                issues.append(
                    ValidationIssue(
                        target=target,
                        path=str(lane_artifact_path),
                        message="lane artifact fields must match v1 exactly",
                    )
                )
            lane_label = (
                str(lane_artifact["lane_artifact_label"])
                if "lane_artifact_label" in lane_artifact
                else None
            )
        if lane_error is None and lane_label != expected.get("lane_artifact_label"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(lane_artifact_path),
                    message=(
                        "lane artifact label mismatch: "
                        f"expected {expected.get('lane_artifact_label')!r}, "
                        f"got {lane_label!r}"
                    ),
                )
            )

    summary_path = report_dir / "run_summary.txt"
    if summary_path.is_file():
        try:
            summary = _summary_fields(summary_path)
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(summary_path),
                    message=f"run summary is invalid: {exc}",
                )
            )
            summary = {}
        if summary.get("status") != "success":
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(summary_path),
                    message="run summary does not record status: success",
                )
            )
        if summary.get("lane_artifact_label") != expected.get("lane_artifact_label"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(summary_path),
                    message="run summary lane label does not match matrix",
                )
            )
        if (
            "training_binding.json" in artifacts
            and summary.get("training_binding_status") != "verified"
        ):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(summary_path),
                    message="run summary does not record verified training binding",
                )
            )

    run_command_path = report_dir / "run_command.txt"
    if (
        is_strict_lane
        and expected_quantized_backend is not None
        and run_command_path.is_file()
    ):
        try:
            run_commands = _command_fields(run_command_path)
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(run_command_path),
                    message=f"strict quantized run command is invalid: {exc}",
                )
            )
        else:
            wrapper_tokens = run_commands["wrapper"]
            for required_flag in (
                "--require-backend-inventory",
                "--require-runtime-quantization-proof",
            ):
                if wrapper_tokens.count(required_flag) != 1:
                    issues.append(
                        ValidationIssue(
                            target=target,
                            path=str(run_command_path),
                            message=(
                                "strict module-backed quantized run command is missing "
                                f"{required_flag}"
                            ),
                        )
                    )

    verify_path = report_dir / "verify.json"
    verify_payload: Any = None
    if verify_path.is_file():
        verify_payload, verify_error = _read_required_json_object(
            target=target,
            path=verify_path,
            artifact_name="verify artifact",
        )
        if verify_error is not None or verify_payload is None:
            if verify_error is not None:
                issues.append(verify_error)
            verify_payload = {}
        issues.extend(
            _validate_verification_output_shape(
                target=target,
                path=verify_path,
                payload=verify_payload,
            )
        )
        status = _verify_status(verify_payload)
        if status != expected.get("verify_status"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=(
                        "verify status mismatch: "
                        f"expected {expected.get('verify_status')!r}, got {status!r}"
                    ),
                )
            )

    report_path = report_dir / "evaluation.report.json"
    if (
        report_path.is_file()
        and verify_path.is_file()
        and isinstance(verification_profile, str)
    ):
        if acceptance_inputs is None:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(report_path),
                    message=(
                        "strict artifact validation requires an acceptance baseline, "
                        "policy pack, and runtime image digest"
                    ),
                )
            )
        elif not acceptance_inputs.baseline_report.is_file():
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(acceptance_inputs.baseline_report),
                    message="acceptance baseline report is missing",
                )
            )
        elif not acceptance_inputs.policy_pack.is_file():
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(acceptance_inputs.policy_pack),
                    message="acceptance policy pack is missing",
                )
            )
        else:
            try:
                verification_snapshots = _capture_verification_inputs(
                    report_path=report_path,
                    acceptance_inputs=acceptance_inputs,
                )
            except ValueError as exc:
                issues.append(
                    ValidationIssue(
                        target=target,
                        path=str(report_path),
                        message=f"strict verification inputs are invalid: {exc}",
                    )
                )
            else:
                issues.extend(
                    _validate_verification_receipt(
                        target=target,
                        verify_path=verify_path,
                        payload=verify_payload,
                        snapshots=verification_snapshots,
                        acceptance_inputs=acceptance_inputs,
                        verification_profile=verification_profile,
                    )
                )
                issues.extend(
                    _replay_strict_verification(
                        repo_root=repo_root,
                        target=target,
                        report_path=report_path,
                        verify_path=verify_path,
                        stored_payload=verify_payload,
                        acceptance_inputs=acceptance_inputs,
                        snapshots=verification_snapshots,
                        verification_profile=verification_profile,
                    )
                )

        runtime = _runtime_provenance(verify_payload)
        declared = runtime.get("declared_mode")
        if declared != expected.get("runtime_provenance_declared"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=(
                        "runtime provenance declared mode mismatch: "
                        f"expected {expected.get('runtime_provenance_declared')!r}, "
                        f"got {declared!r}"
                    ),
                )
            )
        if runtime.get("verified") != expected.get("runtime_provenance_verified"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=(
                        "runtime provenance verified flag mismatch: "
                        f"expected {expected.get('runtime_provenance_verified')!r}, "
                        f"got {runtime.get('verified')!r}"
                    ),
                )
            )
        if runtime.get("status") != expected.get("runtime_provenance_status"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=(
                        "runtime provenance status mismatch: "
                        f"expected {expected.get('runtime_provenance_status')!r}, "
                        f"got {runtime.get('status')!r}"
                    ),
                )
            )
        if runtime.get("expected_digest_matched") != expected.get(
            "runtime_expected_digest_matched"
        ):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=(
                        "runtime expected-digest match flag mismatch: "
                        f"expected {expected.get('runtime_expected_digest_matched')!r}, "
                        f"got {runtime.get('expected_digest_matched')!r}"
                    ),
                )
            )

    backend_inventory_path = report_dir / "backend_inventory.json"
    backend_inventory: dict[str, Any] | None = None
    if backend_inventory_path.is_file():
        backend_inventory, backend_error = _read_required_json_object(
            target=target,
            path=backend_inventory_path,
            artifact_name="backend inventory",
        )
        if backend_error is not None:
            issues.append(backend_error)
        elif backend_inventory is not None:
            issues.extend(
                _validate_backend_inventory(
                    target=target,
                    path=backend_inventory_path,
                    payload=backend_inventory,
                    expected_adapter=subject_adapter,
                    expected_backend=expected_quantized_backend,
                )
            )

    runtime_quantization_proof_path = report_dir / RUNTIME_QUANTIZATION_PROOF_FILENAME
    if is_strict_lane and expected_quantized_backend is not None:
        assert isinstance(subject_adapter, str)
        if runtime_quantization_proof_path.is_file():
            runtime_quantization_proof, runtime_proof_error = _read_strict_json_object(
                target=target,
                path=runtime_quantization_proof_path,
                artifact_name="runtime quantization proof",
            )
            if runtime_proof_error is not None:
                issues.append(runtime_proof_error)
            elif runtime_quantization_proof is not None:
                issues.extend(
                    _validate_runtime_quantization_proof(
                        target=target,
                        path=runtime_quantization_proof_path,
                        payload=runtime_quantization_proof,
                        expected_adapter=subject_adapter,
                        expected_backend=expected_quantized_backend,
                        backend_inventory=backend_inventory,
                    )
                )

    runtime_manifest_path = report_dir / "runtime.manifest.json"
    runtime_image = entry.get("runtime_image")
    if (
        runtime_manifest_path.is_file()
        and isinstance(runtime_image, dict)
        and runtime_image.get("declared_digest_source") == "runtime.manifest.json"
    ):
        runtime_manifest, runtime_error = _read_required_json_object(
            target=target,
            path=runtime_manifest_path,
            artifact_name="runtime manifest",
        )
        if runtime_error is not None:
            issues.append(runtime_error)
        elif runtime_manifest is not None:
            issues.extend(
                _validate_runtime_manifest(
                    target=target,
                    path=runtime_manifest_path,
                    payload=runtime_manifest,
                )
            )

    if "training_binding.json" in artifacts:
        issues.extend(
            _validate_training_binding(
                repo_root=repo_root,
                target=target,
                report_dir=report_dir,
                expected_training_profile=entry.get("training_profile"),
            )
        )
    if TRAINING_EVIDENCE_ARTIFACTS.issubset(artifact_names):
        issues.extend(
            _validate_training_evidence(
                repo_root=repo_root,
                target=target,
                report_dir=report_dir,
                expected_training_profile=entry.get("training_profile"),
                expected_training_scope=entry.get("training_scope"),
            )
        )

    return issues


def validate_matrix(
    *,
    repo_root: Path,
    matrix_path: Path,
    targets: set[str] | None = None,
    acceptance_inputs: AcceptanceInputs | None = None,
) -> tuple[list[str], list[ValidationIssue]]:
    payload = _read_source_matrix_snapshot(matrix_path)
    entries = _validate_source_matrix_shape(payload)
    matrix_targets = {entry["target"] for entry in entries}
    if targets is None and matrix_targets != SOURCE_MATRIX_TARGETS:
        missing = sorted(SOURCE_MATRIX_TARGETS - matrix_targets)
        extra = sorted(matrix_targets - SOURCE_MATRIX_TARGETS)
        raise ValueError(
            "source matrix target coverage must be exact: "
            f"missing={missing}, unexpected={extra}"
        )

    selected: list[str] = []
    issues: list[ValidationIssue] = []
    for entry in entries:
        target = str(entry.get("target", ""))
        if targets is not None and target not in targets:
            continue
        selected.append(target)
        issues.extend(
            validate_entry(
                repo_root,
                entry,
                acceptance_inputs=acceptance_inputs,
            )
        )

    if targets is not None:
        missing = sorted(targets - set(selected))
        for target in missing:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(matrix_path),
                    message="target is not present in source matrix",
                )
            )

    return selected, issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate generated integration run artifacts against "
            "examples/integrations/source_matrix.json."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing examples/integrations.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Path to source_matrix.json. Defaults under --repo-root.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional matrix targets to validate. Defaults to all entries.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        required=True,
        help="Independently supplied raw baseline report for strict replay.",
    )
    parser.add_argument(
        "--policy-pack",
        type=Path,
        required=True,
        help="Independently supplied policy pack for strict replay.",
    )
    parser.add_argument(
        "--expected-runtime-image-digest",
        required=True,
        help="Independently supplied sha256 runtime image digest for strict replay.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter containing InvarLock. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable validation output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    matrix_path = (
        args.matrix.resolve()
        if args.matrix is not None
        else (repo_root / DEFAULT_MATRIX).resolve()
    )
    targets = set(args.targets) if args.targets is not None else None
    selected, issues = validate_matrix(
        repo_root=repo_root,
        matrix_path=matrix_path,
        targets=targets,
        acceptance_inputs=AcceptanceInputs(
            baseline_report=args.baseline_report.resolve(),
            policy_pack=args.policy_pack.resolve(),
            expected_runtime_image_digest=args.expected_runtime_image_digest,
            python_bin=args.python_bin,
        ),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not issues,
                    "matrix": str(matrix_path),
                    "targets": selected,
                    "issues": [issue.as_dict() for issue in issues],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        if issues:
            print("Source matrix artifact validation failed:")
            for issue in issues:
                print(f"- {issue.target}: {issue.path}: {issue.message}")
        else:
            print("Source matrix artifact validation passed for " + ", ".join(selected))

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
