#!/usr/bin/env python3
"""Audit public evidence classification and verifier metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_EVIDENCE_ROOT = REPO_ROOT / "public_evidence"
META_FILENAME = "evidence.meta.json"
SCHEMA = "invarlock.public_evidence.meta.v1"

ALLOWED_CLASSES = {
    "contract_fixture",
    "strict_pass_fixture",
    "caught_regression_fixture",
    "policy_failure_fixture",
    "byoe_subject_fixture",
    "real_model_run",
    "real_guard_value_demo",
    "signed_real_model_pack",
    "runtime_backend_compat_sweep",
    "evidence_pack_queue_stress_resume",
    "fa2_fallback_compatibility",
    "larger_model_smoke_findings",
    "larger_model_queue_drain_findings",
}
REAL_CLASSES = {"real_model_run", "real_guard_value_demo", "signed_real_model_pack"}
NON_FIXTURE_CLASSES = REAL_CLASSES | {
    "runtime_backend_compat_sweep",
    "evidence_pack_queue_stress_resume",
    "fa2_fallback_compatibility",
    "larger_model_smoke_findings",
    "larger_model_queue_drain_findings",
}
RUNTIME_BACKEND_COMPAT_SCHEMA = "invarlock.runtime_backend_compat.cuda128.summary.v1"
RUNTIME_BACKEND_HASH_SCHEMA = (
    "invarlock.runtime_backend_compat.cuda128.hash_inventory.v1"
)
QUEUE_STRESS_SUMMARY_SCHEMA = "invarlock.evidence_pack_queue_stress_resume.summary.v1"
QUEUE_STRESS_HASH_SCHEMA = (
    "invarlock.evidence_pack_queue_stress_resume.hash_inventory.v1"
)
FA2_FALLBACK_SUMMARY_SCHEMA = "invarlock.fa2_fallback_compatibility.summary.v1"
FA2_FALLBACK_HASH_SCHEMA = "invarlock.fa2_fallback_compatibility.hash_inventory.v1"
LARGER_MODEL_SMOKE_FINDINGS_SCHEMA = "invarlock.larger_model_smoke_findings.summary.v1"
LARGER_MODEL_SMOKE_HASH_SCHEMA = (
    "invarlock.larger_model_smoke_findings.hash_inventory.v1"
)
LARGER_MODEL_QUEUE_DRAIN_FINDINGS_SCHEMA = (
    "invarlock.larger_model_queue_drain_findings.summary.v1"
)
LARGER_MODEL_QUEUE_DRAIN_ADDENDUM_SCHEMA = (
    "invarlock.larger_model_queue_drain_findings.late_clean_addendum.v1"
)
LARGER_MODEL_QUEUE_DRAIN_MODERN_ADDENDUM_SCHEMA = (
    "invarlock.larger_model_queue_drain_findings.modern_followon_addendum.v1"
)
LARGER_MODEL_QUEUE_DRAIN_HASH_SCHEMA = (
    "invarlock.larger_model_queue_drain_findings.hash_inventory.v1"
)
RUNTIME_BACKEND_FAMILIES = {
    "cuda-bnb": ("hf_bnb",),
    "cuda-compressed-tensors": ("hf_ct",),
    "cuda-gptqmodel": ("hf_awq", "hf_gptq"),
    "cuda-hqq": ("hf_hqq",),
    "cuda-quanto": ("hf_quanto",),
    "cuda-torchao": ("hf_torchao",),
}
RUNTIME_BACKEND_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PUBLISHED_BASIS_MULTIMODAL_MIN_FINAL_ACCURACY = 0.10
PUBLISHED_BASIS_MULTIMODAL_MIN_FINAL_EXAMPLES = 200
PUBLISHED_BASIS_MULTIMODAL_MIN_ANSWER_SHAPE_RATE = 0.95
PUBLISHED_BASIS_MULTIMODAL_MAX_ANSWER_WORDS = 12
PUBLISHED_BASIS_MULTIMODAL_MAX_ANSWER_CHARS = 80
PUBLIC_TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}

PRIVATE_EXECUTION_PATTERNS = (
    (
        "root_ssh_target",
        re.compile(r"\broot@[A-Za-z0-9._-]+\b"),
        "replace root SSH targets with a generic CUDA validation host label",
    ),
    (
        "private_ip_address",
        re.compile(
            r"(?<![A-Za-z0-9])"
            r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?![A-Za-z0-9])"
        ),
        "replace private host IP addresses with a generic host label",
    ),
    (
        "absolute_root_path",
        re.compile(r"(?<![A-Za-z0-9._-])/root(?:/[^\s\"'`,)}\]]*)?"),
        "replace absolute root paths with generic validation-root placeholders",
    ),
    (
        "private_tmp_path",
        re.compile(r"(?<![A-Za-z0-9._-])/private/tmp(?:/[^\s\"'`,)}\]]*)?"),
        "replace private temporary paths with generic local-run placeholders",
    ),
    (
        "macos_var_folder_path",
        re.compile(r"(?<![A-Za-z0-9._-])/var/folders(?:/[^\s\"'`,)}\]]*)?"),
        "replace macOS temporary paths with generic local-temp placeholders",
    ),
    (
        "home_directory_path",
        re.compile(r"(?<![A-Za-z0-9._-])/home/[A-Za-z0-9._-]+(?:/[^\s\"'`,)}\]]*)?"),
        "replace home-directory paths with generic validation-root placeholders",
    ),
)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"{path}: unable to read JSON: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"{path}: invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path}: expected JSON object"
    return payload, None


def _is_inside_special_dir(path: Path, root: Path) -> bool:
    parts = set(path.relative_to(root).parts)
    return bool(parts & {"artifact_package", "evidence_pack"})


def _artifact_dirs(root: Path) -> set[Path]:
    dirs: set[Path] = set()
    for metadata in root.rglob(META_FILENAME):
        if metadata.is_file() and not _is_inside_special_dir(metadata, root):
            dirs.add(metadata.parent)
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if _is_inside_special_dir(path, root):
            continue
        if path.name in {
            "evaluation.report.json",
            "runtime.manifest.json",
            "checkpoint_refs.json",
            "evidence_pack_recipe.json",
        }:
            dirs.add(path.parent)
    for manifest in root.rglob("evidence_pack/manifest.json"):
        dirs.add(manifest.parent.parent)
    return dirs


def _relative(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _check_public_evidence_privacy(errors: list[str], root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in PUBLIC_TEXT_SUFFIXES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{_relative(path)}: unable to scan public text: {exc}")
            continue
        for line_number, line in enumerate(lines, start=1):
            for name, pattern, message in PRIVATE_EXECUTION_PATTERNS:
                if pattern.search(line):
                    errors.append(f"{_relative(path)}:{line_number}: {name}: {message}")


def _require_path(
    errors: list[str],
    base: Path,
    artifact_paths: dict[str, Any],
    key: str,
    *,
    directory: bool = False,
) -> Path | None:
    raw = artifact_paths.get(key)
    if not isinstance(raw, str) or not raw.strip():
        errors.append(f"{_relative(base)}: artifact_paths.{key} is required")
        return None
    path = base / raw
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        errors.append(f"{_relative(base)}: missing {kind} {raw!r}")
        return None
    return path


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _report_primary_metric(report: dict[str, Any]) -> dict[str, Any]:
    primary = report.get("primary_metric")
    if isinstance(primary, dict):
        return primary
    metrics = report.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("primary_metric"), dict):
        return metrics["primary_metric"]
    return {}


def _classification_final_counts(
    report: dict[str, Any],
) -> tuple[int | None, int | None]:
    metrics = report.get("metrics")
    classification = (
        metrics.get("classification") if isinstance(metrics, dict) else None
    )
    if not isinstance(classification, dict):
        classification = report.get("classification")
    if not isinstance(classification, dict):
        return None, None
    final = classification.get("final")
    if isinstance(final, dict):
        return _as_int(final.get("correct_total")), _as_int(final.get("total"))
    return None, None


def _is_direct_published_basis_artifact(artifact_dir: Path, root: Path) -> bool:
    try:
        parts = artifact_dir.relative_to(root).parts
    except ValueError:
        return False
    return len(parts) == 2 and parts[0] == "published_basis"


def _is_vision_text_accuracy_report(report: dict[str, Any]) -> bool:
    dataset = report.get("dataset")
    provider = dataset.get("provider") if isinstance(dataset, dict) else None
    return (
        provider == "vision_text"
        and _report_primary_metric(report).get("kind") == "accuracy"
    )


_ANSWER_FIELD_RE = re.compile(
    r'"answer"\s*:\s*"(?P<answer>(?:\\.|[^"\\])*)"', re.DOTALL
)


def _extract_answer_shape_text(prediction: Any) -> str:
    text = str(prediction or "").strip()
    if not text:
        return ""
    for candidate in (text, text.strip("` \n")):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            answer = parsed.get("answer")
            if isinstance(answer, str):
                return " ".join(answer.split())
    match = _ANSWER_FIELD_RE.search(text)
    if match:
        try:
            return " ".join(json.loads(f'"{match.group("answer")}"').split())
        except json.JSONDecodeError:
            return " ".join(match.group("answer").split())
    return " ".join(text.split())


def _answer_shape_ok(prediction: Any) -> bool:
    answer = _extract_answer_shape_text(prediction)
    if not answer:
        return False
    return (
        len(answer) <= PUBLISHED_BASIS_MULTIMODAL_MAX_ANSWER_CHARS
        and len(answer.split()) <= PUBLISHED_BASIS_MULTIMODAL_MAX_ANSWER_WORDS
    )


def _embedded_answer_shape_rate(report: dict[str, Any]) -> tuple[int, int] | None:
    eval_windows = report.get("eval_windows")
    final_window = eval_windows.get("final") if isinstance(eval_windows, dict) else None
    records = final_window.get("records") if isinstance(final_window, dict) else None
    if not isinstance(records, list) or not records:
        return None
    total = 0
    ok = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if "prediction" not in record:
            continue
        total += 1
        ok += int(_answer_shape_ok(record.get("prediction")))
    if total <= 0:
        return None
    return ok, total


def _check_published_basis_multimodal_quality(
    errors: list[str],
    base: Path,
    report_path: Path,
) -> None:
    report, error = _load_json(report_path)
    if error:
        errors.append(error)
        return
    assert report is not None
    if not _is_vision_text_accuracy_report(report):
        return

    primary = _report_primary_metric(report)
    final_accuracy = _as_finite_float(primary.get("final"))
    n_final = _as_int(primary.get("n_final"))
    correct_total, total = _classification_final_counts(report)
    if n_final is None:
        n_final = total
    if final_accuracy is None and correct_total is not None and total:
        final_accuracy = correct_total / total

    if primary.get("counts_source") != "measured" or primary.get("estimated") is True:
        errors.append(
            f"{_relative(base)}: published image-text basis requires measured accuracy counts"
        )
    if n_final is None or n_final < PUBLISHED_BASIS_MULTIMODAL_MIN_FINAL_EXAMPLES:
        errors.append(
            f"{_relative(base)}: published image-text basis requires at least "
            f"{PUBLISHED_BASIS_MULTIMODAL_MIN_FINAL_EXAMPLES} final examples"
        )
    if (
        final_accuracy is None
        or final_accuracy < PUBLISHED_BASIS_MULTIMODAL_MIN_FINAL_ACCURACY
    ):
        observed = "missing" if final_accuracy is None else f"{final_accuracy:.4f}"
        errors.append(
            f"{_relative(base)}: published image-text basis final accuracy "
            f"{observed} is below "
            f"{PUBLISHED_BASIS_MULTIMODAL_MIN_FINAL_ACCURACY:.2f}"
        )

    shape_counts = _embedded_answer_shape_rate(report)
    if shape_counts is not None:
        ok, total_shape = shape_counts
        rate = ok / total_shape
        if rate < PUBLISHED_BASIS_MULTIMODAL_MIN_ANSWER_SHAPE_RATE:
            errors.append(
                f"{_relative(base)}: published image-text basis answer-shape rate "
                f"{rate:.4f} is below "
                f"{PUBLISHED_BASIS_MULTIMODAL_MIN_ANSWER_SHAPE_RATE:.2f}"
            )


def _check_signed_pack(
    errors: list[str],
    base: Path,
    metadata: dict[str, Any],
    artifact_paths: dict[str, Any],
) -> None:
    pack_dir = _require_path(
        errors,
        base,
        artifact_paths,
        "evidence_pack",
        directory=True,
    )
    expected = metadata.get("expected_fingerprint")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        errors.append(
            f"{_relative(base)}: signed evidence pack requires expected_fingerprint"
        )
        return
    if pack_dir is None:
        return
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.is_file():
        errors.append(f"{_relative(pack_dir)}: missing manifest.json")
        return
    manifest, error = _load_json(manifest_path)
    if error:
        errors.append(error)
        return
    signer = manifest.get("signing_key_fingerprint") if manifest else None
    if signer != expected:
        errors.append(
            f"{_relative(base)}: expected_fingerprint does not match pack signer"
        )
    commands = metadata.get("verifier_commands")
    command_text = "\n".join(commands) if isinstance(commands, list) else ""
    if "--expected-fingerprint" not in command_text:
        errors.append(
            f"{_relative(base)}: signed pack verifier command must pin fingerprint"
        )


def _check_guard_value_demo(
    errors: list[str],
    base: Path,
    artifact_paths: dict[str, Any],
) -> None:
    manifest_path = _require_path(errors, base, artifact_paths, "guard_value_manifest")
    _require_path(errors, base, artifact_paths, "guard_value_summary")
    _require_path(errors, base, artifact_paths, "artifact_package", directory=True)
    if manifest_path is None:
        return
    manifest, error = _load_json(manifest_path)
    if error:
        errors.append(error)
        return
    assert manifest is not None
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append(f"{_relative(manifest_path)}: files must be a non-empty list")
        return
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append(f"{_relative(manifest_path)}: files[{index}] must be object")
            continue
        rel_path = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(f"{_relative(manifest_path)}: files[{index}].path required")
            continue
        path = base / rel_path
        if not path.is_file():
            errors.append(f"{_relative(base)}: manifest file missing {rel_path!r}")
            continue
        content = path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"{_relative(base)}: manifest hash mismatch for {rel_path!r}")
        if len(content) != expected_size:
            errors.append(f"{_relative(base)}: manifest size mismatch for {rel_path!r}")


def _check_runtime_backend_hash_inventory(
    errors: list[str],
    base: Path,
    inventory_path: Path,
) -> None:
    inventory, error = _load_json(inventory_path)
    if error:
        errors.append(error)
        return
    assert inventory is not None
    if inventory.get("schema") != RUNTIME_BACKEND_HASH_SCHEMA:
        errors.append(
            f"{_relative(inventory_path)}: schema must be {RUNTIME_BACKEND_HASH_SCHEMA}"
        )
    if inventory.get("status") != "completed":
        errors.append(f"{_relative(inventory_path)}: status must be completed")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{_relative(inventory_path)}: artifacts must be non-empty")
        return
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}] must be object"
            )
            continue
        rel_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path required"
            )
            continue
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path must be relative"
            )
            continue
        path = base / rel_path
        if not path.is_file():
            errors.append(
                f"{_relative(base)}: hash inventory file missing {rel_path!r}"
            )
            continue
        content = path.read_bytes()
        actual_sha = "sha256:" + hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"{_relative(base)}: hash inventory mismatch for {rel_path!r}"
            )
        if len(content) != expected_bytes:
            errors.append(
                f"{_relative(base)}: hash inventory byte mismatch for {rel_path!r}"
            )


def _check_runtime_backend_compat_sweep(
    errors: list[str],
    base: Path,
    artifact_paths: dict[str, Any],
) -> None:
    summary_path = _require_path(errors, base, artifact_paths, "compatibility_summary")
    inventory_path = _require_path(errors, base, artifact_paths, "hash_inventory")
    if inventory_path is not None:
        _check_runtime_backend_hash_inventory(errors, base, inventory_path)
    if summary_path is None:
        return
    summary, error = _load_json(summary_path)
    if error:
        errors.append(error)
        return
    assert summary is not None
    if summary.get("schema") != RUNTIME_BACKEND_COMPAT_SCHEMA:
        errors.append(
            f"{_relative(summary_path)}: schema must be {RUNTIME_BACKEND_COMPAT_SCHEMA}"
        )
    if summary.get("status") != "completed":
        errors.append(f"{_relative(summary_path)}: status must be completed")
    if summary.get("validation_environment") != "CUDA-capable validation host":
        errors.append(
            f"{_relative(summary_path)}: validation_environment must be generic"
        )
    if summary.get("raw_logs_published") is not False:
        errors.append(f"{_relative(summary_path)}: raw_logs_published must be false")
    if summary.get("weights_vendored") is not False:
        errors.append(f"{_relative(summary_path)}: weights_vendored must be false")

    families = summary.get("families")
    if not isinstance(families, list) or not families:
        errors.append(f"{_relative(summary_path)}: families must be non-empty")
        return
    observed: set[str] = set()
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            errors.append(
                f"{_relative(summary_path)}: families[{index}] must be object"
            )
            continue
        family_name = family.get("family")
        if family_name not in RUNTIME_BACKEND_FAMILIES:
            errors.append(f"{_relative(summary_path)}: unknown family {family_name!r}")
            continue
        observed.add(str(family_name))
        expected_adapters = list(RUNTIME_BACKEND_FAMILIES[str(family_name)])
        if family.get("adapter_smoke") != expected_adapters:
            errors.append(
                f"{_relative(summary_path)}: {family_name} adapter_smoke mismatch"
            )
        if family.get("build_rc") != 0 or family.get("smoke_rc") != 0:
            errors.append(f"{_relative(summary_path)}: {family_name} rc must be zero")
        if family.get("gpu_required") is not True:
            errors.append(f"{_relative(summary_path)}: {family_name} must require GPU")
        requirements_lock = family.get("requirements_lock")
        if not isinstance(requirements_lock, str) or not requirements_lock:
            errors.append(
                f"{_relative(summary_path)}: {family_name} requirements_lock required"
            )
        elif (
            requirements_lock.startswith("/")
            or not (REPO_ROOT / requirements_lock).is_file()
        ):
            errors.append(
                f"{_relative(summary_path)}: {family_name} requirements_lock invalid"
            )
        for command_key in ("build_command", "smoke_command"):
            command = family.get(command_key)
            if not isinstance(command, str) or command.startswith("/"):
                errors.append(
                    f"{_relative(summary_path)}: {family_name} {command_key} invalid"
                )
        image_id = family.get("image_id")
        if not isinstance(image_id, str) or not RUNTIME_BACKEND_IMAGE_ID_RE.match(
            image_id
        ):
            errors.append(f"{_relative(summary_path)}: {family_name} image_id invalid")
        image_size = family.get("image_size_bytes")
        if not isinstance(image_size, int) or image_size <= 0:
            errors.append(
                f"{_relative(summary_path)}: {family_name} image_size_bytes invalid"
            )
        smoke_result = family.get("smoke_result")
        if not isinstance(smoke_result, str) or not smoke_result.startswith(
            "quant runtime image imports ok:"
        ):
            errors.append(
                f"{_relative(summary_path)}: {family_name} smoke_result invalid"
            )
    if observed != set(RUNTIME_BACKEND_FAMILIES):
        missing = sorted(set(RUNTIME_BACKEND_FAMILIES) - observed)
        extra = sorted(observed - set(RUNTIME_BACKEND_FAMILIES))
        errors.append(
            f"{_relative(summary_path)}: family coverage mismatch "
            f"missing={missing} extra={extra}"
        )


def _check_queue_stress_hash_inventory(
    errors: list[str],
    base: Path,
    inventory_path: Path,
) -> None:
    inventory, error = _load_json(inventory_path)
    if error:
        errors.append(error)
        return
    assert inventory is not None
    if inventory.get("schema") != QUEUE_STRESS_HASH_SCHEMA:
        errors.append(
            f"{_relative(inventory_path)}: schema must be {QUEUE_STRESS_HASH_SCHEMA}"
        )
    if inventory.get("status") != "completed":
        errors.append(f"{_relative(inventory_path)}: status must be completed")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{_relative(inventory_path)}: artifacts must be non-empty")
        return
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}] must be object"
            )
            continue
        rel_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path required"
            )
            continue
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path must be relative"
            )
            continue
        path = base / rel_path
        if not path.is_file():
            errors.append(
                f"{_relative(base)}: hash inventory file missing {rel_path!r}"
            )
            continue
        content = path.read_bytes()
        actual_sha = "sha256:" + hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"{_relative(base)}: hash inventory mismatch for {rel_path!r}"
            )
        if len(content) != expected_bytes:
            errors.append(
                f"{_relative(base)}: hash inventory byte mismatch for {rel_path!r}"
            )


def _check_evidence_pack_queue_stress_resume(
    errors: list[str],
    base: Path,
    artifact_paths: dict[str, Any],
) -> None:
    summary_path = _require_path(errors, base, artifact_paths, "stress_summary")
    inventory_path = _require_path(errors, base, artifact_paths, "hash_inventory")
    if inventory_path is not None:
        _check_queue_stress_hash_inventory(errors, base, inventory_path)
    if summary_path is None:
        return
    summary, error = _load_json(summary_path)
    if error:
        errors.append(error)
        return
    assert summary is not None
    if summary.get("schema") != QUEUE_STRESS_SUMMARY_SCHEMA:
        errors.append(
            f"{_relative(summary_path)}: schema must be {QUEUE_STRESS_SUMMARY_SCHEMA}"
        )
    if summary.get("status") != "completed":
        errors.append(f"{_relative(summary_path)}: status must be completed")
    if summary.get("validation_environment") != "CUDA-capable validation host":
        errors.append(
            f"{_relative(summary_path)}: validation_environment must be generic"
        )
    if summary.get("raw_logs_published") is not False:
        errors.append(f"{_relative(summary_path)}: raw_logs_published must be false")
    if summary.get("weights_vendored") is not False:
        errors.append(f"{_relative(summary_path)}: weights_vendored must be false")

    suites = summary.get("suites")
    if not isinstance(suites, list) or len(suites) < 2:
        errors.append(f"{_relative(summary_path)}: suites must include both checks")
        return
    observed = {suite.get("name"): suite for suite in suites if isinstance(suite, dict)}
    expected = {
        "queue_manager_shell": 74,
        "queue_state_python": 3,
    }
    for name, expected_passed in expected.items():
        suite = observed.get(name)
        if not isinstance(suite, dict):
            errors.append(f"{_relative(summary_path)}: missing suite {name}")
            continue
        if suite.get("rc") != 0 or suite.get("tests_failed") != 0:
            errors.append(f"{_relative(summary_path)}: {name} must pass cleanly")
        if suite.get("tests_passed") != expected_passed:
            errors.append(
                f"{_relative(summary_path)}: {name} tests_passed must be {expected_passed}"
            )
        command = suite.get("command")
        if not isinstance(command, str) or command.startswith("/"):
            errors.append(f"{_relative(summary_path)}: {name} command invalid")
        surface = suite.get("coverage_surface")
        if not isinstance(surface, list) or not surface:
            errors.append(
                f"{_relative(summary_path)}: {name} coverage_surface required"
            )


def _check_fa2_fallback_hash_inventory(
    errors: list[str],
    base: Path,
    inventory_path: Path,
) -> None:
    inventory, error = _load_json(inventory_path)
    if error:
        errors.append(error)
        return
    assert inventory is not None
    if inventory.get("schema") != FA2_FALLBACK_HASH_SCHEMA:
        errors.append(
            f"{_relative(inventory_path)}: schema must be {FA2_FALLBACK_HASH_SCHEMA}"
        )
    if inventory.get("status") != "completed":
        errors.append(f"{_relative(inventory_path)}: status must be completed")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{_relative(inventory_path)}: artifacts must be non-empty")
        return
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}] must be object"
            )
            continue
        rel_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path required"
            )
            continue
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path must be relative"
            )
            continue
        path = base / rel_path
        if not path.is_file():
            errors.append(
                f"{_relative(base)}: hash inventory file missing {rel_path!r}"
            )
            continue
        content = path.read_bytes()
        actual_sha = "sha256:" + hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"{_relative(base)}: hash inventory mismatch for {rel_path!r}"
            )
        if len(content) != expected_bytes:
            errors.append(
                f"{_relative(base)}: hash inventory byte mismatch for {rel_path!r}"
            )


def _check_fa2_fallback_compatibility(
    errors: list[str],
    base: Path,
    artifact_paths: dict[str, Any],
) -> None:
    summary_path = _require_path(errors, base, artifact_paths, "compatibility_summary")
    inventory_path = _require_path(errors, base, artifact_paths, "hash_inventory")
    if inventory_path is not None:
        _check_fa2_fallback_hash_inventory(errors, base, inventory_path)
    if summary_path is None:
        return
    summary, error = _load_json(summary_path)
    if error:
        errors.append(error)
        return
    assert summary is not None
    if summary.get("schema") != FA2_FALLBACK_SUMMARY_SCHEMA:
        errors.append(
            f"{_relative(summary_path)}: schema must be {FA2_FALLBACK_SUMMARY_SCHEMA}"
        )
    if summary.get("status") != "completed":
        errors.append(f"{_relative(summary_path)}: status must be completed")
    if summary.get("validation_environment") != "CUDA-capable validation host":
        errors.append(
            f"{_relative(summary_path)}: validation_environment must be generic"
        )
    if summary.get("raw_logs_published") is not False:
        errors.append(f"{_relative(summary_path)}: raw_logs_published must be false")
    if summary.get("weights_vendored") is not False:
        errors.append(f"{_relative(summary_path)}: weights_vendored must be false")
    if summary.get("fa2_success_claimed") is not False:
        errors.append(f"{_relative(summary_path)}: fa2_success_claimed must be false")

    probe = summary.get("cuda_probe")
    if not isinstance(probe, dict):
        errors.append(f"{_relative(summary_path)}: cuda_probe must be object")
    else:
        if probe.get("rc") != 0:
            errors.append(f"{_relative(summary_path)}: cuda_probe rc must be zero")
        if probe.get("torch_cuda_available") is not True:
            errors.append(f"{_relative(summary_path)}: CUDA must be available")
        if (
            not isinstance(probe.get("torch_cuda_device_count"), int)
            or probe.get("torch_cuda_device_count") < 1
        ):
            errors.append(f"{_relative(summary_path)}: CUDA device count invalid")
        if probe.get("flash_attn_importable") is not False:
            errors.append(
                f"{_relative(summary_path)}: flash_attn_importable must be false"
            )
        if probe.get("transformers_flash_attn_2_available") is not False:
            errors.append(
                f"{_relative(summary_path)}: transformers FA2 availability must be false"
            )
        command = probe.get("command")
        if not isinstance(command, str) or command.startswith("/"):
            errors.append(f"{_relative(summary_path)}: cuda_probe command invalid")

    checks = summary.get("checks")
    if not isinstance(checks, list) or len(checks) < 2:
        errors.append(f"{_relative(summary_path)}: checks must include both checks")
        return
    observed = {check.get("name"): check for check in checks if isinstance(check, dict)}
    expected = {
        "flash_attn_dependency_fallbacks": 3,
        "flash_attention_config_fallback": 1,
    }
    for name, expected_passed in expected.items():
        check = observed.get(name)
        if not isinstance(check, dict):
            errors.append(f"{_relative(summary_path)}: missing check {name}")
            continue
        if check.get("rc") != 0 or check.get("tests_failed") != 0:
            errors.append(f"{_relative(summary_path)}: {name} must pass cleanly")
        if check.get("tests_passed") != expected_passed:
            errors.append(
                f"{_relative(summary_path)}: {name} tests_passed must be {expected_passed}"
            )
        command = check.get("command")
        if not isinstance(command, str) or command.startswith("/"):
            errors.append(f"{_relative(summary_path)}: {name} command invalid")
        surface = check.get("coverage_surface")
        if not isinstance(surface, list) or not surface:
            errors.append(
                f"{_relative(summary_path)}: {name} coverage_surface required"
            )


def _check_larger_model_smoke_hash_inventory(
    errors: list[str],
    base: Path,
    inventory_path: Path,
    *,
    expected_schema: str = LARGER_MODEL_SMOKE_HASH_SCHEMA,
) -> None:
    inventory, error = _load_json(inventory_path)
    if error:
        errors.append(error)
        return
    assert inventory is not None
    if inventory.get("schema") != expected_schema:
        errors.append(f"{_relative(inventory_path)}: schema must be {expected_schema}")
    if inventory.get("status") != "completed":
        errors.append(f"{_relative(inventory_path)}: status must be completed")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{_relative(inventory_path)}: artifacts must be non-empty")
        return
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}] must be object"
            )
            continue
        rel_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path required"
            )
            continue
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path must be relative"
            )
            continue
        path = base / rel_path
        if not path.is_file():
            errors.append(
                f"{_relative(base)}: hash inventory file missing {rel_path!r}"
            )
            continue
        content = path.read_bytes()
        actual_sha = "sha256:" + hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"{_relative(base)}: hash inventory mismatch for {rel_path!r}"
            )
        if len(content) != expected_bytes:
            errors.append(
                f"{_relative(base)}: hash inventory byte mismatch for {rel_path!r}"
            )


def _check_larger_model_smoke_findings(
    errors: list[str],
    base: Path,
    artifact_paths: dict[str, Any],
) -> None:
    summary_path = _require_path(errors, base, artifact_paths, "findings_summary")
    inventory_path = _require_path(errors, base, artifact_paths, "hash_inventory")
    if inventory_path is not None:
        _check_larger_model_smoke_hash_inventory(errors, base, inventory_path)
    if summary_path is None:
        return
    summary, error = _load_json(summary_path)
    if error:
        errors.append(error)
        return
    assert summary is not None
    if summary.get("schema") != LARGER_MODEL_SMOKE_FINDINGS_SCHEMA:
        errors.append(
            f"{_relative(summary_path)}: schema must be "
            f"{LARGER_MODEL_SMOKE_FINDINGS_SCHEMA}"
        )
    if summary.get("status") != "completed":
        errors.append(f"{_relative(summary_path)}: status must be completed")
    if summary.get("validation_environment") != "CUDA-capable validation host":
        errors.append(
            f"{_relative(summary_path)}: validation_environment must be generic"
        )
    if summary.get("raw_logs_published") is not False:
        errors.append(f"{_relative(summary_path)}: raw_logs_published must be false")
    if summary.get("weights_vendored") is not False:
        errors.append(f"{_relative(summary_path)}: weights_vendored must be false")
    if summary.get("support_matrix_change_claimed") is not False:
        errors.append(
            f"{_relative(summary_path)}: support_matrix_change_claimed must be false"
        )
    if summary.get("suite") != "model-catalog-gpu":
        errors.append(f"{_relative(summary_path)}: suite must be model-catalog-gpu")
    if summary.get("execution_mode") != "container":
        errors.append(f"{_relative(summary_path)}: execution_mode must be container")

    clean_lanes = summary.get("clean_lanes")
    failed_findings = summary.get("failed_findings")
    duplicates = summary.get("duplicate_clean_runs")
    counts = summary.get("counts")
    if not isinstance(clean_lanes, list) or not clean_lanes:
        errors.append(f"{_relative(summary_path)}: clean_lanes must be non-empty")
        clean_lanes = []
    if not isinstance(failed_findings, list) or not failed_findings:
        errors.append(f"{_relative(summary_path)}: failed_findings must be non-empty")
        failed_findings = []
    if not isinstance(duplicates, list):
        errors.append(f"{_relative(summary_path)}: duplicate_clean_runs must be a list")
        duplicates = []
    if not isinstance(counts, dict):
        errors.append(f"{_relative(summary_path)}: counts must be object")
        counts = {}

    seen_clean: set[str] = set()
    duplicate_extra_runs = 0
    for index, lane in enumerate(clean_lanes):
        if not isinstance(lane, dict):
            errors.append(
                f"{_relative(summary_path)}: clean_lanes[{index}] must be object"
            )
            continue
        slug = lane.get("slug")
        if not isinstance(slug, str) or not slug:
            errors.append(
                f"{_relative(summary_path)}: clean_lanes[{index}].slug required"
            )
        elif slug in seen_clean:
            errors.append(f"{_relative(summary_path)}: duplicate clean lane {slug!r}")
        else:
            seen_clean.add(slug)
        model_id = lane.get("model_id")
        preset = lane.get("preset")
        if not isinstance(model_id, str) or not model_id:
            errors.append(
                f"{_relative(summary_path)}: clean_lanes[{index}].model_id required"
            )
        if (
            not isinstance(preset, str)
            or preset.startswith("/")
            or not (REPO_ROOT / preset).is_file()
        ):
            errors.append(
                f"{_relative(summary_path)}: clean_lanes[{index}].preset invalid"
            )
        if lane.get("rc") != 0:
            errors.append(f"{_relative(summary_path)}: {slug} rc must be zero")
        if lane.get("evaluate_exit") != 0 or lane.get("verify_exit") != 0:
            errors.append(
                f"{_relative(summary_path)}: {slug} evaluate/verify exits must be zero"
            )
        if lane.get("report_materialized") is not True:
            errors.append(f"{_relative(summary_path)}: {slug} report must materialize")
        if lane.get("verify_materialized") is not True:
            errors.append(f"{_relative(summary_path)}: {slug} verify must materialize")
        if lane.get("status") != "ok":
            errors.append(f"{_relative(summary_path)}: {slug} status must be ok")

    for index, duplicate in enumerate(duplicates):
        if not isinstance(duplicate, dict):
            errors.append(
                f"{_relative(summary_path)}: duplicate_clean_runs[{index}] must be object"
            )
            continue
        slug = duplicate.get("slug")
        extra = duplicate.get("additional_clean_runs")
        if slug not in seen_clean:
            errors.append(
                f"{_relative(summary_path)}: duplicate_clean_runs[{index}].slug unknown"
            )
        if not isinstance(extra, int) or extra <= 0:
            errors.append(
                f"{_relative(summary_path)}: duplicate_clean_runs[{index}] "
                "additional_clean_runs invalid"
            )
        else:
            duplicate_extra_runs += extra

    unique_failed: set[str] = set()
    failed_attempts = 0
    pre_verification_failures = 0
    for index, finding in enumerate(failed_findings):
        if not isinstance(finding, dict):
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}] must be object"
            )
            continue
        slug = finding.get("slug")
        if not isinstance(slug, str) or not slug:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].slug required"
            )
        else:
            unique_failed.add(slug)
        preset = finding.get("preset")
        if (
            not isinstance(preset, str)
            or preset.startswith("/")
            or not (REPO_ROOT / preset).is_file()
        ):
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].preset invalid"
            )
        attempts = finding.get("attempts")
        if not isinstance(attempts, int) or attempts <= 0:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].attempts invalid"
            )
            attempts = 0
        failed_attempts += attempts
        if finding.get("status") != "evaluate_failed_before_report":
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].status invalid"
            )
        if finding.get("classification") != "pre_verification_evaluate_failure":
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].classification invalid"
            )
        if finding.get("evaluate_exit") != 1 or finding.get("verify_exit") is not None:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}] exit fields invalid"
            )
        if finding.get("report_materialized") is not False:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}] report must be false"
            )
        if finding.get("verify_materialized") is not False:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}] verify must be false"
            )
        pre_verification_failures += attempts

    expected_counts = {
        "unique_clean_lanes": len(seen_clean),
        "unique_failed_lanes": len(unique_failed),
        "clean_runs": len(seen_clean) + duplicate_extra_runs,
        "failed_runs": failed_attempts,
        "pre_verification_failures": pre_verification_failures,
        "report_materialized_clean": len(seen_clean) + duplicate_extra_runs,
        "verify_materialized_clean": len(seen_clean) + duplicate_extra_runs,
    }
    expected_counts["completed_runs"] = (
        expected_counts["clean_runs"] + expected_counts["failed_runs"]
    )
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            errors.append(f"{_relative(summary_path)}: counts.{key} must be {expected}")


QUEUE_DRAIN_FAILURE_CLASSIFICATIONS = {
    "initial_attempt_failed_later_clean",
    "pre_verification_evaluate_failure",
    "grouped_execution_cuda_failure_later_clean",
}
QUEUE_DRAIN_FAILURE_STATUSES = {
    "evaluate_failed_before_verifier",
    "cuda_launch_failure_before_verifier",
}


def _check_larger_model_queue_drain_findings(
    errors: list[str],
    base: Path,
    artifact_paths: dict[str, Any],
) -> None:
    summary_path = _require_path(errors, base, artifact_paths, "findings_summary")
    addendum_path = _require_path(errors, base, artifact_paths, "late_clean_addendum")
    modern_addendum_path = _require_path(
        errors, base, artifact_paths, "modern_followon_addendum"
    )
    inventory_path = _require_path(errors, base, artifact_paths, "hash_inventory")
    if inventory_path is not None:
        _check_larger_model_smoke_hash_inventory(
            errors,
            base,
            inventory_path,
            expected_schema=LARGER_MODEL_QUEUE_DRAIN_HASH_SCHEMA,
        )
    if summary_path is None:
        return
    if addendum_path is not None:
        _check_larger_model_queue_drain_addendum(errors, addendum_path)
    if modern_addendum_path is not None:
        _check_larger_model_queue_drain_modern_addendum(errors, modern_addendum_path)
    summary, error = _load_json(summary_path)
    if error:
        errors.append(error)
        return
    assert summary is not None
    if summary.get("schema") != LARGER_MODEL_QUEUE_DRAIN_FINDINGS_SCHEMA:
        errors.append(
            f"{_relative(summary_path)}: schema must be "
            f"{LARGER_MODEL_QUEUE_DRAIN_FINDINGS_SCHEMA}"
        )
    if summary.get("status") != "completed":
        errors.append(f"{_relative(summary_path)}: status must be completed")
    if summary.get("validation_environment") != "CUDA-capable validation host":
        errors.append(
            f"{_relative(summary_path)}: validation_environment must be generic"
        )
    if summary.get("raw_logs_published") is not False:
        errors.append(f"{_relative(summary_path)}: raw_logs_published must be false")
    if summary.get("weights_vendored") is not False:
        errors.append(f"{_relative(summary_path)}: weights_vendored must be false")
    if summary.get("support_matrix_change_claimed") is not False:
        errors.append(
            f"{_relative(summary_path)}: support_matrix_change_claimed must be false"
        )
    if summary.get("suite") != "model-catalog-gpu":
        errors.append(f"{_relative(summary_path)}: suite must be model-catalog-gpu")
    if summary.get("execution_mode") != "container":
        errors.append(f"{_relative(summary_path)}: execution_mode must be container")
    if summary.get("source_window") != "post_batch_18_cutoff":
        errors.append(f"{_relative(summary_path)}: source_window invalid")

    clean_lanes = summary.get("clean_lanes")
    failed_findings = summary.get("failed_findings")
    duplicates = summary.get("duplicate_clean_runs")
    counts = summary.get("counts")
    if not isinstance(clean_lanes, list) or not clean_lanes:
        errors.append(f"{_relative(summary_path)}: clean_lanes must be non-empty")
        clean_lanes = []
    if not isinstance(failed_findings, list):
        errors.append(f"{_relative(summary_path)}: failed_findings must be a list")
        failed_findings = []
    if not isinstance(duplicates, list):
        errors.append(f"{_relative(summary_path)}: duplicate_clean_runs must be a list")
        duplicates = []
    if not isinstance(counts, dict):
        errors.append(f"{_relative(summary_path)}: counts must be object")
        counts = {}

    seen_clean: set[str] = set()
    duplicate_extra_runs = 0
    for index, lane in enumerate(clean_lanes):
        if not isinstance(lane, dict):
            errors.append(
                f"{_relative(summary_path)}: clean_lanes[{index}] must be object"
            )
            continue
        slug = lane.get("slug")
        if not isinstance(slug, str) or not slug:
            errors.append(
                f"{_relative(summary_path)}: clean_lanes[{index}].slug required"
            )
        elif slug in seen_clean:
            errors.append(f"{_relative(summary_path)}: duplicate clean lane {slug!r}")
        else:
            seen_clean.add(slug)
        model_id = lane.get("model_id")
        preset = lane.get("preset")
        if not isinstance(model_id, str) or not model_id:
            errors.append(
                f"{_relative(summary_path)}: clean_lanes[{index}].model_id required"
            )
        if (
            not isinstance(preset, str)
            or preset.startswith("/")
            or not (REPO_ROOT / preset).is_file()
        ):
            errors.append(
                f"{_relative(summary_path)}: clean_lanes[{index}].preset invalid"
            )
        if lane.get("rc") != 0:
            errors.append(f"{_relative(summary_path)}: {slug} rc must be zero")
        if lane.get("evaluate_exit") != 0 or lane.get("verify_exit") != 0:
            errors.append(
                f"{_relative(summary_path)}: {slug} evaluate/verify exits must be zero"
            )
        if lane.get("report_materialized") is not True:
            errors.append(f"{_relative(summary_path)}: {slug} report must materialize")
        if lane.get("verify_materialized") is not True:
            errors.append(f"{_relative(summary_path)}: {slug} verify must materialize")
        if lane.get("status") != "ok":
            errors.append(f"{_relative(summary_path)}: {slug} status must be ok")

    for index, duplicate in enumerate(duplicates):
        if not isinstance(duplicate, dict):
            errors.append(
                f"{_relative(summary_path)}: duplicate_clean_runs[{index}] must be object"
            )
            continue
        slug = duplicate.get("slug")
        extra = duplicate.get("additional_clean_runs")
        if slug not in seen_clean:
            errors.append(
                f"{_relative(summary_path)}: duplicate_clean_runs[{index}].slug unknown"
            )
        if not isinstance(extra, int) or extra <= 0:
            errors.append(
                f"{_relative(summary_path)}: duplicate_clean_runs[{index}] "
                "additional_clean_runs invalid"
            )
        else:
            duplicate_extra_runs += extra

    unique_failed: set[str] = set()
    failed_attempts = 0
    for index, finding in enumerate(failed_findings):
        if not isinstance(finding, dict):
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}] must be object"
            )
            continue
        slug = finding.get("slug")
        if not isinstance(slug, str) or not slug:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].slug required"
            )
        else:
            unique_failed.add(slug)
        model_id = finding.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].model_id required"
            )
        preset = finding.get("preset")
        if (
            not isinstance(preset, str)
            or preset.startswith("/")
            or not (REPO_ROOT / preset).is_file()
        ):
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].preset invalid"
            )
        attempts = finding.get("attempts")
        if not isinstance(attempts, int) or attempts <= 0:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].attempts invalid"
            )
            attempts = 0
        failed_attempts += attempts
        if finding.get("status") not in QUEUE_DRAIN_FAILURE_STATUSES:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].status invalid"
            )
        if finding.get("classification") not in QUEUE_DRAIN_FAILURE_CLASSIFICATIONS:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}].classification invalid"
            )
        if finding.get("evaluate_exit") != 1 or finding.get("verify_exit") is not None:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}] exit fields invalid"
            )
        if not isinstance(finding.get("report_materialized"), bool):
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}] report flag invalid"
            )
        if finding.get("verify_materialized") is not False:
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}] verify must be false"
            )
        if not isinstance(finding.get("later_clean_run_observed"), bool):
            errors.append(
                f"{_relative(summary_path)}: failed_findings[{index}] "
                "later_clean_run_observed must be boolean"
            )

    expected_counts = {
        "unique_clean_lanes": len(seen_clean),
        "unique_failed_lanes": len(unique_failed),
        "clean_runs": len(seen_clean) + duplicate_extra_runs,
        "failed_runs": failed_attempts,
        "pre_verification_failures": failed_attempts,
        "report_materialized_clean": len(seen_clean) + duplicate_extra_runs,
        "verify_materialized_clean": len(seen_clean) + duplicate_extra_runs,
    }
    expected_counts["completed_runs"] = (
        expected_counts["clean_runs"] + expected_counts["failed_runs"]
    )
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            errors.append(f"{_relative(summary_path)}: counts.{key} must be {expected}")


def _check_larger_model_queue_drain_addendum(
    errors: list[str],
    addendum_path: Path,
) -> None:
    addendum, error = _load_json(addendum_path)
    if error:
        errors.append(error)
        return
    assert addendum is not None
    if addendum.get("schema") != LARGER_MODEL_QUEUE_DRAIN_ADDENDUM_SCHEMA:
        errors.append(
            f"{_relative(addendum_path)}: schema must be "
            f"{LARGER_MODEL_QUEUE_DRAIN_ADDENDUM_SCHEMA}"
        )
    if addendum.get("status") != "completed":
        errors.append(f"{_relative(addendum_path)}: status must be completed")
    if addendum.get("validation_environment") != "CUDA-capable validation host":
        errors.append(
            f"{_relative(addendum_path)}: validation_environment must be generic"
        )
    if addendum.get("raw_logs_published") is not False:
        errors.append(f"{_relative(addendum_path)}: raw_logs_published must be false")
    if addendum.get("weights_vendored") is not False:
        errors.append(f"{_relative(addendum_path)}: weights_vendored must be false")
    if addendum.get("support_matrix_change_claimed") is not False:
        errors.append(
            f"{_relative(addendum_path)}: support_matrix_change_claimed must be false"
        )
    if addendum.get("model_quality_claimed") is not False:
        errors.append(
            f"{_relative(addendum_path)}: model_quality_claimed must be false"
        )
    if addendum.get("source_window") != "post_pr_109_late_clean_addendum":
        errors.append(f"{_relative(addendum_path)}: source_window invalid")
    if addendum.get("execution_mode") != "container":
        errors.append(f"{_relative(addendum_path)}: execution_mode must be container")

    counts = addendum.get("counts")
    if not isinstance(counts, dict):
        errors.append(f"{_relative(addendum_path)}: counts must be object")
        counts = {}

    clean_lanes = addendum.get("late_clean_lanes")
    if not isinstance(clean_lanes, list) or not clean_lanes:
        errors.append(f"{_relative(addendum_path)}: late_clean_lanes must be non-empty")
        clean_lanes = []

    seen_clean: set[str] = set()
    for index, lane in enumerate(clean_lanes):
        if not isinstance(lane, dict):
            errors.append(
                f"{_relative(addendum_path)}: late_clean_lanes[{index}] must be object"
            )
            continue
        slug = lane.get("slug")
        if not isinstance(slug, str) or not slug:
            errors.append(
                f"{_relative(addendum_path)}: late_clean_lanes[{index}].slug required"
            )
        elif slug in seen_clean:
            errors.append(f"{_relative(addendum_path)}: duplicate clean lane {slug!r}")
        else:
            seen_clean.add(slug)
        model_id = lane.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            errors.append(
                f"{_relative(addendum_path)}: late_clean_lanes[{index}].model_id "
                "required"
            )
        preset = lane.get("preset")
        if (
            not isinstance(preset, str)
            or preset.startswith("/")
            or not (REPO_ROOT / preset).is_file()
        ):
            errors.append(
                f"{_relative(addendum_path)}: late_clean_lanes[{index}].preset invalid"
            )
        if lane.get("suite") != "model-catalog-gpu":
            errors.append(
                f"{_relative(addendum_path)}: late_clean_lanes[{index}].suite invalid"
            )
        if lane.get("rc") != 0:
            errors.append(f"{_relative(addendum_path)}: {slug} rc must be zero")
        if lane.get("evaluate_exit") != 0 or lane.get("verify_exit") != 0:
            errors.append(
                f"{_relative(addendum_path)}: {slug} evaluate/verify exits must be zero"
            )
        if lane.get("report_materialized") is not True:
            errors.append(f"{_relative(addendum_path)}: {slug} report must materialize")
        if lane.get("verify_materialized") is not True:
            errors.append(f"{_relative(addendum_path)}: {slug} verify must materialize")
        if lane.get("status") != "ok":
            errors.append(f"{_relative(addendum_path)}: {slug} status must be ok")

    rerun_classifications = addendum.get("rerun_classifications")
    if not isinstance(rerun_classifications, list):
        errors.append(
            f"{_relative(addendum_path)}: rerun_classifications must be a list"
        )
        rerun_classifications = []
    for index, classification in enumerate(rerun_classifications):
        if not isinstance(classification, dict):
            errors.append(
                f"{_relative(addendum_path)}: rerun_classifications[{index}] "
                "must be object"
            )
            continue
        slug = classification.get("slug")
        if slug not in seen_clean:
            errors.append(
                f"{_relative(addendum_path)}: rerun_classifications[{index}].slug "
                "must reference a clean lane"
            )
        if classification.get("previous_classification") not in (
            QUEUE_DRAIN_FAILURE_CLASSIFICATIONS
        ):
            errors.append(
                f"{_relative(addendum_path)}: rerun_classifications[{index}] "
                "previous_classification invalid"
            )
        if classification.get("later_clean_run_observed") is not True:
            errors.append(
                f"{_relative(addendum_path)}: rerun_classifications[{index}] "
                "must observe a later clean run"
            )

    excluded_lanes = addendum.get("excluded_lanes")
    if not isinstance(excluded_lanes, list):
        errors.append(f"{_relative(addendum_path)}: excluded_lanes must be a list")
        excluded_lanes = []
    for index, lane in enumerate(excluded_lanes):
        if not isinstance(lane, dict):
            errors.append(
                f"{_relative(addendum_path)}: excluded_lanes[{index}] must be object"
            )
            continue
        slug = lane.get("slug")
        model_id = lane.get("model_id")
        reason = lane.get("reason")
        if not isinstance(slug, str) or not slug:
            errors.append(
                f"{_relative(addendum_path)}: excluded_lanes[{index}].slug required"
            )
        if not isinstance(model_id, str) or not model_id:
            errors.append(
                f"{_relative(addendum_path)}: excluded_lanes[{index}].model_id required"
            )
        if not isinstance(reason, str) or not reason:
            errors.append(
                f"{_relative(addendum_path)}: excluded_lanes[{index}].reason required"
            )

    expected_counts = {
        "late_clean_lanes": len(seen_clean),
        "rerun_clean_resolutions": len(rerun_classifications),
        "excluded_lanes": len(excluded_lanes),
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            errors.append(
                f"{_relative(addendum_path)}: counts.{key} must be {expected}"
            )


def _check_larger_model_queue_drain_modern_addendum(
    errors: list[str],
    addendum_path: Path,
) -> None:
    addendum, error = _load_json(addendum_path)
    if error:
        errors.append(error)
        return
    assert addendum is not None
    if addendum.get("schema") != LARGER_MODEL_QUEUE_DRAIN_MODERN_ADDENDUM_SCHEMA:
        errors.append(
            f"{_relative(addendum_path)}: schema must be "
            f"{LARGER_MODEL_QUEUE_DRAIN_MODERN_ADDENDUM_SCHEMA}"
        )
    if addendum.get("status") != "completed":
        errors.append(f"{_relative(addendum_path)}: status must be completed")
    if addendum.get("validation_environment") != "CUDA-capable validation host":
        errors.append(
            f"{_relative(addendum_path)}: validation_environment must be generic"
        )
    for key in (
        "raw_logs_published",
        "weights_vendored",
        "support_matrix_change_claimed",
        "model_quality_claimed",
    ):
        if addendum.get(key) is not False:
            errors.append(f"{_relative(addendum_path)}: {key} must be false")
    if addendum.get("source_window") != "post_pr_111_modern_followon":
        errors.append(f"{_relative(addendum_path)}: source_window invalid")
    if addendum.get("execution_mode") != "container":
        errors.append(f"{_relative(addendum_path)}: execution_mode must be container")

    counts = addendum.get("counts")
    if not isinstance(counts, dict):
        errors.append(f"{_relative(addendum_path)}: counts must be object")
        counts = {}

    clean_lanes = addendum.get("clean_followon_lanes")
    if not isinstance(clean_lanes, list) or not clean_lanes:
        errors.append(
            f"{_relative(addendum_path)}: clean_followon_lanes must be non-empty"
        )
        clean_lanes = []
    for index, lane in enumerate(clean_lanes):
        if not isinstance(lane, dict):
            errors.append(
                f"{_relative(addendum_path)}: clean_followon_lanes[{index}] "
                "must be object"
            )
            continue
        if lane.get("rc") != 0 or lane.get("evaluate_exit") != 0:
            errors.append(
                f"{_relative(addendum_path)}: clean_followon_lanes[{index}] "
                "must have clean evaluation"
            )
        if lane.get("verify_exit") != 0 or lane.get("status") != "ok":
            errors.append(
                f"{_relative(addendum_path)}: clean_followon_lanes[{index}] "
                "must have clean verification"
            )
        preset = lane.get("preset_basis")
        if not isinstance(preset, str) or not (REPO_ROOT / preset).is_file():
            errors.append(
                f"{_relative(addendum_path)}: clean_followon_lanes[{index}] "
                "preset_basis must be a repo file"
            )

    diagnostics = addendum.get("diagnostic_lanes")
    if not isinstance(diagnostics, list) or not diagnostics:
        errors.append(f"{_relative(addendum_path)}: diagnostic_lanes must be non-empty")
        diagnostics = []
    for index, lane in enumerate(diagnostics):
        if not isinstance(lane, dict):
            errors.append(
                f"{_relative(addendum_path)}: diagnostic_lanes[{index}] must be object"
            )
            continue
        if lane.get("evaluate_exit") != 0 or lane.get("verify_exit") != 0:
            errors.append(
                f"{_relative(addendum_path)}: diagnostic_lanes[{index}] must pass"
            )
        if lane.get("support_claimed") is not False:
            errors.append(
                f"{_relative(addendum_path)}: diagnostic_lanes[{index}] "
                "support_claimed must be false"
            )

    strict_findings = addendum.get("strict_policy_findings")
    if not isinstance(strict_findings, list) or not strict_findings:
        errors.append(
            f"{_relative(addendum_path)}: strict_policy_findings must be non-empty"
        )
        strict_findings = []
    for index, finding in enumerate(strict_findings):
        if not isinstance(finding, dict):
            errors.append(
                f"{_relative(addendum_path)}: strict_policy_findings[{index}] "
                "must be object"
            )
            continue
        if finding.get("detail") != "policy_fail":
            errors.append(
                f"{_relative(addendum_path)}: strict_policy_findings[{index}] "
                "detail must be policy_fail"
            )
        if finding.get("verify_exit") != 1 or finding.get("evaluate_exit") != 0:
            errors.append(
                f"{_relative(addendum_path)}: strict_policy_findings[{index}] "
                "must fail only at verification"
            )
        if finding.get("classification") != "strict_spectral_cap_budget_boundary":
            errors.append(
                f"{_relative(addendum_path)}: strict_policy_findings[{index}] "
                "classification invalid"
            )

    dependency_findings = addendum.get("dependency_findings")
    if not isinstance(dependency_findings, list):
        errors.append(f"{_relative(addendum_path)}: dependency_findings must be a list")
        dependency_findings = []
    for index, finding in enumerate(dependency_findings):
        if not isinstance(finding, dict):
            errors.append(
                f"{_relative(addendum_path)}: dependency_findings[{index}] "
                "must be object"
            )
            continue
        if finding.get("classification") != "runtime_dependency_missing":
            errors.append(
                f"{_relative(addendum_path)}: dependency_findings[{index}] "
                "classification invalid"
            )
        if finding.get("verify_exit") is not None:
            errors.append(
                f"{_relative(addendum_path)}: dependency_findings[{index}] "
                "verify_exit must be null"
            )

    expected_counts = {
        "clean_followon_lanes": len(clean_lanes),
        "diagnostic_lanes": len(diagnostics),
        "strict_policy_findings": len(strict_findings),
        "dependency_findings": len(dependency_findings),
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            errors.append(
                f"{_relative(addendum_path)}: counts.{key} must be {expected}"
            )


def check_public_evidence(root: Path = PUBLIC_EVIDENCE_ROOT) -> list[str]:
    errors: list[str] = []
    root = root.resolve()
    if not (root / "README.md").is_file():
        errors.append(f"{_relative(root)}: README.md is required")
    if not root.is_dir():
        return [f"public evidence root not found: {root}"]
    _check_public_evidence_privacy(errors, root)

    for artifact_dir in sorted(_artifact_dirs(root)):
        meta_path = artifact_dir / META_FILENAME
        if not meta_path.is_file():
            errors.append(f"{_relative(artifact_dir)}: missing {META_FILENAME}")
            continue

        metadata, error = _load_json(meta_path)
        if error:
            errors.append(error)
            continue
        assert metadata is not None

        if metadata.get("schema") != SCHEMA:
            errors.append(f"{_relative(meta_path)}: schema must be {SCHEMA}")

        evidence_class = metadata.get("evidence_class")
        if evidence_class not in ALLOWED_CLASSES:
            errors.append(f"{_relative(meta_path)}: invalid evidence_class")
            continue

        summary = str(metadata.get("summary") or "").lower()
        if evidence_class not in NON_FIXTURE_CLASSES and "fixture" not in summary:
            errors.append(f"{_relative(meta_path)}: fixture evidence must say fixture")

        artifact_paths = metadata.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            errors.append(f"{_relative(meta_path)}: artifact_paths must be an object")
            continue

        if (artifact_dir / "evaluation.report.json").is_file():
            report_path = _require_path(
                errors, artifact_dir, artifact_paths, "evaluation_report"
            )
            _require_path(errors, artifact_dir, artifact_paths, "runtime_manifest")
            if report_path is not None and _is_direct_published_basis_artifact(
                artifact_dir, root
            ):
                _check_published_basis_multimodal_quality(
                    errors, artifact_dir, report_path
                )

        if evidence_class in REAL_CLASSES:
            _require_path(errors, artifact_dir, artifact_paths, "run_command")
            if "invarlock evaluate" not in str(metadata.get("generated_by") or ""):
                errors.append(
                    f"{_relative(meta_path)}: real runs must record invarlock evaluate"
                )
            if "fixture" in summary:
                errors.append(
                    f"{_relative(meta_path)}: real-run summary must not say fixture"
                )

        if "evidence_pack" in artifact_paths:
            _check_signed_pack(errors, artifact_dir, metadata, artifact_paths)

        if evidence_class == "real_guard_value_demo":
            _check_guard_value_demo(errors, artifact_dir, artifact_paths)

        if evidence_class == "runtime_backend_compat_sweep":
            _check_runtime_backend_compat_sweep(errors, artifact_dir, artifact_paths)

        if evidence_class == "evidence_pack_queue_stress_resume":
            _check_evidence_pack_queue_stress_resume(
                errors, artifact_dir, artifact_paths
            )

        if evidence_class == "fa2_fallback_compatibility":
            _check_fa2_fallback_compatibility(errors, artifact_dir, artifact_paths)

        if evidence_class == "larger_model_smoke_findings":
            _check_larger_model_smoke_findings(errors, artifact_dir, artifact_paths)

        if evidence_class == "larger_model_queue_drain_findings":
            _check_larger_model_queue_drain_findings(
                errors, artifact_dir, artifact_paths
            )

        commands = metadata.get("verifier_commands")
        if not isinstance(commands, list) or not commands:
            errors.append(
                f"{_relative(meta_path)}: verifier_commands must be a non-empty list"
            )

    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PUBLIC_EVIDENCE_ROOT,
        help="Public evidence root to audit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors = check_public_evidence(args.root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Public evidence audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
