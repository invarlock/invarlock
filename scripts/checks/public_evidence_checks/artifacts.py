"""Artifact, signed-pack, and measured-report validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from scripts.checks.public_evidence_checks.common import _load_json, _relative

PUBLISHED_BASIS_MULTIMODAL_MIN_FINAL_ACCURACY = 0.10
PUBLISHED_BASIS_MULTIMODAL_MIN_FINAL_EXAMPLES = 200
PUBLISHED_BASIS_MULTIMODAL_MIN_ANSWER_SHAPE_RATE = 0.95
PUBLISHED_BASIS_MULTIMODAL_MAX_ANSWER_WORDS = 12
PUBLISHED_BASIS_MULTIMODAL_MAX_ANSWER_CHARS = 80


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
    nested_primary = (
        metrics.get("primary_metric") if isinstance(metrics, dict) else None
    )
    if isinstance(nested_primary, dict):
        return nested_primary
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
