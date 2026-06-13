#!/usr/bin/env python3
"""Audit public evidence classification and verifier metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
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
}
REAL_CLASSES = {"real_model_run", "real_guard_value_demo", "signed_real_model_pack"}


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


def check_public_evidence(root: Path = PUBLIC_EVIDENCE_ROOT) -> list[str]:
    errors: list[str] = []
    root = root.resolve()
    if not (root / "README.md").is_file():
        errors.append(f"{_relative(root)}: README.md is required")
    if not root.is_dir():
        return [f"public evidence root not found: {root}"]

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
        if evidence_class not in REAL_CLASSES and "fixture" not in summary:
            errors.append(f"{_relative(meta_path)}: fixture evidence must say fixture")

        artifact_paths = metadata.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            errors.append(f"{_relative(meta_path)}: artifact_paths must be an object")
            continue

        if (artifact_dir / "evaluation.report.json").is_file():
            _require_path(errors, artifact_dir, artifact_paths, "evaluation_report")
            _require_path(errors, artifact_dir, artifact_paths, "runtime_manifest")

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
