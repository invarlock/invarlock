#!/usr/bin/env python3
"""Copy and verify the subject-specific inputs in a partner review handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "invarlock/design-partner-subject-handoff-v1"
MANIFEST_NAME = "subject-handoff-binding.json"
REPORT_NAME = "evaluation.report.json"
RECEIPT_NAME = "subject-transformation-receipt"
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
REMOTE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
LOCAL_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CHANGE_KIND_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


class HandoffBindingError(ValueError):
    """Raised when the diagnostic handoff cannot be bound or verified."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular_file(
    path: Path, *, label: str, max_bytes: int | None = None
) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise HandoffBindingError(f"{label} must be a regular file")
    data = path.read_bytes()
    if not data:
        raise HandoffBindingError(f"{label} must not be empty")
    if max_bytes is not None and len(data) > max_bytes:
        raise HandoffBindingError(f"{label} exceeds the {max_bytes}-byte handoff limit")
    return data


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffBindingError(f"{label} must be a UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise HandoffBindingError(f"{label} must be a UTF-8 JSON object")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffBindingError(f"{label} must be an object")
    return value


def _model_identity(value: object, *, label: str) -> dict[str, str]:
    identity = _mapping(value, label=label)
    kind = identity.get("kind")
    if (
        kind == "remote_revision"
        and set(identity) == {"kind", "revision"}
        and isinstance(identity.get("revision"), str)
        and REMOTE_REVISION_RE.fullmatch(identity["revision"])
    ):
        return {"kind": kind, "revision": identity["revision"]}
    if (
        kind == "local_checkpoint_tree"
        and set(identity) == {"kind", "sha256"}
        and isinstance(identity.get("sha256"), str)
        and LOCAL_DIGEST_RE.fullmatch(identity["sha256"])
    ):
        return {"kind": kind, "sha256": identity["sha256"]}
    raise HandoffBindingError(f"{label} is not a canonical typed model identity")


def _normalized_remote_model_id(value: str) -> str:
    normalized = value.removeprefix("hf:").strip()
    if not normalized or any(character.isspace() for character in normalized):
        raise HandoffBindingError("remote subject model id is not canonical")
    return normalized


def _report_subject(report: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    meta = _mapping(report.get("meta"), label="evaluation report meta")
    subject_ref = _mapping(
        report.get("subject_ref"), label="evaluation report subject_ref"
    )
    meta_model_id = meta.get("model_id")
    subject_model_id = subject_ref.get("model_id")
    if not isinstance(meta_model_id, str) or not meta_model_id.strip():
        raise HandoffBindingError("evaluation report meta.model_id is missing")
    if subject_model_id != meta_model_id:
        raise HandoffBindingError(
            "evaluation report subject_ref.model_id does not match meta.model_id"
        )
    identity = _model_identity(
        meta.get("model_identity"), label="evaluation report meta.model_identity"
    )
    subject_identity = _model_identity(
        subject_ref.get("model_identity"),
        label="evaluation report subject_ref.model_identity",
    )
    if subject_identity != identity:
        raise HandoffBindingError(
            "evaluation report subject_ref.model_identity does not match meta.model_identity"
        )
    return meta_model_id, identity


def _artifact_record(*, path: str, data: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": _sha256(data),
        "size_bytes": len(data),
    }


def _validate_change_kind(value: str) -> str:
    normalized = value.strip()
    if not CHANGE_KIND_RE.fullmatch(normalized):
        raise HandoffBindingError(
            "subject change kind must be a lowercase hyphen/underscore slug"
        )
    return normalized


def _require_safe_destination(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise HandoffBindingError(
            f"{label} destination must be absent or a regular file"
        )


def _atomic_replace_bytes(path: Path, data: bytes, *, label: str) -> None:
    _require_safe_destination(path, label=label)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _subject_binding_for_create(
    *,
    subject_model: str,
    subject_revision: str,
    subject_change_kind: str,
    report: Mapping[str, Any],
) -> dict[str, object]:
    report_model_id, report_identity = _report_subject(report)
    change_kind = _validate_change_kind(subject_change_kind)
    model_path = Path(subject_model).expanduser()
    local_subject = model_path.exists() or model_path.is_symlink()

    if local_subject:
        if subject_revision.strip():
            raise HandoffBindingError(
                "local subject cannot also declare a remote --subject-revision"
            )
        if report_identity["kind"] != "local_checkpoint_tree":
            raise HandoffBindingError(
                "local subject does not match evaluation report subject identity"
            )
        public_model_id: str | None = None
    else:
        if not REMOTE_REVISION_RE.fullmatch(subject_revision):
            raise HandoffBindingError(
                "remote subject requires an immutable --subject-revision of "
                "40-64 lowercase hexadecimal characters"
            )
        normalized_model_id = _normalized_remote_model_id(subject_model)
        if (
            report_identity != {"kind": "remote_revision", "revision": subject_revision}
            or _normalized_remote_model_id(report_model_id) != normalized_model_id
        ):
            raise HandoffBindingError(
                "declared remote subject does not match evaluation report subject identity"
            )
        public_model_id = normalized_model_id

    return {
        "change_kind": change_kind,
        "model_id": public_model_id,
        "model_identity": report_identity,
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    _atomic_replace_bytes(
        path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        label="subject handoff binding",
    )


def _create(args: argparse.Namespace) -> None:
    handoff_dir = Path(args.handoff_dir)
    if handoff_dir.is_symlink() or not handoff_dir.is_dir():
        raise HandoffBindingError("handoff directory must already be a directory")

    report_path = handoff_dir / REPORT_NAME
    copied_receipt = handoff_dir / RECEIPT_NAME
    manifest_path = handoff_dir / MANIFEST_NAME
    _require_safe_destination(copied_receipt, label="transformation receipt")
    _require_safe_destination(manifest_path, label="subject handoff binding")
    report_bytes = _read_regular_file(report_path, label="bundled evaluation report")
    report = _json_object(report_bytes, label="bundled evaluation report")
    subject = _subject_binding_for_create(
        subject_model=args.subject_model,
        subject_revision=args.subject_revision,
        subject_change_kind=args.subject_change_kind,
        report=report,
    )

    receipt_bytes = _read_regular_file(
        Path(args.transformation_receipt),
        label="transformation receipt",
        max_bytes=MAX_RECEIPT_BYTES,
    )
    _atomic_replace_bytes(
        copied_receipt,
        receipt_bytes,
        label="transformation receipt",
    )

    manifest = {
        "schema": SCHEMA,
        "subject": subject,
        "artifacts": {
            "evaluation_report": _artifact_record(path=REPORT_NAME, data=report_bytes),
            "transformation_receipt": _artifact_record(
                path=RECEIPT_NAME, data=receipt_bytes
            ),
        },
    }
    _write_manifest(manifest_path, manifest)
    _verify_handoff(handoff_dir)


def _artifact_from_manifest(
    artifacts: Mapping[str, Any], *, name: str, expected_path: str
) -> Mapping[str, Any]:
    record = _mapping(artifacts.get(name), label=f"manifest artifacts.{name}")
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise HandoffBindingError(
            f"manifest artifacts.{name} must contain path, sha256, and size_bytes"
        )
    if record.get("path") != expected_path:
        raise HandoffBindingError(f"manifest artifacts.{name} path is not canonical")
    digest = record.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise HandoffBindingError(f"manifest artifacts.{name} sha256 is not canonical")
    size = record.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise HandoffBindingError(
            f"manifest artifacts.{name} size_bytes is not a positive integer"
        )
    return record


def _verify_artifact(
    handoff_dir: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> bytes:
    path = handoff_dir / str(record["path"])
    data = _read_regular_file(path, label=label)
    if len(data) != record["size_bytes"]:
        raise HandoffBindingError(f"{label} size mismatch")
    if _sha256(data) != record["sha256"]:
        raise HandoffBindingError(f"{label} digest mismatch")
    return data


def _verify_handoff(handoff_dir: Path) -> None:
    if handoff_dir.is_symlink() or not handoff_dir.is_dir():
        raise HandoffBindingError("handoff directory must be a directory")
    manifest = _json_object(
        _read_regular_file(
            handoff_dir / MANIFEST_NAME, label="subject handoff binding"
        ),
        label="subject handoff binding",
    )
    if set(manifest) != {"schema", "subject", "artifacts"}:
        raise HandoffBindingError("subject handoff binding has unrecognized fields")
    if manifest.get("schema") != SCHEMA:
        raise HandoffBindingError("subject handoff binding schema is unsupported")

    subject = _mapping(manifest.get("subject"), label="manifest subject")
    if set(subject) != {"change_kind", "model_id", "model_identity"}:
        raise HandoffBindingError("manifest subject has unrecognized fields")
    change_kind = subject.get("change_kind")
    if not isinstance(change_kind, str):
        raise HandoffBindingError("manifest subject change kind must be a string")
    _validate_change_kind(change_kind)
    identity = _model_identity(
        subject.get("model_identity"), label="manifest subject.model_identity"
    )

    model_id = subject.get("model_id")
    if identity["kind"] == "remote_revision":
        if not isinstance(model_id, str):
            raise HandoffBindingError("remote manifest subject model id is missing")
        model_id = _normalized_remote_model_id(model_id)
    elif model_id is not None:
        raise HandoffBindingError("local manifest subject must not expose a model path")

    artifacts = _mapping(manifest.get("artifacts"), label="manifest artifacts")
    if set(artifacts) != {"evaluation_report", "transformation_receipt"}:
        raise HandoffBindingError("manifest artifacts has unrecognized fields")
    report_record = _artifact_from_manifest(
        artifacts, name="evaluation_report", expected_path=REPORT_NAME
    )
    receipt_record = _artifact_from_manifest(
        artifacts, name="transformation_receipt", expected_path=RECEIPT_NAME
    )
    report_bytes = _verify_artifact(
        handoff_dir, report_record, label="evaluation report"
    )
    _verify_artifact(handoff_dir, receipt_record, label="transformation receipt")

    report_model_id, report_identity = _report_subject(
        _json_object(report_bytes, label="evaluation report")
    )
    if report_identity != identity:
        raise HandoffBindingError(
            "manifest subject identity does not match evaluation report"
        )
    if identity["kind"] == "remote_revision" and (
        _normalized_remote_model_id(report_model_id) != model_id
    ):
        raise HandoffBindingError(
            "manifest subject model id does not match evaluation report"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind and verify subject-specific design-partner handoff inputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="copy and bind subject inputs")
    create.add_argument("--handoff-dir", required=True)
    create.add_argument("--subject-model", required=True)
    create.add_argument("--subject-revision", default="")
    create.add_argument("--subject-change-kind", required=True)
    create.add_argument("--transformation-receipt", required=True)

    verify = subparsers.add_parser("verify", help="verify an existing handoff binding")
    verify.add_argument("--handoff-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            _create(args)
        else:
            _verify_handoff(Path(args.handoff_dir))
    except (HandoffBindingError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("subject handoff binding verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
