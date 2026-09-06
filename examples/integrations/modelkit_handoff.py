"""Verify a bounded ModelKit package against the actual model selected for use."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import tempfile
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from invarlock.acceptance_attestation import verify_acceptance_attestation
from invarlock.core.checkpoint_identity import (
    CheckpointIdentityError,
    checkpoint_tree_observation,
    checkpoint_tree_sha256,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.evidence_pack_verification import verify_comparison_evidence

MANIFEST_MEDIA = "application/vnd.oci.image.manifest.v1+json"
ARTIFACT_MEDIA = "application/vnd.kitops.modelkit.manifest.v1+json"
CONFIG_MEDIA = "application/vnd.kitops.modelkit.config.v1+json"
MODEL_TAR = "application/vnd.kitops.modelkit.model.v1.tar"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CHUNK = 1024 * 1024


class ModelKitError(ValueError):
    """The selected package or actual candidate could not be authenticated."""


@dataclass(frozen=True)
class Limits:
    """Recipient-selected resource ceilings; package metadata cannot raise them."""

    max_json_bytes: int = 2 * 1024 * 1024
    max_blob_bytes: int = 160 * 1024**3
    max_archive_bytes: int = 160 * 1024**3
    max_model_bytes: int = 160 * 1024**3
    max_members: int = 200_000

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ModelKitError("resource limits must be positive integers")


def _digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ModelKitError("an independently selected SHA-256 digest is required")
    return value


def _object(value: object, required: set[str], optional: set[str]) -> dict:
    if not isinstance(value, dict) or not required <= value.keys():
        raise ModelKitError("missing required package fields")
    if value.keys() - required - optional:
        raise ModelKitError("unsupported package fields")
    return value


def _copy(source: BinaryIO, target: BinaryIO | None, maximum: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(min(_CHUNK, maximum - size + 1)):
        size += len(chunk)
        if size > maximum:
            raise ModelKitError("package resource limit exceeded")
        digest.update(chunk)
        if target is not None:
            target.write(chunk)
    return "sha256:" + digest.hexdigest(), size


def _stable_stat(value: os.stat_result) -> tuple:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _blob(
    blobs: Path, digest: str, target: BinaryIO, *, maximum: int, size: int | None = None
) -> int:
    path = blobs / _digest(digest).removeprefix("sha256:")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ModelKitError("package blob must be a regular file")
        if before.st_size > maximum:
            raise ModelKitError("package blob size exceeds resource limit")
        if size is not None and before.st_size != size:
            raise ModelKitError("package blob size mismatch")
        actual, length = _copy(source, target, maximum)
        after = os.fstat(source.fileno())
    if actual != digest:
        raise ModelKitError("package blob digest mismatch")
    if _stable_stat(before) != _stable_stat(after):
        raise ModelKitError("package blob changed during verification")
    return length


def _json_blob(blobs: Path, digest: str, limits: Limits, size=None) -> tuple[dict, int]:
    data = io.BytesIO()
    length = _blob(blobs, digest, data, maximum=limits.max_json_bytes, size=size)
    try:
        value = parse_json_bytes(data.getvalue(), label="ModelKit metadata")
    except StrictJsonError as exc:
        raise ModelKitError("invalid or duplicate JSON fields") from exc
    if not isinstance(value, dict):
        raise ModelKitError("package JSON must be an object")
    return value, length


def _descriptor(value: object, media_types: set[str]) -> dict:
    result = _object(value, {"mediaType", "digest", "size"}, {"annotations"})
    if (
        not isinstance(result["mediaType"], str)
        or result["mediaType"] not in media_types
    ):
        raise ModelKitError("unsupported package media type")
    _digest(result["digest"])
    if type(result["size"]) is not int or result["size"] <= 0:
        raise ModelKitError("invalid descriptor size")
    return result


def _relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ModelKitError("unsafe package path")
    parts = value.removesuffix("/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ModelKitError("unsafe package path")
    return PurePosixPath(*parts)


def _model_descriptor(blobs: Path, package_digest: str, limits: Limits):
    manifest, manifest_size = _json_blob(blobs, package_digest, limits)
    _object(
        manifest,
        {"schemaVersion", "mediaType", "artifactType", "config", "layers"},
        {"annotations"},
    )
    if (
        type(manifest["schemaVersion"]) is not int
        or manifest["schemaVersion"] != 2
        or manifest["mediaType"] != MANIFEST_MEDIA
        or manifest["artifactType"] != ARTIFACT_MEDIA
    ):
        raise ModelKitError("unsupported ModelKit manifest")
    descriptor = _descriptor(manifest["config"], {CONFIG_MEDIA})
    config, _ = _json_blob(blobs, descriptor["digest"], limits, descriptor["size"])
    _object(config, {"manifestVersion", "model"}, {"package"})
    if config["manifestVersion"] != "1.0.0":
        raise ModelKitError("unsupported ModelKit config version")
    model = _object(config["model"], {"path", "digest", "diffId"}, set())
    model_path = _relative(model["path"])
    _digest(model["diffId"])
    layers = manifest["layers"]
    if not isinstance(layers, list) or len(layers) != 1:
        raise ModelKitError("exactly one embedded model layer is supported")
    layer = _descriptor(layers[0], {MODEL_TAR, MODEL_TAR + "+gzip"})
    if model["digest"] != layer["digest"]:
        raise ModelKitError("config model digest does not identify the model layer")
    return manifest_size, descriptor, layer, model_path, model["diffId"]


def _inventory(root: Path, limits: Limits) -> dict[str, tuple[str, int]]:
    """Compare every file, including operational files omitted by model identity."""
    result = {}
    total = 0
    members = 0

    def onerror(error: OSError):
        raise ModelKitError(
            "candidate file inventory could not be read completely"
        ) from error

    for directory, dirs, files, directory_fd in os.fwalk(
        root, follow_symlinks=False, onerror=onerror
    ):
        members += len(dirs) + len(files)
        if members > limits.max_members:
            raise ModelKitError("candidate member limit exceeded")
        for name in dirs:
            if not stat.S_ISDIR(
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
            ):
                raise ModelKitError("candidate directories must not be symlinks")
        for name in files:
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            with os.fdopen(os.open(name, flags, dir_fd=directory_fd), "rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ModelKitError("candidate files must be regular files")
                digest, size = _copy(source, None, limits.max_model_bytes - total)
                if _stable_stat(os.fstat(source.fileno())) != _stable_stat(before):
                    raise ModelKitError("candidate file changed while being read")
            total += size
            relative = (Path(directory) / name).relative_to(root).as_posix()
            result[relative] = (digest, size)
    return result


def _inventory_digest(files: dict[str, tuple[str, int]]) -> str:
    raw = json.dumps(
        files, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _archive_headers(archive: BinaryIO, limits: Limits) -> None:
    """Bound metadata parsing before tarfile can consume extension payloads."""
    length = archive.seek(0, os.SEEK_END)
    archive.seek(0)
    members = 0
    while True:
        block = archive.read(512)
        if block == b"\0" * 512:
            if archive.read(512) != b"\0" * 512:
                raise ModelKitError("archive must have two zero end headers")
            while trailing := archive.read(_CHUNK):
                if trailing.strip(b"\0"):
                    raise ModelKitError("unexpected content after archive end headers")
            archive.seek(0)
            return
        if len(block) != 512:
            raise ModelKitError("incomplete archive header")
        member = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
        if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}:
            raise ModelKitError("unsupported extended or special archive header")
        if member.size < 0 or (member.isdir() and member.size):
            raise ModelKitError("invalid archive header size")
        members += 1
        if members > limits.max_members:
            raise ModelKitError("archive member limit exceeded")
        position = archive.tell() + ((member.size + 511) // 512) * 512
        if position > length:
            raise ModelKitError("archive header exceeds available contents")
        archive.seek(position)


def _extract(archive: BinaryIO, root: Path, model_path: PurePosixPath, limits: Limits):
    _archive_headers(archive, limits)
    names: set[PurePosixPath] = set()
    total = 0
    files = {}
    with tarfile.open(fileobj=archive, mode="r:") as tar:
        for index, member in enumerate(tar, start=1):
            if index > limits.max_members:
                raise ModelKitError("archive member limit exceeded")
            name = _relative(member.name)
            if name in names:
                raise ModelKitError("duplicate archive member")
            names.add(name)
            in_model = name == model_path or model_path in name.parents
            ancestor = name in model_path.parents
            if not in_model and not (ancestor and member.isdir()):
                raise ModelKitError("archive member outside declared model path")
            destination = root.joinpath(*name.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issparse() or name == model_path:
                raise ModelKitError(
                    "only regular files in a model directory are supported"
                )
            if member.size < 0:
                raise ModelKitError("invalid archive member size")
            total += member.size
            if total > limits.max_model_bytes:
                raise ModelKitError("extracted model byte limit exceeded")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ModelKitError("missing archive file contents")
            with source, destination.open("xb") as target:
                digest, length = _copy(source, target, member.size)
            if length != member.size:
                raise ModelKitError("incomplete archive member")
            files[name.relative_to(model_path).as_posix()] = (digest, length)
    if not files:
        raise ModelKitError("model layer contains no regular files")
    return files, total


def verify_package_content(
    *,
    blobs: Path,
    expected_package_digest: str,
    candidate: Path,
    expected_content_digest: str,
    limits: Limits = Limits(),
) -> dict:
    """Recompute raw package, model and actual candidate bindings offline.

    This result describes package/content integrity only. Technical evidence and
    current recipient acceptance require their own independently supplied trust
    inputs. The caller must also bind its eventual consumer to the checked path.
    """
    _digest(expected_package_digest)
    _digest(expected_content_digest)
    try:
        before = checkpoint_tree_observation(candidate.absolute())
        if before.digest != expected_content_digest:
            raise ModelKitError(
                "actual candidate content differs from expected content"
            )
        candidate_files = _inventory(candidate.absolute(), limits)
        size, config, layer, model_path, diff_id = _model_descriptor(
            blobs, expected_package_digest, limits
        )
        with tempfile.TemporaryDirectory(prefix="invarlock-modelkit-") as workspace:
            root = Path(workspace).resolve()
            with (
                tempfile.TemporaryFile() as stored,
                tempfile.TemporaryFile() as archive,
            ):
                _blob(
                    blobs,
                    layer["digest"],
                    stored,
                    maximum=limits.max_blob_bytes,
                    size=layer["size"],
                )
                stored.seek(0)
                if layer["mediaType"].endswith("+gzip"):
                    with gzip.GzipFile(fileobj=stored) as source:
                        actual_diff, _ = _copy(
                            source, archive, limits.max_archive_bytes
                        )
                else:
                    actual_diff, _ = _copy(stored, archive, limits.max_archive_bytes)
                if actual_diff != diff_id:
                    raise ModelKitError("model layer DiffID mismatch")
                archive.seek(0)
                files, model_bytes = _extract(archive, root, model_path, limits)
            extracted = checkpoint_tree_sha256(root.joinpath(*model_path.parts))
        if files != candidate_files:
            raise ModelKitError(
                "package and actual candidate content file inventory differ"
            )
        if extracted != expected_content_digest:
            raise ModelKitError("package model content differs from expected content")
        if checkpoint_tree_observation(candidate.absolute()) != before:
            raise ModelKitError("actual candidate changed during package verification")
        if _inventory(candidate.absolute(), limits) != candidate_files:
            raise ModelKitError(
                "actual candidate file inventory changed during verification"
            )
    except (
        OSError,
        EOFError,
        UnicodeError,
        zlib.error,
        tarfile.TarError,
        CheckpointIdentityError,
    ) as exc:
        raise ModelKitError(f"package or candidate verification failed: {exc}") from exc
    return {
        "format": "invarlock/example-modelkit-content-v1",
        "package_digest": expected_package_digest,
        "package_size": size,
        "config_digest": config["digest"],
        "layer_digest": layer["digest"],
        "layer_diff_id": diff_id,
        "layer_media_type": layer["mediaType"],
        "model_path": str(model_path),
        "model_file_count": len(files),
        "model_file_inventory_digest": _inventory_digest(files),
        "model_bytes": model_bytes,
        "artifact_digest_kind": "hf_snapshot_tree_sha256",
        "artifact_content_digest": extracted,
    }


def _path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ModelKitError("recipient input path must be a nonempty string")
    # Preserve symbolic-link components for the underlying verifier to reject.
    return root / value


def verify_point_of_use(
    request: dict, *, root: Path = Path("."), now: datetime | None = None
) -> dict:
    """Apply recipient-owned trust to actual packages, evidence and acceptance.

    Request paths are relative to the recipient request file. The CLI uses the
    current clock; ``now`` is exposed only for reproducible fixture tests.
    """
    _object(
        request,
        {
            "format",
            "sides",
            "evidence",
            "technical_policy",
            "technical_anchors",
            "envelope",
            "recipient_policy",
            "trusted_public_keys",
        },
        {"limits"},
    )
    if request["format"] != "invarlock/example-modelkit-recipient-v1":
        raise ModelKitError("unsupported recipient request format")
    limit_values = _object(request.get("limits", {}), set(), set(vars(Limits())))
    limits = Limits(**limit_values)
    sides = _object(request["sides"], {"baseline", "subject"}, set())
    mappings = {}
    observations = {}
    for role, value in sides.items():
        side = _object(
            value, {"blobs", "package_digest", "content_digest", "candidate"}, set()
        )
        candidate = _path(root, side["candidate"]).absolute()
        observations[role] = checkpoint_tree_observation(candidate)
        mappings[role] = verify_package_content(
            blobs=_path(root, side["blobs"]),
            expected_package_digest=side["package_digest"],
            candidate=candidate,
            expected_content_digest=side["content_digest"],
            limits=limits,
        )
    anchors = _object(
        request["technical_anchors"],
        {
            "artifact_digests",
            "runtime_digests",
            "schedule_digest",
            "evidence_signer_fingerprint",
        },
        {"request_digest"},
    )
    for field in ("artifact_digests", "runtime_digests"):
        for value in _object(anchors[field], {"baseline", "subject"}, set()).values():
            _digest(value)
    _digest(anchors["schedule_digest"])
    _digest(anchors["evidence_signer_fingerprint"])
    if "request_digest" in anchors:
        _digest(anchors["request_digest"])
    keys = request["trusted_public_keys"]
    if not isinstance(keys, dict) or not keys:
        raise ModelKitError("recipient must supply trusted public keys")
    trusted_keys = {_digest(key): _path(root, value) for key, value in keys.items()}
    technical = verify_comparison_evidence(
        _path(root, request["evidence"]),
        policy_path=_path(root, request["technical_policy"]),
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_signer_fingerprint=anchors["evidence_signer_fingerprint"],
        expected_request_digest=anchors.get("request_digest"),
    )
    checked_at = now or datetime.now(UTC)
    decision = verify_acceptance_attestation(
        _path(root, request["envelope"]),
        trusted_public_keys=trusted_keys,
        recipient_policy=_path(root, request["recipient_policy"]),
        subject_artifact_path=_path(root, sides["subject"]["candidate"]),
        now=checked_at,
    )
    errors = list(technical.payload["errors"]) + list(decision.errors)
    bound = False
    if (
        decision.envelope_authenticated
        and decision.receipt_authenticated
        and decision.subject_bound
    ):
        predicate = decision.statement["predicate"]
        statement = predicate["receipt"]["content"]["statement"]
        bound = statement["pack_manifest_digest"] == technical.manifest_digest
        for role in ("baseline", "subject"):
            bound = bound and (
                predicate[role]["artifact_digest"] == sides[role]["content_digest"]
                and predicate[role]["artifact_identity_digest"]
                == anchors["artifact_digests"][role]
            )
    if not bound:
        errors.append(
            "acceptance envelope is not bound to the checked evidence and model contents"
        )
    for role, side in sides.items():
        if (
            checkpoint_tree_observation(_path(root, side["candidate"]).absolute())
            != observations[role]
        ):
            raise ModelKitError(
                "actual candidate changed during recipient verification"
            )
        if (
            _inventory_digest(
                _inventory(_path(root, side["candidate"]).absolute(), limits)
            )
            != mappings[role]["model_file_inventory_digest"]
        ):
            raise ModelKitError(
                "actual candidate file inventory changed during recipient verification"
            )
    integrity = technical.payload["integrity_ok"]
    accepted = (
        integrity
        and technical.payload["policy_verdict"] == "pass"
        and bound
        and decision.accepted
    )
    return {
        "format": "invarlock/example-modelkit-decision-v1",
        "checked_at": checked_at.isoformat(),
        "packages": mappings,
        "technical_integrity_ok": integrity,
        "technical_policy_verdict": technical.payload["policy_verdict"],
        "envelope_authenticated": decision.envelope_authenticated,
        "receipt_authenticated": decision.receipt_authenticated,
        "envelope_evidence_bound": bound,
        "accepted": accepted,
        "evidence_manifest_digest": technical.manifest_digest,
        "exit_code": 0 if accepted else (1 if integrity and bound else 2),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        required=True,
        type=Path,
        help="Independent recipient request; paths are relative to this file",
    )
    args = parser.parse_args(argv)
    try:
        request = parse_json_bytes(
            read_regular_file_bytes(
                args.request,
                label="recipient request",
                max_bytes=Limits().max_json_bytes,
            ),
            label="recipient request",
        )
        result = verify_point_of_use(request, root=args.request.absolute().parent)
    except (ModelKitError, StrictJsonError, OSError, CheckpointIdentityError) as exc:
        result = {
            "format": "invarlock/example-modelkit-decision-v1",
            "accepted": False,
            "exit_code": 2,
            "errors": [str(exc)],
        }
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
