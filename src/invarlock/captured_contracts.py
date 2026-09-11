"""Captured-only contracts, bounded immutable snapshots, and safe output primitives."""

from __future__ import annotations

import base64
import errno
import hashlib
import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from jsonschema import Draft202012Validator

from invarlock.evaluation_record_contracts import contracts as record_contracts
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    validate,
)
from invarlock.evaluation_record_contracts.validation_limits import check_record_counts
from invarlock.evidence_pack_contract import EvidencePackError, canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import StrictJsonError, parse_json_bytes
from invarlock.public_contracts import (
    load_evidence_pack_v2_schema,
    load_evidence_verification_receipt_v3_schema,
)

CAPTURED_PACK_FORMAT = "invarlock/evidence-pack-v2"
CAPTURED_RECEIPT_FORMAT = "invarlock/evidence-verification-receipt-v3"
DETECTOR_LIMIT = 256 * 1024
CONTROL_LIMIT = 64 * 1024
REQUEST_LIMIT = 1024 * 1024
PAYLOAD_LIMIT = 128 * 1024 * 1024
TOTAL_LIMIT = 384 * 1024 * 1024
RECEIPT_LIMIT = 1024 * 1024
PAYLOADS = MappingProxyType(
    {
        "request": "request.json",
        "policy": "inputs/policy.json",
        "baseline": "records/baseline.json",
        "subject": "records/subject.json",
        "report": "reports/evaluation.report.json",
    }
)
DIRECTORIES = ("inputs", "records", "reports")


class CapturedContractError(ValueError):
    """A fixed contract violation, optionally with a safely examined manifest."""

    exit_code = 4
    manifest_bytes: bytes | None = None


class CapturedIntegrityError(CapturedContractError):
    """An authenticated binding failed or a source changed during capture."""

    exit_code = 6


def sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def checksums(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(files.items())
    ).encode("ascii")


def file_limit(name: str) -> int:
    if name == "request.json":
        return REQUEST_LIMIT
    return PAYLOAD_LIMIT if name in PAYLOADS.values() else CONTROL_LIMIT


def check_sizes(files: Mapping[str, bytes]) -> None:
    total = 0
    for name, raw in files.items():
        if len(raw) > file_limit(name):
            raise CapturedContractError(f"{name} exceeds its byte limit")
        total += len(raw)
        if total > TOTAL_LIMIT:
            raise CapturedContractError("captured inventory exceeds total byte limit")


def json_object(raw: bytes, label: str, *, canonical: bool = True) -> dict[str, Any]:
    try:
        value = parse_json_bytes(raw, label=label)
        if not isinstance(value, dict):
            raise CapturedContractError(f"{label} must be a JSON object")
        if canonical and canonical_json_bytes(value) != raw:
            raise CapturedContractError(f"{label} must be canonical JSON")
        return value
    except (StrictJsonError, EvidencePackError, RecursionError, UnicodeError) as exc:
        raise CapturedContractError(f"{label} is not strict JSON") from exc


@lru_cache(maxsize=2)
def _validator(receipt: bool) -> Draft202012Validator:
    return Draft202012Validator(
        load_evidence_verification_receipt_v3_schema()
        if receipt
        else load_evidence_pack_v2_schema()
    )


def validate_contract(value: dict[str, Any], *, receipt: bool = False) -> None:
    if not receipt and (
        not isinstance(value.get("files"), dict) or set(value["files"]) != set(PAYLOADS)
    ):
        raise CapturedContractError("captured manifest role inventory is invalid")
    error = next(_validator(receipt).iter_errors(value), None)
    if error is not None:
        # JSON-schema diagnostics can quote whole attacker-controlled subtrees.
        raise CapturedContractError(
            f"captured contract is invalid: {error.message[:240]}"
        )


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextmanager
def _directory_descriptor(
    path: str, flags: int, *, dir_fd: int | None = None
) -> Iterator[int]:
    descriptor = os.open(path, flags, dir_fd=dir_fd)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def secure_directory(path: Path, *, create: bool = False) -> Iterator[int]:
    """Pin every directory component; never resolve away a submitted symlink."""
    path = Path(path).absolute()
    if ".." in path.parts:
        raise CapturedContractError("directory must not contain parent traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    bindings: list[tuple[int, str, tuple[int, ...]]] = []
    with ExitStack() as descriptors:
        current_descriptor = descriptors.enter_context(
            _directory_descriptor(path.anchor, flags)
        )
        for name in path.parts[1:]:
            parent = current_descriptor
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=parent)
                except FileExistsError:
                    pass
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise CapturedContractError("path must use non-symlink directories")
            try:
                child = descriptors.enter_context(
                    _directory_descriptor(name, flags, dir_fd=parent)
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENOENT}:
                    raise CapturedIntegrityError(
                        "directory changed while opening"
                    ) from exc
                raise
            current_descriptor = child
            identity = _identity(before)[:3]
            if identity != _identity(os.fstat(child))[:3]:
                raise CapturedIntegrityError("directory changed while opening")
            bindings.append((parent, name, identity))
        try:
            yield current_descriptor
        finally:
            for parent, name, identity in bindings:
                try:
                    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError as exc:
                    raise CapturedIntegrityError(
                        "directory source was replaced"
                    ) from exc
                if _identity(current)[:3] != identity:
                    raise CapturedIntegrityError("directory source was replaced")


def _read_at(parent: int, name: str, limit: int) -> tuple[bytes, tuple[int, ...]]:
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise CapturedContractError(f"{name} must be a non-symlink regular file")
    if before.st_size > limit:
        raise CapturedContractError(f"{name} exceeds the {limit}-byte limit")
    identity = _identity(before)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
            raise CapturedIntegrityError(f"{name} changed while opening") from exc
        raise
    try:
        if _identity(os.fstat(descriptor)) != identity:
            raise CapturedIntegrityError(f"{name} changed while opening")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise CapturedContractError(f"{name} exceeds the {limit}-byte limit")
        if _identity(os.fstat(descriptor)) != identity:
            raise CapturedIntegrityError(f"{name} changed while reading")
        try:
            after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise CapturedIntegrityError(f"{name} changed while reading") from exc
        if _identity(after) != identity or len(raw) != before.st_size:
            raise CapturedIntegrityError(f"{name} changed while reading")
        return raw, identity
    finally:
        os.close(descriptor)


def read_file(path: Path, limit: int) -> bytes:
    with secure_directory(Path(path).parent) as parent:
        return _read_at(parent, Path(path).name, limit)[0]


def detect_manifest(raw: bytes) -> dict[str, Any]:
    if len(raw) > DETECTOR_LIMIT:
        raise CapturedContractError("manifest exceeds detector byte limit")
    value = json_object(raw, "evidence manifest", canonical=False)
    if value.get("format") != CAPTURED_PACK_FORMAT or value.get("kind") != "captured":
        raise CapturedContractError("captured manifest format or kind is invalid")
    if len(raw) > CONTROL_LIMIT:
        raise CapturedContractError("captured manifest exceeds control byte limit")
    return value


def _open_snapshot_directory(root: int, name: str) -> int:
    try:
        return os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=root,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
            raise CapturedIntegrityError(
                "captured directory source was replaced"
            ) from exc
        raise


def _inventory(root: int, signed: bool) -> dict[str, tuple[int, ...]]:
    expected = {"manifest.json", "checksums.sha256", *PAYLOADS.values()}
    if signed:
        expected.add("manifest.signature.json")
    found: dict[str, tuple[int, ...]] = {}
    total = 0
    # This is deliberately not recursive. Unknown directories are never opened.
    for directory in ("", *DIRECTORIES):
        descriptor = (
            root if not directory else _open_snapshot_directory(root, directory)
        )
        try:
            if directory and _identity(os.fstat(descriptor)) != found[directory]:
                raise CapturedIntegrityError("captured directory changed during scan")
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    relative = f"{directory}/{entry.name}" if directory else entry.name
                    is_directory = relative in DIRECTORIES
                    if relative in found:
                        raise CapturedIntegrityError(
                            "captured inventory repeated an entry"
                        )
                    if relative not in expected and not is_directory:
                        raise CapturedContractError(
                            "captured evidence inventory has an unknown entry"
                        )
                    try:
                        value = entry.stat(follow_symlinks=False)
                    except FileNotFoundError as exc:
                        raise CapturedIntegrityError(
                            "captured inventory changed during scan"
                        ) from exc
                    if not (
                        stat.S_ISDIR(value.st_mode)
                        if is_directory
                        else stat.S_ISREG(value.st_mode)
                    ):
                        raise CapturedContractError(
                            "captured inventory contains an unsafe entry"
                        )
                    found[relative] = _identity(value)
                    if not is_directory:
                        if value.st_size > file_limit(relative):
                            raise CapturedContractError(
                                f"{relative} exceeds its byte limit"
                            )
                        total += value.st_size
                        if total > TOTAL_LIMIT:
                            raise CapturedContractError(
                                "captured inventory exceeds total byte limit"
                            )
            if not directory and not set(DIRECTORIES).issubset(found):
                raise CapturedContractError(
                    "captured evidence directory inventory is incomplete"
                )
        finally:
            if directory:
                os.close(descriptor)
    if set(found) != expected | set(DIRECTORIES):
        raise CapturedContractError("captured evidence file inventory is incomplete")
    return found


@dataclass(frozen=True)
class CapturedSnapshot:
    files: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if any(type(raw) is not bytes for raw in self.files.values()):
            raise CapturedContractError("snapshot contents must be immutable bytes")
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    @property
    def manifest_bytes(self) -> bytes:
        return self.files["manifest.json"]


@contextmanager
def captured_snapshot(pack: Path) -> Iterator[CapturedSnapshot]:
    with secure_directory(pack) as root:
        try:
            raw, detector_identity = _read_at(root, "manifest.json", DETECTOR_LIMIT)
        except FileNotFoundError as exc:
            raise CapturedContractError("captured manifest is missing") from exc
        detect_manifest(raw)
        try:
            manifest = json_object(raw, "captured manifest")
            validate_contract(manifest)
            signed = manifest["authentication"] == "signed"
            inventory = _inventory(root, signed)
            if inventory["manifest.json"] != detector_identity:
                raise CapturedIntegrityError("manifest changed after detection")
            files: dict[str, bytes] = {}
            total = 0
            for name in inventory:
                if name in DIRECTORIES:
                    continue
                path = Path(name)
                parent = (
                    root
                    if path.parent == Path(".")
                    else _open_snapshot_directory(root, str(path.parent))
                )
                try:
                    if (
                        parent != root
                        and _identity(os.fstat(parent)) != inventory[str(path.parent)]
                    ):
                        raise CapturedIntegrityError(
                            "captured directory source was replaced"
                        )
                    try:
                        payload, identity = _read_at(
                            parent,
                            path.name,
                            min(file_limit(name), TOTAL_LIMIT - total),
                        )
                    except CapturedIntegrityError:
                        raise
                    except (CapturedContractError, FileNotFoundError) as exc:
                        raise CapturedIntegrityError(
                            f"{name} changed during snapshot"
                        ) from exc
                finally:
                    if parent != root:
                        os.close(parent)
                if identity != inventory[name]:
                    raise CapturedIntegrityError(f"{name} changed during snapshot")
                files[name] = payload
                total += len(payload)
            if files["manifest.json"] != raw:
                raise CapturedIntegrityError("manifest changed after detection")
            snapshot = CapturedSnapshot(MappingProxyType(files))
            yield snapshot
            try:
                stable = _inventory(root, signed) == inventory
            except CapturedContractError as exc:
                raise CapturedIntegrityError(
                    "captured inventory changed after capture"
                ) from exc
            if not stable:
                raise CapturedIntegrityError("captured inventory changed after capture")
        except CapturedContractError as exc:
            # Rejections identify bounded bytes examined, not an authenticated pack.
            # Never attach an identity after a manifest/root replacement or I/O error.
            try:
                current, identity = _read_at(root, "manifest.json", CONTROL_LIMIT)
            except (CapturedContractError, FileNotFoundError) as changed:
                raise CapturedIntegrityError(
                    "manifest changed after detection"
                ) from changed
            if current != raw or identity != detector_identity:
                raise CapturedIntegrityError(
                    "manifest changed after detection"
                ) from exc
            exc.manifest_bytes = raw
            raise


def manifest_signature(raw: bytes, key: ed25519.Ed25519PrivateKey) -> dict[str, Any]:
    public = key.public_key()
    return {
        "format": "invarlock/evidence-pack-signature-v1",
        "algorithm": "ed25519",
        "signing_key_fingerprint": public_key_fingerprint(public),
        "public_key": {
            "encoding": "pem",
            "value": public.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii"),
        },
        "signature": {
            "encoding": "base64",
            "value": base64.b64encode(key.sign(raw)).decode("ascii"),
        },
    }


def authenticate_manifest(
    manifest: dict[str, Any], raw: bytes, signature_raw: bytes
) -> str:
    signature = json_object(signature_raw, "manifest signature")
    if set(signature) != {
        "format",
        "algorithm",
        "signing_key_fingerprint",
        "public_key",
        "signature",
    }:
        raise CapturedContractError(
            "captured signature fields are invalid: "
            + ", ".join(sorted(signature))[:240]
        )
    if (
        signature["format"] != "invarlock/evidence-pack-signature-v1"
        or signature["algorithm"] != "ed25519"
    ):
        raise CapturedContractError("captured signature format or algorithm is invalid")
    for field, encoding in (("public_key", "pem"), ("signature", "base64")):
        block = signature[field]
        if (
            not isinstance(block, dict)
            or set(block) != {"encoding", "value"}
            or block["encoding"] != encoding
            or not isinstance(block["value"], str)
        ):
            raise CapturedContractError("captured signature encoding is invalid")
    try:
        public = serialization.load_pem_public_key(
            signature["public_key"]["value"].encode("ascii")
        )
        if not isinstance(public, ed25519.Ed25519PublicKey):
            raise ValueError("manifest signer must be Ed25519")
        public.verify(
            base64.b64decode(signature["signature"]["value"], validate=True), raw
        )
    except (ValueError, TypeError, InvalidSignature) as exc:
        raise CapturedIntegrityError("captured manifest signature is invalid") from exc
    fingerprint = public_key_fingerprint(public)
    if (
        signature["signing_key_fingerprint"] != fingerprint
        or manifest["signing_key_fingerprint"] != fingerprint
    ):
        raise CapturedIntegrityError("captured manifest signer binding is invalid")
    return fingerprint


def load_payloads(
    snapshot: CapturedSnapshot,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    files = snapshot.files
    manifest = json_object(snapshot.manifest_bytes, "captured manifest")
    signer = "Unsigned local evidence"
    if manifest["authentication"] == "signed":
        signer = authenticate_manifest(
            manifest, snapshot.manifest_bytes, files["manifest.signature.json"]
        )
    values = {}
    for role, name in PAYLOADS.items():
        if sha(files[name]) != manifest["files"][role]["digest"]:
            raise CapturedIntegrityError(f"captured {role} binding is invalid")
        values[role] = json_object(files[name], role, canonical=False)
    # Reject both oversized schedules before encoding or schema diagnostics for
    # any payload. The directory pack replaces the former nested-record boundary.
    try:
        for role in ("baseline", "subject"):
            check_record_counts(
                values[role], "run", max_records=record_contracts.MAX_RECORDS
            )
    except ValueError as exc:
        raise CapturedContractError(str(exc)) from exc
    for role, name in PAYLOADS.items():
        try:
            canonical = canonical_json_bytes(values[role])
        except (EvidencePackError, RecursionError) as exc:
            raise CapturedContractError(f"{role} is not strict JSON") from exc
        if canonical != files[name]:
            raise CapturedContractError(f"{role} must be canonical JSON")
        if role != "request":
            try:
                validate(
                    values[role],
                    {
                        "baseline": "run",
                        "subject": "run",
                        "report": "comparison",
                        "policy": "policy",
                    }[role],
                )
            except EvaluationRecordsError as exc:
                raise CapturedContractError(str(exc)[:240]) from exc
    ledger = files["checksums.sha256"]
    if hashlib.sha256(ledger).hexdigest() != manifest["checksums_sha256_digest"]:
        raise CapturedIntegrityError("captured checksum ledger binding is invalid")
    if ledger != checksums({name: files[name] for name in PAYLOADS.values()}):
        raise CapturedIntegrityError("captured checksum ledger inventory is invalid")
    from invarlock.captured_normalization import (
        captured_comparison_id,
        captured_request_digest,
        normalize_captured_request,
    )

    if manifest["request_digest"] != manifest["files"]["request"]["digest"] or manifest[
        "comparison_id"
    ] != captured_comparison_id(
        request_digest=manifest["request_digest"],
        baseline_run_digest=manifest["files"]["baseline"]["digest"],
        subject_run_digest=manifest["files"]["subject"]["digest"],
        policy_digest=manifest["files"]["policy"]["digest"],
    ):
        raise CapturedIntegrityError(
            "captured request or comparison identity is invalid"
        )
    # A signed collection of individually valid documents must also agree about
    # which inputs and source intent produced its recorded comparison.
    normalized = values["request"]
    try:
        captured_request_digest(normalized)
    except EvaluationRecordsError as exc:
        raise CapturedContractError(str(exc)[:240]) from exc
    authored = {
        "format_version": normalized["format"],
        "execution": normalized["execution"],
        "comparison": {
            **{
                side: {
                    **{
                        key: value
                        for key, value in normalized["comparison"][side].items()
                        if key != "run_digest"
                    },
                    "path": PAYLOADS[side],
                }
                for side in ("baseline", "subject")
            },
            "policy": PAYLOADS["policy"],
        },
        "output": {"evidence": "evidence"},
    }
    try:
        expected_request = normalize_captured_request(
            authored,
            baseline=values["baseline"],
            subject=values["subject"],
            policy=values["policy"],
        )
    except EvaluationRecordsError as exc:
        raise CapturedIntegrityError(str(exc)[:240]) from exc
    if canonical_json_bytes(expected_request) != files[PAYLOADS["request"]]:
        raise CapturedIntegrityError("captured request contradicts its bound inputs")
    expected_bindings = {
        "baseline": sha(files[PAYLOADS["baseline"]]),
        "subject": sha(files[PAYLOADS["subject"]]),
        "policy": sha(files[PAYLOADS["policy"]]),
    }
    if values["report"]["bindings"] != expected_bindings:
        raise CapturedIntegrityError("captured comparison contradicts its bound inputs")
    return manifest, values, signer


def require_outside(path: Path, pack: Path) -> None:
    if path.absolute().is_relative_to(pack.absolute()) or path.resolve().is_relative_to(
        pack.resolve()
    ):
        raise CapturedContractError("destination must be outside the evidence pack")


def atomic_write(path: Path, raw: bytes) -> None:
    """Publish a fully fsynced private file with a descriptor-relative hard link."""
    path = Path(path)
    if path.name in {"", ".", ".."}:
        raise CapturedContractError("destination must name a file")
    cleanup = None
    published = False
    identity = None
    try:
        with secure_directory(path.parent, create=True) as parent:
            cleanup = os.dup(parent)
            name = ".captured-" + secrets.token_hex(16)
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            try:
                try:
                    handle = os.fdopen(descriptor, "wb")
                except BaseException:
                    os.close(descriptor)
                    raise
                with handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                    identity = _identity(os.fstat(handle.fileno()))[:3]
                os.link(
                    name,
                    path.name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
                published = True
                os.fsync(parent)
            finally:
                os.unlink(name, dir_fd=parent)
    except BaseException:
        # Keep the pinned parent open through its final pathname stability check.
        if published and cleanup is not None:
            try:
                current = os.stat(path.name, dir_fd=cleanup, follow_symlinks=False)
                if _identity(current)[:3] == identity:
                    os.unlink(path.name, dir_fd=cleanup)
            except FileNotFoundError:
                pass
        raise
    finally:
        if cleanup is not None:
            os.close(cleanup)
