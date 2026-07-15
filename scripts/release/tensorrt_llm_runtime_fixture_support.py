"""Security contracts for TensorRT-LLM fixture creation and promotion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Final

from invarlock.core.runtime_provider import TensorRTLLMArtifactIdentity

MODEL_REPOSITORY: Final = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION: Final = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
BACKEND_VERSION: Final = "1.2.1"
MANIFEST_FORMAT: Final = "invarlock/tensorrt-llm-runtime-fixture-v1"
QUALIFICATION_FORMAT: Final = "invarlock/tensorrt-llm-dual-gpu-qualification-v1"
TARGET_COMPUTE_CAPABILITY: Final = "9.0"
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
STABLE_TAG_RE = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r":[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$"
)
IDENTITY_KEYS: Final = frozenset(
    {
        "artifact_format",
        "builder_config_sha256",
        "bundle_name",
        "engine_bundle_tree_sha256",
        "engine_metadata_sha256",
        "file_inventory_sha256",
        "format_version",
        "target_compute_capability",
        "tokenizer_metadata_sha256",
    }
)
QUALIFICATION_KEYS: Final = frozenset(
    {
        "candidate_image_digest",
        "engine_bundle_tree_sha256",
        "format_version",
        "gpu_count",
        "ok",
        "output_sha256",
        "runtime_provider_receipt_sha256",
        "tokenizer_sha256",
    }
)
BUILD_RECIPE: Final[Mapping[str, object]] = {
    "dtype": "float16",
    "gemm_plugin": "auto",
    "max_batch_size": 1,
    "max_input_len": 8,
    "max_num_tokens": 9,
    "max_seq_len": 9,
    "opt_num_tokens": 8,
    "tensor_parallel_size": 1,
    "target_compute_capability": TARGET_COMPUTE_CAPABILITY,
}
MANIFEST_KEYS: Final = frozenset(
    {
        "backend_version",
        "build_recipe",
        "candidate_image_digest",
        "engine_builds",
        "engine_byte_reproduction",
        "expected_output_sha256",
        "format_version",
        "model",
        "selected_engine_identity",
        "tokenizer_sha256",
        "worker",
    }
)


class FixtureContractError(RuntimeError):
    """Raised when an authenticated fixture contract is invalid."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureContractError("JSON contains a duplicate object key")
        result[key] = value
    return result


def parse_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        decoded = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FixtureContractError(f"{label} contains {value}")
            ),
        )
    except FixtureContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FixtureContractError(f"{label} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise FixtureContractError(f"{label} must be a JSON object")
    return decoded


def sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size <= 0:
            raise FixtureContractError("an input must be a non-empty regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        final = os.fstat(descriptor)
    except OSError as exc:
        raise FixtureContractError("an input cannot be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
    ):
        raise FixtureContractError("an input changed while being authenticated")
    return digest.hexdigest()


def model_inventory_sha256(root: Path) -> str:
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise FixtureContractError("the local model snapshot is unavailable") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) or root.is_symlink():
        raise FixtureContractError("the local model snapshot must be a directory")
    records: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        names.sort()
        files.sort()
        directory_path = Path(directory)
        for name in names:
            if (directory_path / name).is_symlink():
                raise FixtureContractError("the model snapshot contains a symlink")
        for name in files:
            path = directory_path / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise FixtureContractError("the model snapshot contains a symlink")
            if not stat.S_ISREG(metadata.st_mode):
                raise FixtureContractError("the model snapshot contains a special file")
            inode = (metadata.st_dev, metadata.st_ino)
            if metadata.st_nlink != 1 or inode in seen:
                raise FixtureContractError("the model snapshot contains a hard link")
            seen.add(inode)
            records.append(
                {
                    "byte_length": metadata.st_size,
                    "name": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    if not records:
        raise FixtureContractError("the model snapshot is empty")
    return hashlib.sha256(
        b"invarlock/tensorrt-llm-model-inventory-v1\0" + canonical_json(records)
    ).hexdigest()


def snapshot_regular_file(source: Path, destination: Path) -> None:
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        named = source.lstat()
        if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1:
            raise FixtureContractError(
                "a snapshot input must be a single-link regular file"
            )
        source_descriptor = os.open(source, source_flags)
        opened = os.fstat(source_descriptor)
        if (named.st_dev, named.st_ino, named.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise FixtureContractError("a snapshot input changed before copying")
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        remaining = opened.st_size
        while remaining:
            block = os.read(source_descriptor, min(1024 * 1024, remaining))
            if not block:
                raise FixtureContractError("a snapshot input changed while copying")
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise FixtureContractError("an owned snapshot cannot be written")
                view = view[written:]
            remaining -= len(block)
        if os.read(source_descriptor, 1):
            raise FixtureContractError("a snapshot input grew while copying")
        final = os.fstat(source_descriptor)
        if (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        ) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ):
            raise FixtureContractError("a snapshot input changed while copying")
        os.fsync(destination_descriptor)
    except FixtureContractError:
        raise
    except OSError as exc:
        raise FixtureContractError(
            "an owned snapshot cannot be created safely"
        ) from exc
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)


def snapshot_model_tree(source: Path, destination: Path) -> None:
    try:
        metadata = source.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or source.is_symlink():
            raise FixtureContractError("the model snapshot source must be a directory")
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
        for directory, names, files in os.walk(source, followlinks=False):
            names.sort()
            files.sort()
            source_directory = Path(directory)
            destination_directory = destination / source_directory.relative_to(source)
            for name in names:
                child = source_directory / name
                child_metadata = child.lstat()
                if not stat.S_ISDIR(child_metadata.st_mode) or child.is_symlink():
                    raise FixtureContractError(
                        "the model snapshot source contains a non-directory entry"
                    )
                (destination_directory / name).mkdir(mode=0o700, exist_ok=False)
            for name in files:
                snapshot_regular_file(
                    source_directory / name,
                    destination_directory / name,
                )
    except FixtureContractError:
        raise
    except OSError as exc:
        raise FixtureContractError(
            "the model snapshot cannot be created safely"
        ) from exc


def _validated_identity_mapping(
    value: object, *, tokenizer_sha256: str, label: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != IDENTITY_KEYS:
        raise FixtureContractError(f"the {label} identity has an unexpected schema")
    for name in (
        "builder_config_sha256",
        "engine_bundle_tree_sha256",
        "engine_metadata_sha256",
        "file_inventory_sha256",
        "tokenizer_metadata_sha256",
    ):
        digest = value.get(name)
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise FixtureContractError(f"the {label} identity {name} is invalid")
    if value.get("tokenizer_metadata_sha256") != tokenizer_sha256:
        raise FixtureContractError(f"the {label} identity tokenizer binding changed")
    try:
        normalized = asdict(
            TensorRTLLMArtifactIdentity(
                bundle_name=str(value.get("bundle_name", "")),
                engine_bundle_tree_sha256=str(
                    value.get("engine_bundle_tree_sha256", "")
                ),
                file_inventory_sha256=str(value.get("file_inventory_sha256", "")),
                builder_config_sha256=str(value.get("builder_config_sha256", "")),
                tokenizer_metadata_sha256=str(
                    value.get("tokenizer_metadata_sha256", "")
                ),
                engine_metadata_sha256=str(value.get("engine_metadata_sha256", "")),
                target_compute_capability=str(
                    value.get("target_compute_capability", "")
                ),
            )
        )
    except (TypeError, ValueError) as exc:
        raise FixtureContractError(f"the {label} identity is invalid") from exc
    expected_bundle_name = (
        "tensorrt-llm-sha256-" + normalized["engine_bundle_tree_sha256"]
    )
    if normalized != value or normalized["bundle_name"] != expected_bundle_name:
        raise FixtureContractError(f"the {label} identity binding is not canonical")
    return normalized


def _read_bounded_object(path: Path, *, label: str) -> dict[str, object]:
    descriptor: int | None = None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        named = path.lstat()
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise FixtureContractError(f"the {label} is not a bounded regular file")
        if (named.st_dev, named.st_ino, named.st_size) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
        ):
            raise FixtureContractError(f"the {label} changed before reading")
        payload = os.read(descriptor, metadata.st_size + 1)
        final = os.fstat(descriptor)
        if len(payload) != metadata.st_size or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
            raise FixtureContractError(f"the {label} changed while reading")
    except OSError as exc:
        raise FixtureContractError(f"the {label} cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return parse_object(payload, label=label)


def load_manifest(path: Path) -> dict[str, object]:
    manifest = _read_bounded_object(path, label="fixture manifest")
    if (
        set(manifest) != MANIFEST_KEYS
        or manifest.get("format_version") != MANIFEST_FORMAT
    ):
        raise FixtureContractError("the fixture manifest has an unexpected schema")
    if manifest.get("backend_version") != BACKEND_VERSION:
        raise FixtureContractError("the fixture manifest backend version is invalid")
    if manifest.get("build_recipe") != BUILD_RECIPE:
        raise FixtureContractError("the fixture manifest build recipe is invalid")
    for name in ("tokenizer_sha256", "expected_output_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise FixtureContractError(f"the fixture manifest {name} is invalid")
    worker = manifest.get("worker")
    if (
        not isinstance(worker, dict)
        or set(worker) != {"sha256"}
        or not isinstance(worker.get("sha256"), str)
        or SHA256_RE.fullmatch(str(worker.get("sha256"))) is None
    ):
        raise FixtureContractError("the fixture manifest worker binding is invalid")
    image_digest = manifest.get("candidate_image_digest")
    if (
        not isinstance(image_digest, str)
        or IMAGE_DIGEST_RE.fullmatch(image_digest) is None
    ):
        raise FixtureContractError("the fixture manifest image digest is invalid")
    model = manifest.get("model")
    if (
        not isinstance(model, dict)
        or set(model) != {"inventory_sha256", "repository", "revision"}
        or model.get("repository") != MODEL_REPOSITORY
        or model.get("revision") != MODEL_REVISION
        or not isinstance(model.get("inventory_sha256"), str)
        or SHA256_RE.fullmatch(str(model.get("inventory_sha256"))) is None
    ):
        raise FixtureContractError("the fixture manifest model binding is invalid")
    tokenizer_sha256 = str(manifest["tokenizer_sha256"])
    builds = manifest.get("engine_builds")
    if not isinstance(builds, dict) or set(builds) != {"primary", "secondary"}:
        raise FixtureContractError("the fixture manifest engine builds are invalid")
    primary = _validated_identity_mapping(
        builds["primary"], tokenizer_sha256=tokenizer_sha256, label="primary"
    )
    secondary = _validated_identity_mapping(
        builds["secondary"], tokenizer_sha256=tokenizer_sha256, label="secondary"
    )
    selected = _validated_identity_mapping(
        manifest.get("selected_engine_identity"),
        tokenizer_sha256=tokenizer_sha256,
        label="selected",
    )
    if selected != primary:
        raise FixtureContractError("the selected engine is not the primary build")
    expected_reproduction = (
        "matched"
        if primary["engine_bundle_tree_sha256"]
        == secondary["engine_bundle_tree_sha256"]
        else "different"
    )
    if manifest.get("engine_byte_reproduction") != expected_reproduction:
        raise FixtureContractError(
            "the fixture manifest engine reproduction claim is invalid"
        )
    return manifest


def load_qualification_summary(path: Path) -> dict[str, object]:
    summary = _read_bounded_object(path, label="qualification summary")
    if (
        set(summary) != QUALIFICATION_KEYS
        or summary.get("format_version") != QUALIFICATION_FORMAT
        or summary.get("gpu_count") != 2
        or summary.get("ok") is not True
    ):
        raise FixtureContractError("the qualification summary has an invalid schema")
    image_digest = summary.get("candidate_image_digest")
    if (
        not isinstance(image_digest, str)
        or IMAGE_DIGEST_RE.fullmatch(image_digest) is None
    ):
        raise FixtureContractError("the qualification summary image digest is invalid")
    for name in (
        "engine_bundle_tree_sha256",
        "output_sha256",
        "runtime_provider_receipt_sha256",
        "tokenizer_sha256",
    ):
        value = summary.get(name)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise FixtureContractError(f"the qualification summary {name} is invalid")
    return summary
