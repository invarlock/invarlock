#!/usr/bin/env python3
"""Build one runtime image from an authenticated Git archive context."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path

try:
    from scripts.qualification_source import authenticate_bundle
except ImportError:  # pragma: no cover - direct script execution
    from qualification_source import (  # type: ignore[import-not-found, no-redef]
        authenticate_bundle,
    )

_BUILD_ARGUMENT = re.compile(r"^[A-Z][A-Z0-9_]{0,127}=.*$", re.DOTALL)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLATFORM = re.compile(r"^linux/[a-z0-9_]+(?:/[a-z0-9_]+)?$")
_OCI_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*"
_OCI_REPOSITORY = rf"(?:{_OCI_COMPONENT}(?::[1-9][0-9]*)?/)*{_OCI_COMPONENT}"
_OCI_TAG = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"
_BUILD_IMAGE_TAG = re.compile(rf"^{_OCI_REPOSITORY}:{_OCI_TAG}$")
_DOCKERFILE_BASE_IMAGE = re.compile(
    rf"^{_OCI_REPOSITORY}(?::{_OCI_TAG})?@sha256:[0-9a-f]{{64}}$"
)
_MAX_INSPECT_BYTES = 1024 * 1024
_MAX_DOCKERFILE_BYTES = 1024 * 1024
_SOURCE_LABELS = {
    "commit": "org.opencontainers.image.revision",
    "bundle": "dev.invarlock.source-bundle-sha256",
}
_RESERVED_BUILD_ARGUMENTS = {
    "INVARLOCK_SOURCE_BUNDLE_SHA256",
    "INVARLOCK_SOURCE_COMMIT",
}
_DOCKERFILE_BASE_ARGUMENTS = {
    "LLAMA_CPP_BUILD_BASE",
    "RUNTIME_BASE_IMAGE",
    "RUNTIME_BUILD_BASE_IMAGE",
    "WHEEL_BUILD_BASE",
}


def _engine(value: str) -> str:
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise SystemExit("container engine is invalid")
    executable = shutil.which(value)
    if executable is None:
        raise SystemExit("container engine is unavailable")
    return str(Path(executable).resolve(strict=True))


def _relative_dockerfile(value: str) -> str:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value != path.as_posix()
    ):
        raise SystemExit("runtime Dockerfile must be a canonical archive-relative path")
    return value


def _dockerfile_base_image(value: str, *, label: str) -> str:
    if (
        not value
        or len(value) > 512
        or value != value.strip()
        or _DOCKERFILE_BASE_IMAGE.fullmatch(value) is None
    ):
        raise SystemExit(f"{label} must use a canonical repository@sha256 reference")
    return value


def _build_image_tag(value: str) -> str:
    if (
        not value
        or len(value) > 255
        or value != value.strip()
        or _DIGEST.fullmatch(value) is not None
        or _BUILD_IMAGE_TAG.fullmatch(value) is None
    ):
        raise SystemExit("runtime image name must be a canonical repository tag")
    return value


def _build_arguments(values: list[str]) -> list[str]:
    observed: set[str] = set()
    normalized: list[str] = []
    for value in values:
        if _BUILD_ARGUMENT.fullmatch(value) is None or any(
            ord(character) < 32 for character in value
        ):
            raise SystemExit("runtime build argument is invalid")
        name = value.partition("=")[0]
        if name in _RESERVED_BUILD_ARGUMENTS:
            raise SystemExit("source identity build arguments are driver-owned")
        if name in observed:
            raise SystemExit(f"runtime build argument {name!r} is repeated")
        if name in _DOCKERFILE_BASE_ARGUMENTS:
            _dockerfile_base_image(
                value.partition("=")[2], label=f"runtime build argument {name}"
            )
        observed.add(name)
        normalized.append(value)
    return normalized


def _inspect_payload(raw: bytes) -> dict[str, object]:
    if len(raw) > _MAX_INSPECT_BYTES:
        raise SystemExit("container image inspection exceeds the size limit")

    def reject_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise SystemExit("container image inspection contains a duplicate key")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit("container image inspection is not strict JSON") from exc
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise SystemExit("container image inspection must identify one image")
        return payload[0]
    if not isinstance(payload, dict):
        raise SystemExit("container image inspection must identify one image")
    return payload


def _image_metadata(engine: str, image: str) -> tuple[dict[str, str], tuple[str, ...]]:
    completed = subprocess.run(
        [engine, "image", "inspect", image],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[:512].strip()
        raise SystemExit(
            f"container image {image!r} is unavailable for source-label verification"
            + (f": {diagnostic}" if diagnostic else "")
        )
    payload = _inspect_payload(completed.stdout)
    config = payload.get("Config", payload.get("config"))
    if not isinstance(config, dict):
        raise SystemExit("container image inspection is missing Config")
    labels = config.get("Labels", config.get("labels"))
    if not isinstance(labels, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in labels.items()
    ):
        raise SystemExit("container image inspection is missing source labels")
    rootfs = payload.get("RootFS", payload.get("rootfs"))
    layers = (
        rootfs.get("Layers", rootfs.get("layers")) if isinstance(rootfs, dict) else None
    )
    if not isinstance(layers, list) or any(
        not isinstance(layer, str) or _DIGEST.fullmatch(layer) is None
        for layer in layers
    ):
        raise SystemExit("container image inspection is missing filesystem layers")
    return labels, tuple(layers)


def _require_source_labels(
    *, engine: str, image: str, source_commit: str, source_bundle_sha256: str
) -> tuple[str, ...]:
    labels, layers = _image_metadata(engine, image)
    expected = {
        _SOURCE_LABELS["commit"]: source_commit,
        _SOURCE_LABELS["bundle"]: source_bundle_sha256,
    }
    for name, value in expected.items():
        if labels.get(name) != value:
            raise SystemExit(
                f"container image {image!r} source label {name!r} does not match"
            )
    return layers


def _built_image_identity(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SystemExit("container build did not record its image identity") from exc
    if len(payload) > 128:
        raise SystemExit("container build recorded an invalid image identity")
    try:
        image = payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise SystemExit("container build recorded an invalid image identity") from exc
    if _DIGEST.fullmatch(image) is None:
        raise SystemExit("container build recorded an invalid image identity")
    return image


def _write_build_statement(path: Path, payload: dict[str, object]) -> None:
    """Publish one private, canonical, no-clobber build statement."""

    destination = Path(os.path.abspath(os.fspath(path)))
    try:
        parent = destination.parent.resolve(strict=True)
    except OSError as exc:
        raise SystemExit("runtime build-statement parent is unavailable") from exc
    if parent != destination.parent or not parent.is_dir() or parent.is_symlink():
        raise SystemExit("runtime build-statement parent must be one real directory")
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = -1
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise SystemExit("runtime build-statement destination already exists") from exc
    except OSError as exc:
        raise SystemExit("runtime build statement could not be published") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--container-engine", default="docker")
    parser.add_argument("--dockerfile", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--platform")
    parser.add_argument("--build-arg", action="append", default=[])
    parser.add_argument("--require-base-source-labels")
    parser.add_argument("--statement", type=Path)
    arguments = parser.parse_args(argv)

    result, source_context = authenticate_bundle(
        repository=arguments.repository,
        commit=arguments.source_commit,
        bundle=arguments.source_bundle,
        bundle_sha256=arguments.source_bundle_sha256,
    )
    dockerfile = _relative_dockerfile(arguments.dockerfile)
    image = _build_image_tag(arguments.image)
    base_image = (
        None
        if arguments.require_base_source_labels is None
        else _dockerfile_base_image(
            arguments.require_base_source_labels, label="base runtime image"
        )
    )
    build_arguments = _build_arguments(arguments.build_arg)
    normalized_arguments = {
        value.partition("=")[0]: value.partition("=")[2] for value in build_arguments
    }
    if (
        base_image is not None
        and normalized_arguments.get("RUNTIME_BASE_IMAGE") != base_image
    ):
        raise SystemExit(
            "authenticated base runtime image must match the RUNTIME_BASE_IMAGE "
            "build argument"
        )
    if (
        arguments.platform is not None
        and _PLATFORM.fullmatch(arguments.platform) is None
    ):
        raise SystemExit("runtime build platform is invalid")

    with tarfile.open(fileobj=io.BytesIO(source_context), mode="r:") as archive:
        try:
            dockerfile_member = archive.getmember(dockerfile)
        except KeyError as exc:
            raise SystemExit("runtime Dockerfile is absent from source bundle") from exc
        if not dockerfile_member.isfile():
            raise SystemExit("runtime Dockerfile is not a regular source-bundle file")
        if dockerfile_member.size > _MAX_DOCKERFILE_BYTES:
            raise SystemExit("runtime Dockerfile exceeds the size limit")
        extracted_dockerfile = archive.extractfile(dockerfile_member)
        if extracted_dockerfile is None:
            raise SystemExit("runtime Dockerfile is unreadable")
        dockerfile_bytes = extracted_dockerfile.read(_MAX_DOCKERFILE_BYTES + 1)
        if len(dockerfile_bytes) > _MAX_DOCKERFILE_BYTES:
            raise SystemExit("runtime Dockerfile exceeds the size limit")
    engine = _engine(arguments.container_engine)
    base_layers: tuple[str, ...] | None = None
    if base_image is not None:
        base_layers = _require_source_labels(
            engine=engine,
            image=base_image,
            source_commit=arguments.source_commit,
            source_bundle_sha256=arguments.source_bundle_sha256,
        )

    with tempfile.TemporaryDirectory(prefix="invarlock-runtime-build-") as temporary:
        image_identity = Path(temporary) / "image-id"
        command = [engine, "build", "--iidfile", str(image_identity)]
        if arguments.platform is not None:
            command.extend(("--platform", arguments.platform))
        for value in build_arguments:
            command.extend(("--build-arg", value))
        command.extend(
            (
                "--build-arg",
                f"INVARLOCK_SOURCE_COMMIT={result['source_commit']}",
                "--build-arg",
                f"INVARLOCK_SOURCE_BUNDLE_SHA256={result['source_bundle_sha256']}",
                "--file",
                dockerfile,
                "--tag",
                image,
                "-",
            )
        )
        completed = subprocess.run(command, check=False, input=source_context)
        if completed.returncode != 0:
            return completed.returncode
        built_image = _built_image_identity(image_identity)
        built_layers = _require_source_labels(
            engine=engine,
            image=built_image,
            source_commit=arguments.source_commit,
            source_bundle_sha256=arguments.source_bundle_sha256,
        )
        if base_layers is not None and (
            not base_layers or built_layers[: len(base_layers)] != base_layers
        ):
            raise SystemExit(
                "built runtime image filesystem does not derive from the "
                "authenticated base image"
            )
        if arguments.statement is not None:
            _write_build_statement(
                arguments.statement,
                {
                    "base_image": base_image,
                    "build_arguments": dict(sorted(normalized_arguments.items())),
                    "dockerfile": {
                        "path": dockerfile,
                        "sha256": "sha256:"
                        + hashlib.sha256(dockerfile_bytes).hexdigest(),
                    },
                    "format_version": "invarlock/runtime-image-build-v1",
                    "image": image,
                    "ok": True,
                    "platform": arguments.platform,
                    "runtime_image_id": built_image,
                    "source_bundle_sha256": arguments.source_bundle_sha256,
                    "source_commit": arguments.source_commit,
                },
            )
    return 0


def main() -> int:
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
