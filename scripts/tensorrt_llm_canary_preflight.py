#!/usr/bin/env python3
"""Authenticate TensorRT-LLM canary inputs before starting a GPU container."""

from __future__ import annotations

import argparse
import os
import re
import stat
from pathlib import Path

from invarlock_addins.tensorrt_llm.execution import TensorRTLLMExecutionError
from invarlock_addins.tensorrt_llm.inspection import (
    authenticate_tensorrt_llm_tokenizer_contract,
)

from invarlock.runtime_providers.tensorrt_llm_identity import (
    TensorRTLLMIdentityError,
    read_tensorrt_llm_engine_tree_sha256,
)

_IMAGE_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_POSITIVE_INTEGER = re.compile(r"^[0-9]+$")
_OCI_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*"
_OCI_REFERENCE = re.compile(
    rf"^(?:{_OCI_COMPONENT}(?::[1-9][0-9]*)?/)*{_OCI_COMPONENT}$"
)
_MAX_TOKENIZER_CONTRACT_BYTES = 128 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


class CanaryPreflightError(ValueError):
    """Raised when a deterministic host input cannot reach the GPU canary."""


def _portable_relative_path(value: str, *, label: str) -> tuple[str, ...]:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise CanaryPreflightError(f"{label} must be a portable relative path")
    parts = tuple(value.split("/"))
    if (
        value.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or ":/" in value
        or (len(value) >= 2 and value[1] == ":")
    ):
        raise CanaryPreflightError(f"{label} must be a portable relative path")
    return parts


def _validate_image_reference(image: str, image_digest: str) -> None:
    if _IMAGE_DIGEST.fullmatch(image_digest) is None:
        raise CanaryPreflightError(
            "IMAGE_DIGEST must be a canonical sha256 image digest"
        )
    if image == image_digest:
        return
    if (
        not image
        or len(image) > 512
        or image.count("@") != 1
        or "," in image
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in image
        )
    ):
        raise CanaryPreflightError(
            "IMAGE must be an exact local digest or canonical OCI digest reference"
        )
    reference, embedded_digest = image.split("@", maxsplit=1)
    if embedded_digest != image_digest or _OCI_REFERENCE.fullmatch(reference) is None:
        raise CanaryPreflightError(
            "IMAGE must be an exact local digest or canonical OCI digest reference"
        )


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise CanaryPreflightError(
            "secure nofollow input authentication is unavailable on this platform"
        )
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW


def _canonical_input_root(value: str) -> Path:
    if "," in value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise CanaryPreflightError(
            "INPUT_ROOT must not contain commas or control characters"
        )
    supplied = Path(value)
    if not supplied.is_absolute():
        raise CanaryPreflightError("INPUT_ROOT must be an absolute path")
    if ".." in supplied.parts:
        raise CanaryPreflightError("INPUT_ROOT must not contain traversal components")
    try:
        canonical = Path(os.path.abspath(supplied))
    except (TypeError, ValueError, OSError) as exc:
        raise CanaryPreflightError("INPUT_ROOT is invalid") from exc
    flags = _directory_flags()
    try:
        descriptor = os.open(canonical.anchor, flags)
    except OSError as exc:
        raise CanaryPreflightError(
            "INPUT_ROOT must be an existing non-symlink directory"
        ) from exc
    try:
        for component in canonical.parts[1:]:
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                opened = os.fstat(next_descriptor)
            except OSError as exc:
                if isinstance(exc, FileNotFoundError):
                    raise CanaryPreflightError(
                        "INPUT_ROOT must be an existing non-symlink directory"
                    ) from exc
                raise CanaryPreflightError(
                    "INPUT_ROOT contains a symlink or inaccessible directory"
                ) from exc
            if (
                not stat.S_ISDIR(before.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                os.close(next_descriptor)
                raise CanaryPreflightError(
                    "INPUT_ROOT directory changed while being authenticated"
                )
            os.close(descriptor)
            descriptor = next_descriptor
    finally:
        os.close(descriptor)
    return canonical


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_tokenizer_contract(
    root: Path,
    relative_path: str,
) -> bytes:
    parts = _portable_relative_path(relative_path, label="TOKENIZER_CONTRACT")
    directory_descriptor = os.open(root, _directory_flags())
    try:
        for component in parts[:-1]:
            next_descriptor = os.open(
                component,
                _directory_flags(),
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        try:
            named = os.stat(
                parts[-1],
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | os.O_NOFOLLOW
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise CanaryPreflightError(
                "TOKENIZER_CONTRACT must exist beneath INPUT_ROOT without symbolic links"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _stat_identity(named) != _stat_identity(opened)
            ):
                raise CanaryPreflightError(
                    "TOKENIZER_CONTRACT must be a stable regular file"
                )
            if not 0 <= opened.st_size <= _MAX_TOKENIZER_CONTRACT_BYTES:
                raise CanaryPreflightError(
                    "TOKENIZER_CONTRACT exceeds the configured size bound"
                )
            remaining = opened.st_size
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
                if not chunk:
                    raise CanaryPreflightError(
                        "TOKENIZER_CONTRACT changed while being read"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            named_after = os.stat(
                parts[-1],
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                os.read(descriptor, 1)
                or _stat_identity(after) != _stat_identity(opened)
                or _stat_identity(named_after) != _stat_identity(opened)
            ):
                raise CanaryPreflightError(
                    "TOKENIZER_CONTRACT changed while being authenticated"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise CanaryPreflightError(
            "TOKENIZER_CONTRACT must exist beneath INPUT_ROOT without symbolic links"
        ) from exc
    finally:
        os.close(directory_descriptor)


def validate(
    *,
    image: str,
    image_digest: str,
    input_root: str,
    engine_bundle: str,
    tokenizer_contract: str,
    expected_engine_tree_sha256: str,
    expected_tokenizer_sha256: str,
    expected_output_sha256: str,
    tmpfs_gib: str,
) -> str:
    """Authenticate every deterministic host input and return the canonical root."""

    _validate_image_reference(image, image_digest)
    for label, digest in (
        ("EXPECTED_ENGINE_TREE_SHA256", expected_engine_tree_sha256),
        ("EXPECTED_TOKENIZER_SHA256", expected_tokenizer_sha256),
        ("EXPECTED_OUTPUT_SHA256", expected_output_sha256),
    ):
        if _SHA256.fullmatch(digest) is None:
            raise CanaryPreflightError(f"{label} must be a lowercase sha256 digest")
    if _POSITIVE_INTEGER.fullmatch(tmpfs_gib) is None or not (
        4 <= int(tmpfs_gib) <= 64
    ):
        raise CanaryPreflightError("CANARY_TMPFS_GIB must be an integer from 4 to 64")

    canonical_root = _canonical_input_root(input_root)
    engine_parts = _portable_relative_path(engine_bundle, label="ENGINE_BUNDLE")
    engine_path = canonical_root.joinpath(*engine_parts)
    try:
        observed_engine_tree_sha256 = read_tensorrt_llm_engine_tree_sha256(engine_path)
    except TensorRTLLMIdentityError as exc:
        raise CanaryPreflightError(
            "ENGINE_BUNDLE engine identity cannot be authenticated: " + str(exc)
        ) from exc
    if observed_engine_tree_sha256 != expected_engine_tree_sha256:
        raise CanaryPreflightError(
            "ENGINE_BUNDLE engine bundle does not match EXPECTED_ENGINE_TREE_SHA256"
        )

    tokenizer_payload = _read_tokenizer_contract(canonical_root, tokenizer_contract)
    try:
        observed_tokenizer_sha256 = authenticate_tensorrt_llm_tokenizer_contract(
            tokenizer_payload
        )
    except TensorRTLLMExecutionError as exc:
        raise CanaryPreflightError(str(exc)) from exc
    if observed_tokenizer_sha256 != expected_tokenizer_sha256:
        raise CanaryPreflightError(
            "TOKENIZER_CONTRACT tokenizer contract does not match EXPECTED_TOKENIZER_SHA256"
        )
    return str(canonical_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--engine-bundle", required=True)
    parser.add_argument("--tokenizer-contract", required=True)
    parser.add_argument("--expected-engine-tree-sha256", required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--expected-output-sha256", required=True)
    parser.add_argument("--tmpfs-gib", required=True)
    arguments = parser.parse_args(argv)
    try:
        canonical_root = validate(
            image=arguments.image,
            image_digest=arguments.image_digest,
            input_root=arguments.input_root,
            engine_bundle=arguments.engine_bundle,
            tokenizer_contract=arguments.tokenizer_contract,
            expected_engine_tree_sha256=arguments.expected_engine_tree_sha256,
            expected_tokenizer_sha256=arguments.expected_tokenizer_sha256,
            expected_output_sha256=arguments.expected_output_sha256,
            tmpfs_gib=arguments.tmpfs_gib,
        )
    except CanaryPreflightError as exc:
        parser.error(str(exc))
    print(canonical_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
