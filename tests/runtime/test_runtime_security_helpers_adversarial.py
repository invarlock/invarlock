from __future__ import annotations

import os
from pathlib import Path

import pytest

import invarlock.runtime_security_helpers as helpers

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64


def test_kernel_evidence_readers_reject_non_files_and_oversized_data(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "marker"
    marker.write_text("container", encoding="utf-8")

    assert helpers._regular_file_marker_present(str(marker)) is True
    assert helpers._regular_file_marker_present(str(tmp_path)) is False
    assert helpers._regular_file_marker_present(str(tmp_path / "missing")) is False
    assert helpers._read_bounded_kernel_file(str(marker), max_bytes=9) == b"container"
    assert helpers._read_bounded_kernel_file(str(marker), max_bytes=8) is None
    assert helpers._read_bounded_kernel_file(str(tmp_path)) is None
    assert helpers._read_bounded_kernel_file(str(tmp_path / "missing")) is None


def test_kernel_file_read_failure_is_fail_closed_and_closes_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(os, "open", lambda *_args: 17)
    monkeypatch.setattr(
        os,
        "fstat",
        lambda _descriptor: type("Stat", (), {"st_mode": 0o100644})(),
    )
    monkeypatch.setattr(
        os,
        "read",
        lambda *_args: (_ for _ in ()).throw(OSError("unreadable")),
    )
    monkeypatch.setattr(os, "close", closed.append)

    assert helpers._read_bounded_kernel_file("/proc/1/cgroup") is None
    assert closed == [17]


@pytest.mark.parametrize(
    ("cgroup", "expected"),
    [
        (b"0::/system.slice/docker-abc.scope\n", True),
        (b"0::/user.slice/session.scope\n", False),
        (None, False),
    ],
)
def test_container_boundary_requires_intent_and_kernel_evidence(
    monkeypatch: pytest.MonkeyPatch,
    cgroup: bytes | None,
    expected: bool,
) -> None:
    monkeypatch.setenv(helpers.CONTAINER_EXECUTION_ENV, "true")
    monkeypatch.setattr(helpers, "_regular_file_marker_present", lambda _path: False)
    monkeypatch.setattr(helpers, "_read_bounded_kernel_file", lambda _path: cgroup)

    assert helpers.strict_container_boundary_present() is expected

    monkeypatch.delenv(helpers.CONTAINER_EXECUTION_ENV)
    assert helpers.strict_container_boundary_present() is False


def test_container_boundary_accepts_regular_runtime_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(helpers.CONTAINER_EXECUTION_ENV, "yes")
    monkeypatch.setattr(
        helpers,
        "_regular_file_marker_present",
        lambda path: path == "/run/.containerenv",
    )
    monkeypatch.setattr(
        helpers,
        "_read_bounded_kernel_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not read cgroup")),
    )

    assert helpers.strict_container_boundary_present() is True


def test_runtime_image_resolution_rejects_ambiguous_or_malformed_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(helpers.RUNTIME_IMAGE_ENV, f"registry/runtime@{_DIGEST_A}")
    monkeypatch.setenv(helpers.RUNTIME_IMAGE_DIGEST_ENV, _DIGEST_B)
    with pytest.raises(RuntimeError, match="does not match"):
        helpers.resolve_runtime_image_digest()

    monkeypatch.setenv(helpers.RUNTIME_IMAGE_DIGEST_ENV, "sha256:ABC")
    with pytest.raises(RuntimeError, match="lowercase"):
        helpers.resolve_runtime_image()

    monkeypatch.delenv(helpers.RUNTIME_IMAGE_DIGEST_ENV)
    monkeypatch.setenv(helpers.RUNTIME_IMAGE_ENV, "registry/runtime@sha256:short")
    with pytest.raises(RuntimeError, match="reference"):
        helpers.resolve_runtime_image_digest()


def test_runtime_image_resolution_preserves_or_attaches_one_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(helpers.RUNTIME_IMAGE_ENV, "registry/runtime:release")
    monkeypatch.setenv(helpers.RUNTIME_IMAGE_DIGEST_ENV, _DIGEST_A)
    assert helpers.resolve_runtime_image() == f"registry/runtime:release@{_DIGEST_A}"

    monkeypatch.setenv(helpers.RUNTIME_IMAGE_ENV, _DIGEST_A)
    assert helpers.resolve_runtime_image() == _DIGEST_A

    monkeypatch.delenv(helpers.RUNTIME_IMAGE_DIGEST_ENV)
    monkeypatch.setenv(helpers.RUNTIME_IMAGE_ENV, f"registry/runtime@{_DIGEST_A}")
    assert helpers.resolve_runtime_image_digest() == _DIGEST_A
    assert helpers.resolve_runtime_image() == f"registry/runtime@{_DIGEST_A}"


@pytest.mark.parametrize(
    ("image_ref", "digest", "message"),
    [
        ("runtime", None, "lowercase image digest"),
        ("", _DIGEST_A, "portable reference"),
        (" runtime", _DIGEST_A, "portable reference"),
        ("/runtime", _DIGEST_A, "portable reference"),
        ("C:\\runtime", _DIGEST_A, "portable reference"),
        ("runtime\nname", _DIGEST_A, "portable reference"),
        (f"runtime@{_DIGEST_A}@{_DIGEST_A}", _DIGEST_A, "portable reference"),
        (f"@{_DIGEST_A}", _DIGEST_A, "name an image"),
        ("runtime@sha256:short", _DIGEST_A, "invalid digest"),
        (f"runtime@{_DIGEST_B}", _DIGEST_A, "do not agree"),
    ],
)
def test_runtime_provenance_rejects_nonportable_or_conflicting_references(
    image_ref: str,
    digest: str | None,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        helpers._runtime_provenance_image_ref(image_ref, digest)


def test_runtime_provenance_and_json_canonicalization_are_deterministic(
    tmp_path: Path,
) -> None:
    assert helpers._runtime_provenance_image_ref(_DIGEST_A, _DIGEST_A) == _DIGEST_A
    assert (
        helpers._runtime_provenance_image_ref(f"runtime@{_DIGEST_A}", _DIGEST_A)
        == f"runtime@{_DIGEST_A}"
    )
    assert (
        helpers.serialize_canonical_json({"z": (tmp_path / "model",), "a": {7: True}})
        == f'{{"a":{{"7":true}},"z":["{tmp_path / "model"}"]}}'
    )

    with pytest.raises(TypeError, match="unsupported type set"):
        helpers.serialize_canonical_json({"unsafe": {"value"}})
    with pytest.raises(ValueError, match="Out of range float values"):
        helpers.serialize_canonical_json({"unsafe": float("nan")})
