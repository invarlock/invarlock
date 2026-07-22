from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.core.runtime_provider import RuntimeArtifactResources

_IMAGE_DIGEST = "sha256:" + "a" * 64


def _resources(root: Path, **changes: object) -> RuntimeArtifactResources:
    values: dict[str, object] = {
        "root": root,
        "primary_artifact": "model/checkpoint",
        "support_resources": {"backend_executable": "bin/backend"},
        "device_kind": "cpu",
        "container_image_digest": _IMAGE_DIGEST,
    }
    values.update(changes)
    return RuntimeArtifactResources(**values)  # type: ignore[arg-type]


def test_runtime_artifact_resources_repeat_root_confined_no_follow_validation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resources"
    root.joinpath("model", "checkpoint").mkdir(parents=True)
    root.joinpath("bin").mkdir()
    backend = root / "bin" / "backend"
    backend.write_bytes(b"backend")
    resources = _resources(root)

    assert resources.primary_path() == root / "model" / "checkpoint"
    assert resources.support_path("backend_executable") == backend
    assert str(root) not in repr(resources)

    backend.unlink()
    backend.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="without symbolic links"):
        resources.support_path("backend_executable")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("primary_artifact", "../outside", "absolute or traversal"),
        ("primary_artifact", "/outside", "absolute or traversal"),
        ("primary_artifact", "model\\checkpoint", "portable relative"),
        ("device_kind", "mps", "cpu or cuda"),
        ("container_image_digest", "mutable:tag", "sha256 image digest"),
    ],
)
def test_runtime_artifact_resources_reject_invalid_capabilities_and_paths(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    root = tmp_path / "resources"
    root.joinpath("model", "checkpoint").mkdir(parents=True)
    root.joinpath("bin").mkdir()
    root.joinpath("bin", "backend").write_bytes(b"backend")

    with pytest.raises(ValueError, match=message):
        _resources(root, **{field: value})


def test_runtime_artifact_resources_reject_symlinked_components(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.joinpath("checkpoint").mkdir()
    root = tmp_path / "resources"
    root.mkdir()
    root.joinpath("model").symlink_to(outside, target_is_directory=True)
    root.joinpath("bin").mkdir()
    root.joinpath("bin", "backend").write_bytes(b"backend")

    with pytest.raises(ValueError, match="without symbolic links"):
        _resources(root)


def test_runtime_artifact_resources_reject_missing_and_unexpected_supports(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resources"
    root.joinpath("model", "checkpoint").mkdir(parents=True)
    root.joinpath("bin").mkdir()
    root.joinpath("bin", "backend").write_bytes(b"backend")
    resources = _resources(root)

    with pytest.raises(ValueError, match="missing required support"):
        resources.require_support_names(
            frozenset({"backend_executable", "backend_source"})
        )
    with pytest.raises(ValueError, match="unexpected support"):
        resources.require_support_names(frozenset())


def test_runtime_artifact_resources_reject_symlink_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.joinpath("model", "checkpoint").mkdir(parents=True)
    real_root.joinpath("bin").mkdir()
    real_root.joinpath("bin", "backend").write_bytes(b"backend")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink directory"):
        _resources(linked_root)
