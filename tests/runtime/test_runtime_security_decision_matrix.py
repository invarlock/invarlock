from __future__ import annotations

from pathlib import Path

import invarlock.runtime_security_helpers as runtime_security


def test_runtime_security_helpers_cover_build_command_and_covered_mounts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assert (
        runtime_security._runtime_image_build_command(
            runtime_security.RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT
        )
        == "make runtime-image-cuda"
    )
    assert runtime_security._runtime_image_build_command("invarlock-runtime:local") == (
        "make runtime-image"
    )

    cwd = tmp_path / "repo"
    external = tmp_path / "shared"
    cwd.mkdir()
    external.mkdir()

    monkeypatch.setenv("PYTHONPATH", str(external))
    monkeypatch.setattr(
        runtime_security,
        "_mount_is_already_covered",
        lambda _mount, *, cwd: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_iter_external_symlink_target_mounts",
        lambda _path, *, cwd, recursive=True: [tmp_path / "ignored-symlink"],
        raising=True,
    )

    entries, mounts = runtime_security._container_pythonpath_entries(cwd=cwd)
    assert entries == [str(external.resolve())]
    assert mounts == [tmp_path / "ignored-symlink"]

    recorded_mounts: set[Path] = set()
    inside = runtime_security._record_path_dependencies(
        external,
        recorded_mounts,
        cwd=cwd,
    )
    assert inside is False
    assert recorded_mounts == {tmp_path / "ignored-symlink"}
