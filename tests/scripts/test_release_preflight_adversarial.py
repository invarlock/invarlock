from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.release import release_preflight as preflight


def _config(tmp_path: Path) -> preflight.ReleasePreflightConfig:
    root = tmp_path / "checkout"
    root.mkdir(exist_ok=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='invarlock'\nversion='1.2.3'\n", encoding="utf-8"
    )
    return preflight.ReleasePreflightConfig(
        repo_root=root,
        release_sha="a" * 40,
        expected_version="1.2.3",
        dist_dir=root / "dist",
        hash_manifest=root / "hashes.txt",
    )


def _inventory() -> dict[str, object]:
    return {
        "category": "runtime-providers",
        "items": [
            {
                "name": name,
                "entry_point": name,
                "entry_point_group": "invarlock.runtime_providers",
                "kind": "runtime_provider",
                "origin": "builtin",
                "status": "ready",
                **expected,
            }
            for name, expected in preflight._FIRST_PARTY_RUNTIME_PROVIDERS.items()
        ],
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_git_output_returns_stripped_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, " value \n", ""),
    )
    assert preflight._git_output(tmp_path, "rev-parse", "HEAD") == "value"


def test_dependency_bridge_rejects_missing_parent_and_child_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight.site, "getsitepackages", lambda: [])
    with pytest.raises(
        preflight.ReleasePreflightError, match="environment is unavailable"
    ):
        preflight._install_isolated_dependency_bridge(tmp_path / "environment")

    parent = tmp_path / "locked" / "site-packages"
    parent.mkdir(parents=True)
    monkeypatch.setattr(preflight.site, "getsitepackages", lambda: [str(parent)])
    with pytest.raises(
        preflight.ReleasePreflightError, match="site-packages directory"
    ):
        preflight._install_isolated_dependency_bridge(tmp_path / "environment")


def test_dependency_bridge_normalizes_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "locked" / "site-packages"
    parent.mkdir(parents=True)
    environment = tmp_path / "environment"
    child = (
        environment
        / "lib"
        / f"python{preflight.sys.version_info.major}.{preflight.sys.version_info.minor}"
        / "site-packages"
    )
    child.mkdir(parents=True)
    monkeypatch.setattr(preflight.site, "getsitepackages", lambda: [str(parent)])

    original = Path.write_text

    def fail_write(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == "invarlock-release-dependencies.pth":
            raise OSError("disk failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(preflight.ReleasePreflightError, match="unable to bind"):
        preflight._install_isolated_dependency_bridge(environment)


def test_installed_wheel_command_requires_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = subprocess.CompletedProcess([], 0, "ok", "")
    monkeypatch.setattr(
        preflight, "_run_isolated_wheel_command", lambda *_args, **_kwargs: completed
    )
    assert (
        preflight._require_successful_installed_wheel_command(
            ["candidate"], cwd=tmp_path, label="probe"
        )
        is completed
    )

    monkeypatch.setattr(
        preflight,
        "_run_isolated_wheel_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "private"),
    )
    with pytest.raises(preflight.ReleasePreflightError, match="probe failed"):
        preflight._require_successful_installed_wheel_command(
            ["candidate"], cwd=tmp_path, label="probe"
        )


def test_wheel_probe_rejects_checkout_local_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    inside = config.repo_root / "temporary"
    inside.mkdir()

    class TemporaryDirectory:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> str:
            return str(inside)

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(preflight.tempfile, "TemporaryDirectory", TemporaryDirectory)
    with pytest.raises(preflight.ReleasePreflightError, match="outside checkout"):
        preflight._probe_installed_wheel(config, tmp_path / "candidate.whl")


def test_wheel_probe_rejects_import_outside_created_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    completed = subprocess.CompletedProcess([], 0, "{}", "")
    imported = preflight.InstalledWheelImport(
        module_file=config.repo_root / "elsewhere" / "invarlock" / "__init__.py",
        module_version="1.2.3",
        distribution_name="invarlock",
        distribution_version="1.2.3",
        distribution_root=config.repo_root / "elsewhere",
    )
    monkeypatch.setattr(
        preflight, "_run_isolated_wheel_command", lambda *_args, **_kwargs: completed
    )
    monkeypatch.setattr(
        preflight, "_install_isolated_dependency_bridge", lambda _path: None
    )
    monkeypatch.setattr(preflight, "_require_executable_file", lambda *_args: None)
    monkeypatch.setattr(preflight, "_parse_import_probe", lambda _payload: imported)

    with pytest.raises(preflight.ReleasePreflightError, match="isolated environment"):
        preflight._probe_installed_wheel(config, tmp_path / "candidate.whl")
