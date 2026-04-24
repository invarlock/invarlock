from __future__ import annotations

import os
from pathlib import Path

import pytest

import invarlock.cli.runtime_launch_plan as runtime_launch_plan
import invarlock.runtime_security as runtime_security
import invarlock.runtime_security_helpers as runtime_security_helpers


def _env_value(command: list[str], key: str) -> str:
    needle = f"{key}="
    for idx, token in enumerate(command[:-1]):
        if token == "-e" and command[idx + 1].startswith(needle):
            return command[idx + 1][len(needle) :]
    raise AssertionError(f"environment variable {key} not found")


def _mounts(command: list[str]) -> list[list[str]]:
    return [command[idx : idx + 2] for idx in range(len(command) - 1)]


def _canonical_path(path: str | Path) -> Path:
    return Path(os.path.realpath(os.path.abspath(str(path))))


def _mounted_roots(command: list[str]) -> list[Path]:
    roots: list[Path] = []
    for idx, token in enumerate(command[:-1]):
        if token != "-v":
            continue
        host_root, _, _ = command[idx + 1].partition(":")
        roots.append(_canonical_path(host_root))
    return roots


def _path_is_mounted(command: list[str], path: str | Path) -> bool:
    target = _canonical_path(path)
    for root in _mounted_roots(command):
        try:
            target.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _delegated_argv(
    command: list[str], image: str = "invarlock-runtime:local"
) -> list[str]:
    image_idx = command.index(image)
    return command[image_idx + 1 :]


def _build_container_command(argv: list[str]) -> list[str]:
    return runtime_security.build_container_command(
        runtime_launch_plan.build_current_process_container_launch_plan(argv)
    )


def _stub_container_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )


def test_container_launch_path_env_mounts_skip_recursive_symlink_walk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    tmpdir_root = tmp_path / "runtime-tmp"
    tmpdir_root.mkdir()
    for name in runtime_security_helpers._PATH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmpdir_root))

    recursive_flags: list[bool] = []
    original = runtime_security_helpers._iter_external_symlink_target_mounts

    def _spy(path: Path, *, cwd: Path, recursive: bool = True) -> list[Path]:
        if path.resolve(strict=False) == tmpdir_root.resolve(strict=False):
            recursive_flags.append(recursive)
        return original(path, cwd=cwd, recursive=recursive)

    monkeypatch.setattr(
        runtime_security_helpers,
        "_iter_external_symlink_target_mounts",
        _spy,
        raising=True,
    )

    command = _build_container_command(["evaluate", "--help"])

    assert recursive_flags == [False]
    assert _path_is_mounted(command, tmpdir_root)


def test_container_launch_mounts_external_symlink_targets_for_local_model_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    baseline_dir = tmp_path / "baseline-model"
    baseline_dir.mkdir()
    subject_dir = tmp_path / "subject-model"
    subject_dir.mkdir()
    cache_dir = tmp_path / "hf-cache" / "blobs"
    cache_dir.mkdir(parents=True)
    blob_path = cache_dir / "weights.bin"
    blob_path.write_bytes(b"weights")
    (baseline_dir / "model.safetensors").symlink_to(blob_path)

    command = _build_container_command(
        [
            "evaluate",
            "--baseline",
            str(baseline_dir),
            "--subject",
            str(subject_dir),
        ]
    )

    assert _path_is_mounted(command, baseline_dir)
    assert _path_is_mounted(command, cache_dir)


def test_container_launch_mounts_absolute_model_paths_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    subject_dir = tmp_path / "subject-model"
    subject_dir.mkdir()
    config_path = repo_dir / "subject.yaml"
    config_path.write_text(
        f"model:\n  id: {subject_dir}\n  adapter: hf_causal\n",
        encoding="utf-8",
    )

    command = _build_container_command(["run", "--config", str(config_path)])

    assert _path_is_mounted(command, subject_dir)


def test_container_launch_preserves_repo_relative_output_args_and_mirrors_cwd_for_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    config_path = repo_dir / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)

    command = _build_container_command(
        [
            "evaluate",
            "--config",
            "config.yaml",
            "--out",
            "runs/eval",
            "--report-out",
            "reports/eval",
        ]
    )

    delegated = _delegated_argv(command)
    assert _path_is_mounted(command, repo_dir)
    assert delegated[delegated.index("--config") + 1] == str(config_path)
    assert delegated[delegated.index("--out") + 1] == "runs/eval"
    assert delegated[delegated.index("--report-out") + 1] == "reports/eval"


def test_container_launch_mounts_absolute_edit_config_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)

    edit_root = tmp_path / "edit-root"
    edit_root.mkdir()
    edit_path = edit_root / "overlay.yaml"
    edit_path.write_text("edit:\n  name: noop\n  plan: {}\n", encoding="utf-8")

    command = _build_container_command(
        [
            "evaluate",
            "--baseline",
            "sshleifer/tiny-gpt2",
            "--subject",
            "sshleifer/tiny-gpt2",
            "--edit-config",
            str(edit_path),
        ]
    )

    delegated = _delegated_argv(command)
    assert _path_is_mounted(command, edit_root)
    assert delegated[delegated.index("--edit-config") + 1] == str(edit_path)


def test_container_launch_leaves_missing_local_model_args_as_model_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)

    command = _build_container_command(
        [
            "evaluate",
            "--baseline",
            "sshleifer/tiny-gpt2",
            "--subject",
            "org/model-name",
        ]
    )

    delegated = _delegated_argv(command)
    mounts = _mounts(command)
    assert delegated[delegated.index("--baseline") + 1] == "sshleifer/tiny-gpt2"
    assert delegated[delegated.index("--subject") + 1] == "org/model-name"
    assert not any(
        token == "-v" and value.endswith("org/model-name:org/model-name")
        for token, value in mounts
    )
