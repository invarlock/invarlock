from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.core.registry as registry_mod
import invarlock.runtime_security as runtime_security
from invarlock.cli.run_config import extract_model_load_kwargs
from invarlock.core.exceptions import InvarlockError


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


def _stub_container_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )


def test_third_party_plugin_discovery_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "0")

    def _entry_points_disabled() -> None:
        raise AssertionError("entry_points should not be called by default")

    monkeypatch.setattr(
        registry_mod,
        "entry_points",
        _entry_points_disabled,
        raising=True,
    )

    registry = registry_mod.CoreRegistry()
    assert "hf_causal" in registry.list_adapters()
    assert "invariants" in registry.list_guards()


def test_model_trust_remote_code_requires_explicit_allow() -> None:
    cfg = SimpleNamespace(model_dump=lambda: {"model": {"trust_remote_code": True}})

    with pytest.raises(InvarlockError, match="REMOTE-CODE-DISABLED"):
        extract_model_load_kwargs(cfg, invarlock_error_cls=InvarlockError)


def test_container_launch_requires_local_image_when_network_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: False,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="runtime image"):
        runtime_security.build_container_command(["evaluate", "--help"])


def test_runtime_image_prefers_local_build_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE", raising=False)
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: image == "invarlock-runtime:local",
        raising=True,
    )

    assert runtime_security.resolve_runtime_image() == "invarlock-runtime:local"


def test_runtime_image_defaults_to_registry_when_local_build_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE", raising=False)
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: False,
        raising=True,
    )

    assert (
        runtime_security.resolve_runtime_image()
        == "ghcr.io/invarlock/invarlock-runtime:latest"
    )


def test_container_launch_uses_runtime_image_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    command = runtime_security.build_container_command(["evaluate", "--help"])

    assert command[-3:] == ["invarlock-runtime:local", "evaluate", "--help"]
    assert "python" not in command[command.index("invarlock-runtime:local") + 1 :]


def test_container_launch_adds_gpu_passthrough_for_cuda_model_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )

    command = runtime_security.build_container_command(
        ["evaluate", "--device", "cuda", "--help"]
    )

    assert command[:5] == ["docker", "run", "--rm", "--gpus", "all"]


def test_container_launch_skips_gpu_passthrough_for_cpu_model_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )

    command = runtime_security.build_container_command(
        ["evaluate", "--device", "cpu", "--help"]
    )

    assert "--gpus" not in command


def test_container_launch_mounts_absolute_output_and_report_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    config_parent = tmp_path / "config-parent"
    config_parent.mkdir()
    config_path = config_parent / "preset.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    out_parent = tmp_path / "out-parent"
    report_parent = tmp_path / "report-parent"
    out_path = out_parent / "run-out"
    report_path = report_parent / "report-out"

    command = runtime_security.build_container_command(
        [
            "evaluate",
            "--config",
            str(config_path),
            "--out",
            str(out_path),
            "--report-out",
            str(report_path),
        ]
    )

    assert _path_is_mounted(command, config_parent)
    assert _path_is_mounted(command, out_parent)
    assert _path_is_mounted(command, report_parent)


def test_container_launch_maps_repo_pythonpath_to_workspace_src(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    config_path = repo_dir / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    monkeypatch.chdir(repo_dir)
    monkeypatch.setenv("PYTHONPATH", str(src_dir))
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    command = runtime_security.build_container_command(
        ["run", "--config", "config.yaml"]
    )

    assert _env_value(command, "PYTHONPATH") == "/workspace/src"


def test_container_launch_mounts_absolute_pythonpath_when_running_from_workdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    workdir = tmp_path / "run-root" / ".workdir"
    workdir.mkdir(parents=True)
    config_path = workdir / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("PYTHONPATH", str(src_dir))
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    command = runtime_security.build_container_command(
        ["run", "--config", "config.yaml"]
    )

    assert _env_value(command, "PYTHONPATH") == str(src_dir)
    assert _path_is_mounted(command, src_dir)


def test_container_launch_mounts_absolute_model_paths_from_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    baseline_dir = tmp_path / "baseline-model"
    baseline_dir.mkdir()
    subject_dir = tmp_path / "subject-model"
    subject_dir.mkdir()

    command = runtime_security.build_container_command(
        [
            "evaluate",
            "--baseline",
            str(baseline_dir),
            "--subject",
            str(subject_dir),
        ]
    )

    assert _path_is_mounted(command, baseline_dir)
    assert _path_is_mounted(command, subject_dir)


def test_container_launch_mounts_absolute_source_and_edited_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    source_dir = tmp_path / "source-model"
    source_dir.mkdir()
    edited_dir = tmp_path / "edited-model"
    edited_dir.mkdir()

    command = runtime_security.build_container_command(
        [
            "evaluate",
            "--source",
            str(source_dir),
            "--edited",
            str(edited_dir),
        ]
    )

    assert _path_is_mounted(command, source_dir)
    assert _path_is_mounted(command, edited_dir)


def test_container_launch_mounts_absolute_preset_and_baseline_report_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    baseline_dir = tmp_path / "baseline-model"
    baseline_dir.mkdir()
    subject_dir = tmp_path / "subject-model"
    subject_dir.mkdir()
    preset_dir = tmp_path / "preset-root"
    preset_dir.mkdir()
    preset_path = preset_dir / "preset.yaml"
    preset_path.write_text("dataset:\n  seq_len: 128\n", encoding="utf-8")
    baseline_report_dir = tmp_path / "baseline-report-root"
    baseline_report_dir.mkdir()
    baseline_report_path = baseline_report_dir / "baseline_report.json"
    baseline_report_path.write_text('{"ok": true}\n', encoding="utf-8")

    command = runtime_security.build_container_command(
        [
            "evaluate",
            "--baseline",
            str(baseline_dir),
            "--subject",
            str(subject_dir),
            "--preset",
            str(preset_path),
            "--baseline-report",
            str(baseline_report_path),
        ]
    )

    assert _path_is_mounted(command, preset_dir)
    assert _path_is_mounted(command, baseline_report_dir)


def test_container_launch_mounts_absolute_config_root_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(repo_dir)
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(tmp_path / "config-root"))
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )

    config_root = tmp_path / "config-root"
    config_root.mkdir()

    command = runtime_security.build_container_command(["evaluate", "--help"])

    assert _path_is_mounted(command, config_root)


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
    for name in runtime_security._PATH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmpdir_root))

    recursive_flags: list[bool] = []
    original = runtime_security._iter_external_symlink_target_mounts

    def _spy(path: Path, *, cwd: Path, recursive: bool = True) -> list[Path]:
        if path.resolve(strict=False) == tmpdir_root.resolve(strict=False):
            recursive_flags.append(recursive)
        return original(path, cwd=cwd, recursive=recursive)

    monkeypatch.setattr(
        runtime_security,
        "_iter_external_symlink_target_mounts",
        _spy,
        raising=True,
    )

    command = runtime_security.build_container_command(["evaluate", "--help"])

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
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
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

    command = runtime_security.build_container_command(
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
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
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

    command = runtime_security.build_container_command(
        ["run", "--config", str(config_path)]
    )

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

    command = runtime_security.build_container_command(
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

    command = runtime_security.build_container_command(
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

    command = runtime_security.build_container_command(
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


def test_container_launch_forwards_reviewed_runtime_env_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    config_path = repo_dir / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    export_dir = repo_dir / "exports"
    export_dir.mkdir()
    eval_tmp_dir = repo_dir / "tmp-eval"
    eval_tmp_dir.mkdir()
    config_root = tmp_path / "config-root"
    config_root.mkdir()
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    hub_cache = hf_home / "hub"
    hub_cache.mkdir()
    datasets_cache = hf_home / "datasets"
    datasets_cache.mkdir()
    transformers_cache = tmp_path / "transformers-cache"
    transformers_cache.mkdir()
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()

    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)
    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(config_root))
    monkeypatch.setenv("INVARLOCK_EVALUATE_TMP_DIR", str(eval_tmp_dir))
    monkeypatch.setenv("INVARLOCK_EXPORT_DIR", str(export_dir))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(datasets_cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(transformers_cache))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", "1")
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "auto")
    monkeypatch.setenv("INVARLOCK_SKIP_OVERHEAD_CHECK", "1")
    monkeypatch.setenv("INVARLOCK_DETERMINISM", "strict")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    command = runtime_security.build_container_command(
        ["run", "--config", str(config_path)]
    )

    assert _path_is_mounted(command, config_root)
    assert _path_is_mounted(command, hf_home)
    assert _path_is_mounted(command, transformers_cache)
    assert _path_is_mounted(command, tmpdir)

    assert _env_value(command, "INVARLOCK_CONFIG_ROOT") == str(config_root)
    assert _env_value(command, "INVARLOCK_EVALUATE_TMP_DIR") == "/workspace/tmp-eval"
    assert _env_value(command, "INVARLOCK_EXPORT_DIR") == "/workspace/exports"
    assert _env_value(command, "HF_HOME") == str(hf_home)
    assert _env_value(command, "HF_HUB_CACHE") == str(hub_cache)
    assert _env_value(command, "HF_DATASETS_CACHE") == str(datasets_cache)
    assert _env_value(command, "TRANSFORMERS_CACHE") == str(transformers_cache)
    assert _env_value(command, "TMPDIR") == str(tmpdir)
    assert _env_value(command, "INVARLOCK_STORE_EVAL_WINDOWS") == "1"
    assert _env_value(command, "INVARLOCK_SNAPSHOT_MODE") == "auto"
    assert _env_value(command, "INVARLOCK_SKIP_OVERHEAD_CHECK") == "1"
    assert _env_value(command, "INVARLOCK_DETERMINISM") == "strict"
    assert _env_value(command, "HF_DATASETS_OFFLINE") == "1"


def test_container_launch_scans_config_includes_and_absolute_references(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    dataset_dir = external_root / "dataset"
    dataset_dir.mkdir()
    include_path = external_root / "include.yaml"
    include_path.write_text(
        "dataset:\n"
        f"  file: {dataset_dir / 'corpus.jsonl'}\n"
        "model:\n"
        "  id: sshleifer/tiny-gpt2\n"
        "  adapter: hf_causal\n",
        encoding="utf-8",
    )
    config_path = repo_dir / "config.yaml"
    config_path.write_text(
        f"defaults: !include ../external/{include_path.name}\nedit:\n  name: noop\n  plan: {{}}\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)
    monkeypatch.setenv("INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE", "1")

    command = runtime_security.build_container_command(
        ["run", "--config", "config.yaml"]
    )

    assert _path_is_mounted(command, external_root)
    assert _env_value(command, "INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE") == "1"


def test_container_launch_fails_closed_when_config_scan_rejects_include(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    include_path = external_root / "include.yaml"
    include_path.write_text("model: {}\n", encoding="utf-8")
    config_path = repo_dir / "config.yaml"
    config_path.write_text(
        f"defaults: !include ../external/{include_path.name}\nedit:\n  name: noop\n  plan: {{}}\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)

    with pytest.raises(RuntimeError, match="Delegated runtime config"):
        runtime_security.build_container_command(["run", "--config", "config.yaml"])
