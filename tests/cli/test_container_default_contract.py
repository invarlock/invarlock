from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.core.registry as registry_mod
import invarlock.runtime_security as runtime_launch_plan
import invarlock.runtime_security as runtime_security
import invarlock.runtime_security_helpers as runtime_security_helpers
from invarlock.cli.run_config import extract_model_load_kwargs
from invarlock.core.exceptions import InvarlockError

_IMMUTABLE_IMAGE_ID = "sha256:" + "e" * 64


@pytest.fixture(autouse=True)
def _observe_immutable_runtime_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_security_helpers,
        "_resolve_observed_container_image",
        lambda engine, image: runtime_security_helpers._ObservedContainerImage(
            immutable_ref=_IMMUTABLE_IMAGE_ID,
            image_digest=_IMMUTABLE_IMAGE_ID,
            image_id=_IMMUTABLE_IMAGE_ID,
            repo_digests=(),
        ),
        raising=True,
    )


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
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image=None, *, engine=None: False,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="runtime image"):
        _build_container_command(["evaluate", "--help"])


def test_runtime_image_prefers_local_build_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE", raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image=None, *, engine=None: image == "invarlock-runtime:local",
        raising=True,
    )

    assert runtime_security.resolve_runtime_image() == "invarlock-runtime:local"


def test_runtime_image_prefers_local_cuda_build_when_gpu_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE", raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image=None, *, engine=None: image == "invarlock-runtime:cuda-local",
        raising=True,
    )

    assert runtime_security.resolve_runtime_image() == "invarlock-runtime:cuda-local"


def test_runtime_image_defaults_to_registry_when_local_build_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_RUNTIME_IMAGE", raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
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

    command = _build_container_command(["evaluate", "--help"])

    assert command[-3:] == [_IMMUTABLE_IMAGE_ID, "evaluate", "--help"]
    assert "python" not in command[command.index(_IMMUTABLE_IMAGE_ID) + 1 :]
    assert _env_value(command, "INVARLOCK_RUNTIME_IMAGE") == _IMMUTABLE_IMAGE_ID
    assert _env_value(command, "INVARLOCK_RUNTIME_IMAGE_DIGEST") == _IMMUTABLE_IMAGE_ID


@pytest.mark.parametrize("engine", ["docker", "podman"])
def test_container_launch_adds_gpu_passthrough_for_cuda_model_commands(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: engine,
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
    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )

    command = _build_container_command(["evaluate", "--device", "cuda", "--help"])

    assert command[:6] == [engine, "run", "--rm", "--init", "--gpus", "all"]


def test_container_launch_forwards_gpu_pinning_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_container_launch(monkeypatch)
    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")

    command = _build_container_command(["evaluate", "--device", "cuda", "--help"])

    assert _env_value(command, "CUDA_VISIBLE_DEVICES") == "1"
    assert _env_value(command, "NVIDIA_VISIBLE_DEVICES") == "1"
    assert _env_value(command, "HF_HUB_DISABLE_XET") == "1"


def test_container_launch_skips_gpu_passthrough_for_cpu_model_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )

    command = _build_container_command(["evaluate", "--device", "cpu", "--help"])

    assert "--gpus" not in command


def test_container_launch_mounts_absolute_output_and_report_paths(
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

    config_parent = tmp_path / "config-parent"
    config_parent.mkdir()
    config_path = config_parent / "preset.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    out_parent = tmp_path / "out-parent"
    report_parent = tmp_path / "report-parent"
    out_path = out_parent / "run-out"
    report_path = report_parent / "report-out"

    command = _build_container_command(
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

    command = _build_container_command(["run", "--config", "config.yaml"])

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

    command = _build_container_command(["run", "--config", "config.yaml"])

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
    assert _path_is_mounted(command, subject_dir)


def test_container_launch_mounts_absolute_source_and_edited_paths(
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

    source_dir = tmp_path / "source-model"
    source_dir.mkdir()
    edited_dir = tmp_path / "edited-model"
    edited_dir.mkdir()

    command = _build_container_command(
        [
            "evaluate",
            "--baseline",
            str(source_dir),
            "--subject",
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
    preset_dir = tmp_path / "preset-root"
    preset_dir.mkdir()
    preset_path = preset_dir / "preset.yaml"
    preset_path.write_text("dataset:\n  seq_len: 128\n", encoding="utf-8")
    baseline_report_dir = tmp_path / "baseline-report-root"
    baseline_report_dir.mkdir()
    baseline_report_path = baseline_report_dir / "baseline_report.json"
    baseline_report_path.write_text('{"ok": true}\n', encoding="utf-8")

    command = _build_container_command(
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

    config_root = tmp_path / "config-root"
    config_root.mkdir()

    command = _build_container_command(["evaluate", "--help"])

    assert _path_is_mounted(command, config_root)
