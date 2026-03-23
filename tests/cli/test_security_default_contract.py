from __future__ import annotations

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

    mounts = _mounts(command)
    assert ["-v", f"{config_parent}:{config_parent}"] in mounts
    assert ["-v", f"{out_parent}:{out_parent}"] in mounts
    assert ["-v", f"{report_parent}:{report_parent}"] in mounts


def test_container_launch_maps_repo_pythonpath_to_workspace_src(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
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
        ["run", "--config", "/tmp/dummy"]
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
        ["run", "--config", "/tmp/dummy"]
    )

    assert _env_value(command, "PYTHONPATH") == str(src_dir)
    assert ["-v", f"{src_dir}:{src_dir}"] in _mounts(command)


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

    mounts = [command[idx : idx + 2] for idx in range(len(command) - 1)]
    assert ["-v", f"{baseline_dir}:{baseline_dir}"] in mounts
    assert ["-v", f"{subject_dir}:{subject_dir}"] in mounts


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

    mounts = [command[idx : idx + 2] for idx in range(len(command) - 1)]
    assert ["-v", f"{baseline_dir}:{baseline_dir}"] in mounts
    assert ["-v", f"{cache_dir}:{cache_dir}"] in mounts


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

    mounts = [command[idx : idx + 2] for idx in range(len(command) - 1)]
    assert ["-v", f"{subject_dir}:{subject_dir}"] in mounts
