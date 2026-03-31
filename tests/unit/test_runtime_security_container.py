from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.runtime_security as runtime_security
import invarlock.runtime_security_helpers as runtime_security_helpers


def _plan(
    argv: list[str],
    *,
    mounts: tuple[Path, ...] = (),
    needs_mirror: bool = False,
    gpu_passthrough: bool = False,
) -> runtime_security.ContainerLaunchPlan:
    return runtime_security.ContainerLaunchPlan(
        argv=tuple(argv),
        argv_mounts=mounts,
        needs_cwd_host_mirror=needs_mirror,
        gpu_passthrough=gpu_passthrough,
    )


def test_apply_runtime_allowances_and_delegate_container_command(monkeypatch) -> None:
    import invarlock.security as security_module

    seen: list[bool] = []

    monkeypatch.setattr(
        security_module,
        "enforce_network_policy",
        lambda enabled: seen.append(enabled),
        raising=False,
    )
    runtime_security.apply_runtime_allowances(
        allow_network=True,
        allow_host_execution=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
        allow_unattested_artifacts=True,
    )

    runtime_security.apply_runtime_allowances(
        allow_network=False,
        allow_host_execution=False,
        allow_third_party_plugins=False,
        allow_remote_code=False,
        allow_unattested_artifacts=False,
    )

    assert seen == [True, False]
    assert runtime_security.network_allowed() is False
    assert runtime_security.host_execution_allowed() is False
    assert runtime_security.remote_code_allowed() is False
    assert runtime_security.unattested_artifacts_allowed() is False
    assert runtime_security.third_party_plugins_allowed() is False

    monkeypatch.setattr(
        runtime_security_helpers,
        "build_container_command",
        lambda plan: ["docker", "run"],
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda command, check=False, timeout=None: SimpleNamespace(returncode=7),
        raising=True,
    )
    try:
        assert runtime_security.delegate_container_command(_plan(["evaluate"])) == 7
    finally:
        runtime_security.reset_runtime_allowances()


def test_delegate_container_command_passes_timeout_and_surfaces_expiry(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        runtime_security_helpers,
        "build_container_command",
        lambda plan: ["docker", "run"],
        raising=True,
    )

    def _run(command, check=False, timeout=None):
        seen["timeout"] = timeout
        raise runtime_security.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(runtime_security.subprocess, "run", _run, raising=True)

    with pytest.raises(RuntimeError, match="timed out"):
        runtime_security.delegate_container_command(_plan(["evaluate"]))

    assert seen["timeout"] == runtime_security._CONTAINER_EXECUTION_TIMEOUT_SECONDS


def test_apply_runtime_allowances_rolls_back_network_policy_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.security as security_module

    monkeypatch.delenv(runtime_security.ALLOW_NETWORK_ENV, raising=False)
    monkeypatch.setattr(
        security_module,
        "enforce_network_policy",
        lambda enabled: (_ for _ in ()).throw(RuntimeError("network boom")),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="network boom"):
        runtime_security.apply_runtime_allowances(allow_network=True)

    assert runtime_security.network_allowed() is False


def test_build_container_python_command_uses_python_entrypoint_for_repo_script(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    script_path = (
        repo_root / "scripts" / "proof_packs" / "python" / "run_from_config.py"
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# stub\n", encoding="utf-8")
    config_path = repo_root / "configs" / "demo.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("demo: true\n", encoding="utf-8")
    out_dir = repo_root / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:abc",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: False,
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
        "_container_pythonpath_entries",
        lambda *, cwd: (["/workspace/src"], []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_delegated_env_pairs",
        lambda *, cwd: ({"EXTRA": "1"}, []),
        raising=True,
    )

    command = runtime_security.build_container_python_command(
        script_path,
        _plan(
            ["--config", "configs/demo.yaml", "--out", "runs"],
            gpu_passthrough=True,
        ),
    )

    assert command[:6] == [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python",
        "--gpus",
    ]
    assert "all" in command
    assert "--network" in command
    assert "none" in command
    assert "-e" in command
    assert "PYTHONPATH=/workspace/src" in command
    assert "EXTRA=1" in command
    assert "ghcr.io/invarlock/runtime:test" in command
    assert "/workspace/scripts/proof_packs/python/run_from_config.py" in command


def test_build_container_python_command_adds_cwd_host_mirror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    script_path = repo_root / "scripts" / "run.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# stub\n", encoding="utf-8")

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:abc",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_container_pythonpath_entries",
        lambda *, cwd: ([], []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_delegated_env_pairs",
        lambda *, cwd: ({}, []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )

    command = runtime_security.build_container_python_command(
        script_path,
        _plan(["--help"], needs_mirror=True),
    )

    cwd = str(repo_root.resolve())
    assert f"{cwd}:{cwd}" in command


def test_delegate_python_script_to_container_uses_python_builder(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_security_helpers,
        "build_container_python_command",
        lambda script_path, plan: ["docker", "run", "python", str(script_path)],
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda command, check=False, timeout=None: SimpleNamespace(returncode=9),
        raising=True,
    )

    assert (
        runtime_security.delegate_python_script_to_container(
            "scripts/proof_packs/python/run_from_config.py",
            _plan(["--config", "demo.yaml"]),
        )
        == 9
    )


def test_delegate_python_script_to_container_passes_timeout(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        runtime_security_helpers,
        "build_container_python_command",
        lambda script_path, plan: ["docker", "run", "python", str(script_path)],
        raising=True,
    )

    def _run(command, check=False, timeout=None):
        seen["timeout"] = timeout
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(runtime_security.subprocess, "run", _run, raising=True)

    assert (
        runtime_security.delegate_python_script_to_container(
            "scripts/proof_packs/python/run_from_config.py",
            _plan(["--config", "demo.yaml"]),
        )
        == 9
    )
    assert seen["timeout"] == runtime_security._CONTAINER_EXECUTION_TIMEOUT_SECONDS


def test_delegate_python_script_to_container_surfaces_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers,
        "build_container_python_command",
        lambda script_path, plan: ["docker", "run", "python", str(script_path)],
        raising=True,
    )

    def _run(command, check=False, timeout=None):
        raise runtime_security.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(runtime_security.subprocess, "run", _run, raising=True)

    with pytest.raises(RuntimeError, match="timed out"):
        runtime_security.delegate_python_script_to_container(
            "scripts/proof_packs/python/run_from_config.py",
            _plan(["--config", "demo.yaml"]),
        )


def test_build_container_python_command_raises_for_missing_engine_or_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "run_from_config.py"
    script_path.write_text("# stub\n", encoding="utf-8")

    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: None,
        raising=True,
    )
    with pytest.raises(RuntimeError, match="no container engine"):
        runtime_security.build_container_python_command(
            script_path, _plan(["--config", "cfg"])
        )

    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: None,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image, engine=None: False,
        raising=True,
    )
    with pytest.raises(RuntimeError, match="not available locally"):
        runtime_security.build_container_python_command(
            script_path, _plan(["--config", "cfg"])
        )


def test_build_container_python_command_leaves_external_script_as_host_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    script_path = external_root / "run_from_config.py"
    script_path.write_text("# stub\n", encoding="utf-8")

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: None,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_container_pythonpath_entries",
        lambda *, cwd: (["/workspace/src"], []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_delegated_env_pairs",
        lambda *, cwd: ({}, []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )

    command = runtime_security.build_container_python_command(
        script_path,
        _plan(["verify", "--help"]),
    )

    assert "--network" not in command
    assert str(script_path.resolve()) in command
    assert "/workspace/scripts/proof_packs/python/run_from_config.py" not in command


def test_build_container_command_raises_when_no_engine_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: None,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="no container engine"):
        runtime_security.build_container_command(_plan(["evaluate", "--help"]))


def test_build_container_command_uses_launch_plan_and_deduplicates_mounts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    shared_mount = tmp_path / "shared"
    shared_mount.mkdir()

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:attested",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_container_pythonpath_entries",
        lambda *, cwd: (["/workspace/src"], [shared_mount]),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_delegated_env_pairs",
        lambda *, cwd: ({"EXTRA": "1"}, [shared_mount]),
        raising=True,
    )
    command = runtime_security.build_container_command(
        _plan(
            ["evaluate", "--config", "cfg.yaml"],
            mounts=(shared_mount,),
            needs_mirror=True,
            gpu_passthrough=True,
        )
    )

    assert command[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in command
    assert "all" in command
    assert "--network" in command
    assert "none" in command
    assert f"{repo_root}:/workspace" in command
    assert f"{repo_root}:{repo_root}" in command
    assert sum(token == f"{shared_mount}:{shared_mount}" for token in command) == 1
    assert "PYTHONPATH=/workspace/src" in command
    assert "EXTRA=1" in command
    assert command[-4:] == [
        "ghcr.io/invarlock/runtime:test",
        "evaluate",
        "--config",
        "cfg.yaml",
    ]
