from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.runtime_security as runtime_security


def test_apply_runtime_allowances_and_delegate_current_process(monkeypatch) -> None:
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

    assert seen == [True]
    assert runtime_security.network_allowed() is True
    assert runtime_security.host_execution_allowed() is True
    assert runtime_security.remote_code_allowed() is True
    assert runtime_security.unattested_artifacts_allowed() is True
    assert runtime_security.third_party_plugins_allowed() is True

    monkeypatch.setattr(
        runtime_security,
        "build_container_command",
        lambda argv=None: ["docker", "run"],
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda command, check=False: SimpleNamespace(returncode=7),
        raising=True,
    )
    assert runtime_security.delegate_current_process_to_container(["evaluate"]) == 7


def test_apply_runtime_allowances_ignores_network_policy_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.security as security_module

    monkeypatch.setattr(
        security_module,
        "enforce_network_policy",
        lambda enabled: (_ for _ in ()).throw(RuntimeError("network boom")),
        raising=False,
    )

    runtime_security.apply_runtime_allowances(allow_network=True)

    assert runtime_security.network_allowed() is True


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
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:abc",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_container_pythonpath_entries",
        lambda *, cwd: (["/workspace/src"], []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_delegated_env_pairs",
        lambda *, cwd: ({"EXTRA": "1"}, []),
        raising=True,
    )

    command = runtime_security.build_container_python_command(
        script_path,
        ["--config", "configs/demo.yaml", "--out", "runs"],
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


def test_delegate_python_script_to_container_uses_python_builder(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_security,
        "build_container_python_command",
        lambda script_path, argv=None: ["docker", "run", "python", str(script_path)],
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda command, check=False: SimpleNamespace(returncode=9),
        raising=True,
    )

    assert (
        runtime_security.delegate_python_script_to_container(
            "scripts/proof_packs/python/run_from_config.py",
            ["--config", "demo.yaml"],
        )
        == 9
    )


def test_build_container_python_command_raises_for_missing_engine_or_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "run_from_config.py"
    script_path.write_text("# stub\n", encoding="utf-8")

    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: None,
        raising=True,
    )
    with pytest.raises(RuntimeError, match="no container engine"):
        runtime_security.build_container_python_command(
            script_path, ["--config", "cfg"]
        )

    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: None,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image, engine=None: False,
        raising=True,
    )
    with pytest.raises(RuntimeError, match="not available locally"):
        runtime_security.build_container_python_command(
            script_path, ["--config", "cfg"]
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
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: None,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "network_allowed",
        lambda: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_container_pythonpath_entries",
        lambda *, cwd: (["/workspace/src"], []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_delegated_env_pairs",
        lambda *, cwd: ({}, []),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )

    command = runtime_security.build_container_python_command(
        script_path,
        ["verify", "--help"],
    )

    assert "--network" not in command
    assert str(script_path.resolve()) in command
    assert "/workspace/scripts/proof_packs/python/run_from_config.py" not in command


def test_build_container_command_raises_when_no_engine_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: None,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="no container engine"):
        runtime_security.build_container_command(["evaluate", "--help"])


def test_build_container_command_uses_sys_argv_and_deduplicates_mounts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    shared_mount = tmp_path / "shared"
    shared_mount.mkdir()

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        runtime_security,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "resolve_runtime_image_digest",
        lambda: "sha256:attested",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "container_image_available_locally",
        lambda image, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security.sys,
        "argv",
        ["invarlock", "evaluate", "--config", "cfg.yaml"],
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_normalize_delegated_argv",
        lambda argv, *, cwd: (list(argv), [shared_mount], True),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_container_pythonpath_entries",
        lambda *, cwd: (["/workspace/src"], [shared_mount]),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_delegated_env_pairs",
        lambda *, cwd: ({"EXTRA": "1"}, [shared_mount]),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "_needs_gpu_passthrough",
        lambda argv: True,
        raising=True,
    )

    command = runtime_security.build_container_command()

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
