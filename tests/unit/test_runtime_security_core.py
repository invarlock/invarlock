from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.cli.runtime_launch_plan as runtime_launch_plan
import invarlock.runtime_security as runtime_security
import invarlock.runtime_security_helpers as runtime_security_helpers


def test_runtime_bool_helpers_and_execution_mode(monkeypatch) -> None:
    monkeypatch.setenv(runtime_security.ALLOW_NETWORK_ENV, "1")
    monkeypatch.setenv(runtime_security.ALLOW_HOST_EXECUTION_ENV, "0")
    monkeypatch.setenv(runtime_security.ALLOW_REMOTE_CODE_ENV, "yes")
    monkeypatch.setenv(runtime_security.ALLOW_UNATTESTED_ARTIFACTS_ENV, "true")
    monkeypatch.setenv(runtime_security.ALLOW_THIRD_PARTY_PLUGINS_ENV, "1")
    monkeypatch.setenv(runtime_security.CONTAINER_EXECUTION_ENV, "1")

    assert runtime_security._coerce_bool("on") is True
    assert runtime_security._coerce_bool("off") is False
    assert runtime_security._coerce_bool("maybe") is None
    assert runtime_security.network_allowed() is True
    assert runtime_security.host_execution_allowed() is False
    assert runtime_security.remote_code_allowed() is True
    assert runtime_security.unattested_artifacts_allowed() is True
    assert runtime_security.third_party_plugins_allowed() is True
    assert runtime_security.running_inside_container() is True
    assert runtime_security.current_execution_mode() == "container"

    with runtime_security.runtime_allowances_scope(
        allow_network=True,
        allow_host_execution=True,
        allow_remote_code=True,
        allow_unattested_artifacts=True,
        allow_third_party_plugins=True,
    ):
        assert runtime_security.network_allowed() is True
        assert runtime_security.host_execution_allowed() is True
        assert runtime_security.remote_code_allowed() is True
        assert runtime_security.unattested_artifacts_allowed() is True
        assert runtime_security.third_party_plugins_allowed() is True


def test_serialize_canonical_json_normalizes_supported_types() -> None:
    payload = {
        "path": Path("artifact.txt"),
        "values": {3, 1},
        "nested": [
            Path("nested.txt"),
            {"report": Path("payload.json")},
            SimpleNamespace(answer=42),
        ],
    }

    encoded = runtime_security.serialize_canonical_json(payload)
    decoded = json.loads(encoded)

    assert decoded["path"] == "artifact.txt"
    assert decoded["values"] == [1, 3]
    assert decoded["nested"][0] == "nested.txt"
    assert decoded["nested"][1] == {"report": "payload.json"}
    assert decoded["nested"][2].startswith("namespace(")


def test_resolve_runtime_image_digest_prefers_explicit_and_embedded_digest(
    monkeypatch,
) -> None:
    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, "sha256:explicit")
    assert runtime_security.resolve_runtime_image_digest() == "sha256:explicit"

    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, raising=False)
    monkeypatch.setenv(
        runtime_security.RUNTIME_IMAGE_ENV,
        "ghcr.io/invarlock/invarlock-runtime:test@sha256:embedded",
    )
    assert runtime_security.resolve_runtime_image_digest() == "sha256:embedded"


def test_resolve_runtime_image_digest_uses_inspection_when_needed(monkeypatch) -> None:
    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, raising=False)
    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_ENV, raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/invarlock-runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_inspect_container_image",
        lambda engine, image: (True, "sha256:inspected"),
        raising=True,
    )

    assert runtime_security.resolve_runtime_image_digest() == "sha256:inspected"


def test_resolve_runtime_image_digest_returns_none_without_engine(monkeypatch) -> None:
    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, raising=False)
    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_ENV, raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/invarlock-runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: None,
        raising=True,
    )

    assert runtime_security.resolve_runtime_image_digest() is None


def test_resolve_runtime_image_prefers_explicit_local_and_default(monkeypatch) -> None:
    monkeypatch.setenv(
        runtime_security.RUNTIME_IMAGE_ENV,
        "ghcr.io/invarlock/invarlock-runtime:explicit",
    )
    assert (
        runtime_security.resolve_runtime_image()
        == "ghcr.io/invarlock/invarlock-runtime:explicit"
    )

    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_ENV, raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image, engine=None: image
        == runtime_security.RUNTIME_IMAGE_LOCAL_DEFAULT,
        raising=True,
    )
    assert (
        runtime_security.resolve_runtime_image()
        == runtime_security.RUNTIME_IMAGE_LOCAL_DEFAULT
    )

    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image, engine=None: False,
        raising=True,
    )
    assert (
        runtime_security.resolve_runtime_image()
        == runtime_security.RUNTIME_IMAGE_DEFAULT
    )


def test_inspect_container_image_parses_repo_digest_and_image_id(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='["ghcr.io/invarlock/invarlock-runtime:test@sha256:abc"]\nsha256:def\n',
        ),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (
        True,
        "sha256:abc",
    )

    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="not-json\nsha256:def\n",
        ),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (
        True,
        "sha256:def",
    )


def test_inspect_container_image_handles_failures_and_digestless_images(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout=""),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (False, None)

    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[]\nimage-id\n"),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (True, None)

    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (True, None)

    monkeypatch.setattr(
        runtime_security.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='["ghcr.io/invarlock/runtime:test"]\nimage-id\n',
        ),
        raising=True,
    )
    assert runtime_security._inspect_container_image("docker", "img") == (True, None)


def test_container_engine_and_device_helpers(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_security.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
        raising=True,
    )
    assert runtime_security.resolve_container_engine() == "podman"

    assert runtime_launch_plan._requested_device(["evaluate"]) == "auto"
    assert runtime_launch_plan._requested_device(["run"]) == "auto"
    assert runtime_launch_plan._requested_device(["verify"]) is None
    assert (
        runtime_launch_plan._requested_device(["evaluate", "--device", "CUDA"])
        == "cuda"
    )
    assert runtime_launch_plan._requested_device(["evaluate", "--device"]) is None

    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )
    assert runtime_launch_plan._needs_gpu_passthrough(["evaluate"]) is True
    assert (
        runtime_launch_plan._needs_gpu_passthrough(["evaluate", "--device", "cpu"])
        is False
    )

    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )
    assert (
        runtime_launch_plan._needs_gpu_passthrough(["evaluate", "--device", "cuda"])
        is False
    )


def test_runtime_security_device_helpers_cover_missing_tokens_and_devnode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_security.shutil,
        "which",
        lambda name: None,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security.Path,
        "exists",
        lambda self: str(self) == "/dev/nvidiactl",
        raising=False,
    )

    assert runtime_security.resolve_container_engine() is None
    assert runtime_security._host_nvidia_visible() is True
    assert runtime_launch_plan._requested_device(["--help"]) is None


def test_container_image_available_locally_and_runtime_verifier_binary(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: None,
        raising=True,
    )
    assert runtime_security.container_image_available_locally("img") is False

    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_inspect_container_image",
        lambda engine, image: (True, None),
        raising=True,
    )
    assert runtime_security.container_image_available_locally("img") is True

    verifier = tmp_path / runtime_security.RUNTIME_VERIFIER_BINARY_DEFAULT
    verifier.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv(runtime_security.RUNTIME_VERIFIER_BINARY_ENV, str(verifier))
    assert runtime_security.runtime_verifier_binary() == str(verifier)


def test_runtime_verifier_binary_finds_repo_and_script_dir_candidates(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    module_path = repo_root / "src" / "invarlock" / "runtime_security.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("# stub\n", encoding="utf-8")
    debug_binary = (
        repo_root
        / "target"
        / "debug"
        / runtime_security.RUNTIME_VERIFIER_BINARY_DEFAULT
    )
    debug_binary.parent.mkdir(parents=True, exist_ok=True)
    debug_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    debug_binary.chmod(0o755)

    monkeypatch.delenv(runtime_security.RUNTIME_VERIFIER_BINARY_ENV, raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "__file__",
        str(module_path),
        raising=False,
    )
    assert runtime_security.runtime_verifier_binary() == str(debug_binary)

    debug_binary.unlink()
    script_dir = tmp_path / "venv" / "bin"
    script_dir.mkdir(parents=True, exist_ok=True)
    python_bin = script_dir / "python"
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    python_bin.chmod(0o755)
    script_binary = script_dir / runtime_security.RUNTIME_VERIFIER_BINARY_DEFAULT
    script_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    script_binary.chmod(0o755)
    monkeypatch.setattr(
        runtime_security.sys, "executable", str(python_bin), raising=True
    )
    monkeypatch.setattr(
        runtime_security.sys, "argv", [str(script_dir / "cli")], raising=True
    )
    assert runtime_security.runtime_verifier_binary() == str(script_binary)


def test_runtime_verifier_binary_uses_executable_dir_when_argv_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    module_path = repo_root / "src" / "invarlock" / "runtime_security.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("# stub\n", encoding="utf-8")

    script_dir = tmp_path / "venv" / "bin"
    script_dir.mkdir(parents=True, exist_ok=True)
    python_bin = script_dir / "python"
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    python_bin.chmod(0o755)
    verifier = script_dir / runtime_security.RUNTIME_VERIFIER_BINARY_DEFAULT
    verifier.write_text("#!/bin/sh\n", encoding="utf-8")
    verifier.chmod(0o755)

    monkeypatch.delenv(runtime_security.RUNTIME_VERIFIER_BINARY_ENV, raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "__file__",
        str(module_path),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_security.sys, "executable", str(python_bin), raising=True
    )
    monkeypatch.setattr(runtime_security.sys, "argv", [], raising=True)

    assert runtime_security.runtime_verifier_binary() == str(verifier)
