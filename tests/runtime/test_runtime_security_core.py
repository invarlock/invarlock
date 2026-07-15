from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.runtime_security as runtime_launch_plan
import invarlock.runtime_security as runtime_security
import invarlock.runtime_security_helpers as runtime_security_helpers


def test_runtime_bool_helpers_and_execution_mode(monkeypatch) -> None:
    monkeypatch.setenv(runtime_security.ALLOW_NETWORK_ENV, "1")
    monkeypatch.setenv(runtime_security.ALLOW_HOST_EXECUTION_ENV, "0")
    monkeypatch.setenv(runtime_security.ALLOW_REMOTE_CODE_ENV, "yes")
    monkeypatch.setenv(runtime_security.ALLOW_UNVERIFIED_PROVENANCE_ENV, "true")
    monkeypatch.setenv(runtime_security.ALLOW_THIRD_PARTY_PLUGINS_ENV, "1")
    monkeypatch.setenv(runtime_security.CONTAINER_EXECUTION_ENV, "1")

    assert runtime_security_helpers._coerce_bool("on") is True
    assert runtime_security_helpers._coerce_bool("off") is False
    assert runtime_security_helpers._coerce_bool("maybe") is None
    assert runtime_security.network_allowed() is True
    assert runtime_security.host_execution_allowed() is False
    assert runtime_security.remote_code_allowed() is True
    assert runtime_security.unverified_provenance_allowed() is True
    assert runtime_security.third_party_plugins_allowed() is True
    assert runtime_security.running_inside_container() is True
    assert runtime_security.current_execution_mode() == "container"

    with runtime_security.runtime_allowances_scope(
        allow_network=True,
        allow_host_execution=True,
        allow_remote_code=True,
        allow_unverified_provenance=True,
        allow_third_party_plugins=True,
    ):
        assert runtime_security.network_allowed() is True
        assert runtime_security.host_execution_allowed() is True
        assert runtime_security.remote_code_allowed() is True
        assert runtime_security.unverified_provenance_allowed() is True
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

    encoded = runtime_security_helpers.serialize_canonical_json(payload)
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
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image, engine=None: (
            image == runtime_security.RUNTIME_IMAGE_LOCAL_DEFAULT
        ),
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


def test_resolve_runtime_image_prefers_local_cuda_when_gpu_visible(monkeypatch) -> None:
    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_ENV, raising=False)
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
        lambda image, engine=None: (
            image == runtime_security.RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT
        ),
        raising=True,
    )

    assert (
        runtime_security.resolve_runtime_image()
        == runtime_security.RUNTIME_IMAGE_CUDA_LOCAL_DEFAULT
    )


def test_inspect_container_image_parses_repo_digest_and_image_id(monkeypatch) -> None:
    repo_digest = "sha256:" + "a" * 64
    image_id = "sha256:" + "b" * 64
    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                '["ghcr.io/invarlock/invarlock-runtime:test@'
                f'{repo_digest}"]\n{image_id}\n'
            ),
        ),
        raising=True,
    )
    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        True,
        repo_digest,
    )

    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"not-json\n{image_id}\n",
        ),
        raising=True,
    )
    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        True,
        image_id,
    )


def test_inspect_container_image_handles_failures_and_digestless_images(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout=""),
        raising=True,
    )
    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        False,
        None,
    )

    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[]\nimage-id\n"),
        raising=True,
    )
    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        False,
        None,
    )

    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
        raising=True,
    )
    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        False,
        None,
    )

    monkeypatch.setattr(
        runtime_security_helpers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='["ghcr.io/invarlock/runtime:test"]\nimage-id\n',
        ),
        raising=True,
    )
    assert runtime_security_helpers._inspect_container_image("docker", "img") == (
        False,
        None,
    )


def test_observed_container_image_rejects_declared_digest_mismatch(
    monkeypatch,
) -> None:
    image_id = "sha256:" + "b" * 64
    repo_digest = "sha256:" + "c" * 64
    monkeypatch.setenv(
        runtime_security.RUNTIME_IMAGE_DIGEST_ENV,
        "sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_inspect_container_image_identity",
        lambda engine, image: runtime_security_helpers._ContainerImageInspection(
            image_id=image_id,
            repo_digests=(f"registry.example/runtime@{repo_digest}",),
        ),
    )

    with pytest.raises(RuntimeError, match="does not match the observed"):
        runtime_security_helpers._resolve_observed_container_image(
            "docker", "registry.example/runtime:release"
        )


def test_declared_runtime_image_digest_rejects_malformed_or_conflicting_pins(
    monkeypatch,
) -> None:
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64
    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, "sha256:bad")
    with pytest.raises(RuntimeError, match="must be lowercase"):
        runtime_security_helpers._declared_runtime_image_digest("runtime:release")

    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, digest_a.upper())
    with pytest.raises(RuntimeError, match="must be lowercase"):
        runtime_security_helpers._declared_runtime_image_digest("runtime:release")

    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, raising=False)
    with pytest.raises(RuntimeError, match="must be lowercase"):
        runtime_security_helpers._declared_runtime_image_digest("runtime@sha256:bad")

    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, digest_a)
    with pytest.raises(RuntimeError, match="does not match the image reference"):
        runtime_security_helpers._declared_runtime_image_digest(f"runtime@{digest_b}")


def test_observed_container_image_rejects_failed_identity_inspection(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_security_helpers,
        "_inspect_container_image_identity",
        lambda engine, image: None,
    )

    with pytest.raises(RuntimeError, match="immutable local identity inspection"):
        runtime_security_helpers._resolve_observed_container_image(
            "docker", "runtime:release"
        )


def test_observed_container_image_selects_declared_immutable_repo_digest(
    monkeypatch,
) -> None:
    image_id = "sha256:" + "b" * 64
    repo_digest = "sha256:" + "c" * 64
    immutable_ref = f"registry.example/runtime@{repo_digest}"
    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, repo_digest)
    monkeypatch.setattr(
        runtime_security_helpers,
        "_inspect_container_image_identity",
        lambda engine, image: runtime_security_helpers._ContainerImageInspection(
            image_id=image_id,
            repo_digests=(immutable_ref,),
        ),
    )

    observed = runtime_security_helpers._resolve_observed_container_image(
        "docker", "registry.example/runtime:release"
    )

    assert observed.immutable_ref == immutable_ref
    assert observed.image_digest == repo_digest


def test_observed_container_image_accepts_declared_image_id(monkeypatch) -> None:
    image_id = "sha256:" + "d" * 64
    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, image_id)
    monkeypatch.setattr(
        runtime_security_helpers,
        "_inspect_container_image_identity",
        lambda engine, image: runtime_security_helpers._ContainerImageInspection(
            image_id=image_id,
            repo_digests=(),
        ),
    )

    observed = runtime_security_helpers._resolve_observed_container_image(
        "docker", "runtime:release"
    )

    assert observed.immutable_ref == image_id


def test_observed_container_image_selects_repo_digest_without_declaration(
    monkeypatch,
) -> None:
    image_id = "sha256:" + "d" * 64
    repo_digest = "sha256:" + "e" * 64
    immutable_ref = f"registry.example/runtime@{repo_digest}"
    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "_inspect_container_image_identity",
        lambda engine, image: runtime_security_helpers._ContainerImageInspection(
            image_id=image_id,
            repo_digests=(immutable_ref,),
        ),
    )

    observed = runtime_security_helpers._resolve_observed_container_image(
        "docker", "runtime:release"
    )

    assert observed.immutable_ref == immutable_ref
    assert observed.image_digest == repo_digest


def test_observed_container_image_uses_image_id_for_local_build(monkeypatch) -> None:
    image_id = "sha256:" + "d" * 64
    monkeypatch.delenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, raising=False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "_inspect_container_image_identity",
        lambda engine, image: runtime_security_helpers._ContainerImageInspection(
            image_id=image_id,
            repo_digests=(),
        ),
    )

    observed = runtime_security_helpers._resolve_observed_container_image(
        "docker", "invarlock-runtime:local"
    )

    assert observed.immutable_ref == image_id
    assert observed.image_digest == image_id


def test_container_launch_inspects_once_and_executes_observed_immutable_ref(
    monkeypatch, tmp_path: Path
) -> None:
    image = "registry.example/runtime:release"
    image_id = "sha256:" + "b" * 64
    repo_digest = "sha256:" + "c" * 64
    immutable_ref = f"registry.example/runtime@{repo_digest}"
    inspections: list[tuple[str, str]] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_ENV, image)
    monkeypatch.setenv(runtime_security.RUNTIME_IMAGE_DIGEST_ENV, repo_digest)
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda selected, engine=None: True,
    )
    monkeypatch.setattr(runtime_security_helpers, "network_allowed", lambda: False)
    monkeypatch.setattr(
        runtime_security_helpers,
        "_container_pythonpath_entries",
        lambda *, cwd: ([], []),
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "_delegated_env_pairs",
        lambda *, cwd: ({}, []),
    )

    def inspect(engine: str, selected: str):
        inspections.append((engine, selected))
        return runtime_security_helpers._ContainerImageInspection(
            image_id=image_id,
            repo_digests=(immutable_ref,),
        )

    monkeypatch.setattr(
        runtime_security_helpers,
        "_inspect_container_image_identity",
        inspect,
    )
    plan = runtime_security.ContainerLaunchPlan(
        argv=("evaluate", "--help"),
        argv_mounts=(),
        needs_cwd_host_mirror=False,
        gpu_passthrough=False,
    )

    command = runtime_security.build_container_command(plan)

    assert inspections == [("docker", image)]
    assert command[-3:] == [immutable_ref, "evaluate", "--help"]
    assert image not in command
    assert f"{runtime_security.RUNTIME_IMAGE_ENV}={immutable_ref}" in command
    assert f"{runtime_security.RUNTIME_IMAGE_DIGEST_ENV}={repo_digest}" in command


def test_normalized_launch_plan_can_require_read_only_source_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(runtime_security.SOURCE_BUNDLE_READ_ONLY_ENV, "1")
    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )

    plan = runtime_security.normalize_delegated_argv(["verify"], cwd=tmp_path)

    assert plan.workspace_read_only is True


def test_container_engine_and_device_helpers(monkeypatch) -> None:
    monkeypatch.setenv(runtime_security.CONTAINER_ENGINE_ENV, "podman")
    monkeypatch.setattr(
        runtime_security_helpers.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"docker", "podman"} else None,
        raising=True,
    )
    assert runtime_security.resolve_container_engine() == "podman"

    monkeypatch.setenv(runtime_security.CONTAINER_ENGINE_ENV, "bogus")
    monkeypatch.setattr(
        runtime_security_helpers.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
        raising=True,
    )
    assert runtime_security.resolve_container_engine() is None

    monkeypatch.delenv(runtime_security.CONTAINER_ENGINE_ENV, raising=False)
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
        runtime_security_helpers.shutil,
        "which",
        lambda name: None,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers.Path,
        "exists",
        lambda self: str(self) == "/dev/nvidiactl",
        raising=False,
    )

    assert runtime_security.resolve_container_engine() is None
    assert runtime_security_helpers._host_nvidia_visible() is True
    assert runtime_launch_plan._requested_device(["--help"]) is None


def test_container_image_available_locally(monkeypatch) -> None:
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
