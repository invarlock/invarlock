"""Closed runtime profiles preserve explicit origins and pinned OCI validation."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.cli.runtime_profile import load_runtime_profile, resolve_runtime_profile
from invarlock.evaluation_oci import launch_from_resolved_config
from invarlock.evidence_pack_json import StrictJsonError


def _profile(tmp_path: Path, **extra):
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "format": "invarlock/runtime-profile-v1",
                "runtime": {"device": "cpu"},
                **extra,
            }
        )
    )
    return path


def test_side_and_common_cli_precede_profile_and_environment(tmp_path):
    profile = load_runtime_profile(
        _profile(tmp_path, runtime={"device": "cuda:0"}, subject={"device": "cuda:1"})
    )
    result = resolve_runtime_profile(
        profile,
        explicit={"runtime_device": "cuda:2", "baseline_runtime_device": "cuda:3"},
        environment={"INVARLOCK_SUBJECT_RUNTIME_DEVICE": "cuda:7"},
    )
    assert result.baseline.device == "cuda:3"
    assert result.subject.device == "cuda:2"
    assert result.sources["baseline.device"] == "--baseline-runtime-device"
    assert result.sources["subject.device"] == "--runtime-device"
    no_cli = resolve_runtime_profile(
        profile,
        explicit={},
        environment={"INVARLOCK_BASELINE_RUNTIME_DEVICE": "cuda:7"},
    )
    assert no_cli.baseline.device == "cuda:0"
    assert no_cli.subject.device == "cuda:1"


@pytest.mark.parametrize(
    "payload",
    [
        {"runtime": {"network": True}},
        {"runtime": {"signing_key": "private.pem"}},
        {"runtime": {"memory_mib": True}},
        {"runtime": {"memory_mib": 128.0}},
        {"runtime": {"cpus": 4}},
        {"runtime": {"device": ""}},
        {"runtime": None},
        {"subject": {"cpus": "2"}},
        {"baseline": []},
        {"other": True},
    ],
)
def test_closed_profile_rejects_ambiguous_or_authority_fields(tmp_path, payload):
    with pytest.raises(ValueError):
        load_runtime_profile(_profile(tmp_path, **payload))


def test_duplicate_oversize_and_symlink_profiles_rejected(tmp_path):
    path = _profile(tmp_path)
    path.write_text(
        '{"format":"invarlock/runtime-profile-v1","runtime":{},"runtime":{}}'
    )
    with pytest.raises(StrictJsonError, match="duplicate"):
        load_runtime_profile(path)
    path.write_bytes(b" " * (16384 + 1))
    with pytest.raises(StrictJsonError, match="limit"):
        load_runtime_profile(path)
    target = _profile(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(StrictJsonError, match="symlink"):
        load_runtime_profile(link)


def test_image_and_digest_override_conflict_is_not_silently_repaired(tmp_path):
    old = "sha256:" + "a" * 64
    new = "sha256:" + "b" * 64
    profile = load_runtime_profile(
        _profile(
            tmp_path,
            runtime={"image": "registry.example/worker@" + old},
            subject={"image_digest": old},
        )
    )
    with pytest.raises(ValueError, match="subject.*image.*digest"):
        resolve_runtime_profile(
            profile,
            explicit={"subject_runtime_image": "registry.example/worker@" + new},
            environment={},
        )


def test_environment_and_original_resource_defaults_remain_explicit(tmp_path):
    profile = load_runtime_profile(_profile(tmp_path, runtime={}))
    result = resolve_runtime_profile(
        profile,
        explicit={},
        environment={
            "INVARLOCK_BASELINE_RUNTIME_DEVICE": "cuda:1",
            "INVARLOCK_RUNTIME_MEMORY_MIB": "2048",
        },
    )
    assert result.baseline.device == "cuda:1"
    assert result.subject.device == "cpu"
    assert result.memory_mib == 2048
    assert result.cpus == "4"
    assert result.sources["baseline.device"] == "INVARLOCK_BASELINE_RUNTIME_DEVICE"
    assert result.sources["subject.device"] == "default"


def test_launch_uses_the_closed_resolution_after_environment_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    digest = "sha256:" + "a" * 64
    profile = load_runtime_profile(
        _profile(tmp_path, runtime={"image": "registry.example/runtime:stable"})
    )
    monkeypatch.setattr(
        "invarlock.evaluation_oci.shutil.which", lambda _, **_kwargs: "/bin/docker"
    )
    resolved = resolve_runtime_profile(
        profile,
        explicit={"runtime_image_digest": digest},
        environment={},
    )
    monkeypatch.setattr(
        "invarlock.evaluation_oci._inspect_local_image",
        lambda *_: SimpleNamespace(
            repo_digests=(f"registry.example/runtime@{digest}",),
            config_id="sha256:" + "b" * 64,
        ),
    )
    monkeypatch.setenv("INVARLOCK_CONTAINER_ENGINE", "podman")
    monkeypatch.setenv("INVARLOCK_RUNTIME_DEVICE", "cuda:7")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", "registry.example/changed:latest")
    launch = launch_from_resolved_config(resolved)
    assert launch.engine == "docker"
    assert launch.baseline.image_digest == digest
    assert launch.baseline.device == "cpu"
    assert launch.baseline.image_ref == f"registry.example/runtime@{digest}"
    with pytest.raises(FrozenInstanceError):
        resolved.baseline.image_ref = "changed"  # type: ignore[misc]


def test_exact_runtime_profile_byte_limit_accepts_then_one_more_rejects(tmp_path):
    from invarlock.cli.runtime_profile import MAX_RUNTIME_PROFILE_BYTES

    path = _profile(tmp_path, runtime={"device": "cpu", "memory_mib": 4096})
    content = path.read_bytes()
    path.write_bytes(content + b" " * (MAX_RUNTIME_PROFILE_BYTES - len(content)))
    loaded = load_runtime_profile(path)
    assert path.stat().st_size == MAX_RUNTIME_PROFILE_BYTES
    assert loaded.runtime == {"device": "cpu", "memory_mib": "4096"}
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(StrictJsonError, match="limit"):
        load_runtime_profile(path)


_SIDE_VALUES = {
    "image": [f"registry.example/runtime:{index}" for index in range(6)] + [""],
    "image_digest": ["sha256:" + str(index) * 64 for index in range(6)] + [""],
    "device": [f"cuda:{index}" for index in range(6)] + ["cpu"],
    "entrypoint": ["python", "nvidia", "auto", "python", "nvidia", "auto", "auto"],
}


@pytest.mark.parametrize("field", _SIDE_VALUES)
@pytest.mark.parametrize("side", ["baseline", "subject"])
@pytest.mark.parametrize("winner", range(7))
def test_every_side_setting_follows_the_complete_precedence_order(field, side, winner):
    from invarlock.cli.runtime_profile import RuntimeProfile

    values = _SIDE_VALUES[field]
    explicit = {}
    environment = {}
    common = {}
    sides = {"baseline": {}, "subject": {}}
    candidates = [
        (explicit, f"{side}_runtime_{field}"),
        (explicit, f"runtime_{field}"),
        (sides[side], field),
        (common, field),
        (environment, f"INVARLOCK_{side.upper()}_RUNTIME_{field.upper()}"),
        (environment, f"INVARLOCK_RUNTIME_{field.upper()}"),
    ]
    for index, (mapping, name) in enumerate(candidates):
        if index >= winner:
            mapping[name] = values[index]
    profile = RuntimeProfile("sha256:" + "f" * 64, common, **sides)
    resolved = resolve_runtime_profile(
        profile, explicit=explicit, environment=environment
    )
    expected_sources = [
        f"--{side}-runtime-{field.replace('_', '-')}",
        f"--runtime-{field.replace('_', '-')}",
        f"profile.{side}.{field}",
        f"profile.runtime.{field}",
        f"INVARLOCK_{side.upper()}_RUNTIME_{field.upper()}",
        f"INVARLOCK_RUNTIME_{field.upper()}",
        "default",
    ]
    attribute = "image_ref" if field == "image" else field
    assert getattr(getattr(resolved, side), attribute) == values[winner]
    assert resolved.sources[f"{side}.{field}"] == expected_sources[winner]


_COMMON_VALUES = {
    "engine": ["podman", "docker", "podman", "docker"],
    "cpus": ["1", "2", "3", "4"],
    "memory_mib": ["1024", "2048", "4096", "65536"],
    "user": ["1000:1000", "2000:2000", "3000:3000", "65532:65532"],
}


@pytest.mark.parametrize("field", _COMMON_VALUES)
@pytest.mark.parametrize("winner", range(4))
def test_every_common_setting_follows_cli_profile_environment_default(field, winner):
    from invarlock.cli.runtime_profile import RuntimeProfile

    explicit_name = "container_engine" if field == "engine" else "runtime_" + field
    environment_name = (
        "INVARLOCK_CONTAINER_ENGINE"
        if field == "engine"
        else "INVARLOCK_RUNTIME_" + field.upper()
    )
    values = _COMMON_VALUES[field]
    resolved = resolve_runtime_profile(
        RuntimeProfile(
            "sha256:" + "f" * 64, {field: values[1]} if winner <= 1 else {}, {}, {}
        ),
        explicit={explicit_name: values[0]} if winner == 0 else {},
        environment={environment_name: values[2]} if winner <= 2 else {},
    )
    expected = int(values[winner]) if field == "memory_mib" else values[winner]
    assert getattr(resolved, field) == expected
    assert (
        resolved.sources["runtime." + field]
        == [
            "--" + explicit_name.replace("_", "-"),
            "profile.runtime." + field,
            environment_name,
            "default",
        ][winner]
    )


@pytest.mark.parametrize("field", _SIDE_VALUES)
@pytest.mark.parametrize("side", ["baseline", "subject"])
@pytest.mark.parametrize("winner", [0, 1, 4, 5, 6])
def test_no_profile_uses_the_same_side_precedence(field, side, winner):
    values = _SIDE_VALUES[field]
    explicit = {}
    environment = {}
    candidates = [
        (0, explicit, f"{side}_runtime_{field}"),
        (1, explicit, f"runtime_{field}"),
        (4, environment, f"INVARLOCK_{side.upper()}_RUNTIME_{field.upper()}"),
        (5, environment, f"INVARLOCK_RUNTIME_{field.upper()}"),
    ]
    for index, mapping, name in candidates:
        if index >= winner:
            mapping[name] = values[index]
    resolved = resolve_runtime_profile(None, explicit=explicit, environment=environment)
    attribute = "image_ref" if field == "image" else field
    assert getattr(getattr(resolved, side), attribute) == values[winner]
    assert "profile." not in " ".join(resolved.sources.values())


@pytest.mark.parametrize("field", _COMMON_VALUES)
@pytest.mark.parametrize("winner", [0, 2, 3])
def test_no_profile_uses_the_same_common_precedence(field, winner):
    explicit_name = "container_engine" if field == "engine" else "runtime_" + field
    environment_name = (
        "INVARLOCK_CONTAINER_ENGINE"
        if field == "engine"
        else "INVARLOCK_RUNTIME_" + field.upper()
    )
    values = _COMMON_VALUES[field]
    resolved = resolve_runtime_profile(
        None,
        explicit={explicit_name: values[0]} if winner == 0 else {},
        environment={environment_name: values[2]} if winner <= 2 else {},
    )
    expected = int(values[winner]) if field == "memory_mib" else values[winner]
    assert getattr(resolved, field) == expected


@pytest.mark.parametrize("field", [*_SIDE_VALUES, *_COMMON_VALUES])
@pytest.mark.parametrize("source", ["explicit", "environment"])
@pytest.mark.parametrize("invalid", ["", "invalid value"])
def test_selected_invalid_values_never_revert_to_defaults(
    field, source, invalid, monkeypatch
):
    from invarlock.evaluation_oci import OciEvaluationError

    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        "invarlock.evaluation_oci.shutil.which", lambda _, **_kwargs: "/bin/docker"
    )
    monkeypatch.setattr(
        "invarlock.evaluation_oci._inspect_local_image",
        lambda *_: SimpleNamespace(
            repo_digests=(f"registry.example/runtime@{digest}",), config_id=digest
        ),
    )
    # Use lower-priority valid image defaults. The selected invalid value must
    # survive selection and fail validation rather than restoring those defaults.
    from invarlock.cli.runtime_profile import RuntimeProfile

    profile = (
        None
        if source == "environment"
        else RuntimeProfile(
            "sha256:" + "f" * 64,
            {"image": "registry.example/runtime:stable", "image_digest": digest},
            {},
            {},
        )
    )
    explicit = {}
    environment = {
        "INVARLOCK_RUNTIME_IMAGE": "registry.example/runtime:stable",
        "INVARLOCK_RUNTIME_IMAGE_DIGEST": digest,
    }
    if source == "explicit":
        name = "container_engine" if field == "engine" else "runtime_" + field
        explicit[name] = invalid
    else:
        name = (
            "INVARLOCK_CONTAINER_ENGINE"
            if field == "engine"
            else "INVARLOCK_RUNTIME_" + field.upper()
        )
        environment[name] = invalid
    with pytest.raises((OciEvaluationError, ValueError)):
        launch_from_resolved_config(
            resolve_runtime_profile(profile, explicit=explicit, environment=environment)
        )


def test_resolved_config_owns_immutable_values_and_sources():
    explicit = {"runtime_device": "cuda:2"}
    environment = {"INVARLOCK_RUNTIME_CPUS": "2"}
    resolved = resolve_runtime_profile(None, explicit=explicit, environment=environment)
    explicit["runtime_device"] = "cuda:7"
    environment["INVARLOCK_RUNTIME_CPUS"] = "9"
    assert resolved.baseline.device == resolved.subject.device == "cuda:2"
    assert resolved.cpus == "2"
    with pytest.raises(TypeError):
        resolved.sources["runtime.cpus"] = "changed"  # type: ignore[index]


def test_launch_requires_the_typed_complete_configuration():
    with pytest.raises(TypeError, match="ResolvedRuntimeConfig"):
        launch_from_resolved_config({})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unexpected keyword"):
        launch_from_resolved_config(engine="docker")  # type: ignore[call-arg]


@pytest.mark.parametrize("repository", ["", "registry.example/runtime@"])
def test_embedded_image_digest_is_resolved_before_construction(repository):
    digest = "sha256:" + "a" * 64
    resolved = resolve_runtime_profile(
        None, explicit={"runtime_image": repository + digest}, environment={}
    )
    assert resolved.baseline.image_digest == resolved.subject.image_digest == digest
    assert (
        resolved.sources["baseline.image_digest"] == "--runtime-image (embedded digest)"
    )


def test_explicit_empty_digest_does_not_use_embedded_fallback(monkeypatch):
    from invarlock.evaluation_oci import OciEvaluationError

    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        "invarlock.evaluation_oci.shutil.which", lambda _, **_kwargs: "/bin/docker"
    )
    resolved = resolve_runtime_profile(
        None,
        explicit={"runtime_image": digest, "runtime_image_digest": ""},
        environment={},
    )
    assert resolved.baseline.image_digest == ""
    assert resolved.sources["baseline.image_digest"] == "--runtime-image-digest"
    with pytest.raises(OciEvaluationError, match="image digest is required"):
        launch_from_resolved_config(resolved)


@pytest.mark.parametrize("engine", ["docker", "podman"])
def test_engine_executable_uses_the_resolved_path_snapshot(
    tmp_path, monkeypatch, engine
):
    digest = "sha256:" + "a" * 64
    original = tmp_path / "original"
    changed = tmp_path / "changed"
    for directory in (original, changed):
        directory.mkdir()
        executable = directory / engine
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)
    environment = {"PATH": str(original)}
    config = resolve_runtime_profile(
        None,
        explicit={"container_engine": engine, "runtime_image": digest},
        environment=environment,
    )
    assert config.engine_path == str(original / engine)
    assert config.sources["runtime.engine_path"] == "PATH"
    environment["PATH"] = str(changed)
    monkeypatch.setenv("PATH", str(changed))
    inspected = []

    def inspect(path, _image):
        inspected.append(path)
        return SimpleNamespace(repo_digests=(), config_id=digest)

    monkeypatch.setattr("invarlock.evaluation_oci._inspect_local_image", inspect)
    launch = launch_from_resolved_config(config)
    assert launch.engine_path == str(original / engine)
    # Bare configuration IDs need no tag lookup during construction. Preflight
    # must also inspect using the original selected executable.
    from invarlock.evaluation_oci import preflight_oci_launch

    preflight_oci_launch(launch)
    assert inspected == [str(original / engine)]


def test_missing_snapshot_engine_cannot_appear_after_resolution(tmp_path, monkeypatch):
    from invarlock.evaluation_oci import OciEvaluationError

    config = resolve_runtime_profile(
        None, explicit={}, environment={"PATH": str(tmp_path)}
    )
    assert config.engine_path is None
    executable = tmp_path / "docker"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(OciEvaluationError, match="not available"):
        launch_from_resolved_config(config)
