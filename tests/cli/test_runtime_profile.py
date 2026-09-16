"""Closed runtime profiles preserve explicit origins and pinned OCI validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.cli.runtime_profile import load_runtime_profile, resolve_runtime_profile
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
    assert result.arguments["baseline_device"] == "cuda:3"
    assert result.arguments["subject_device"] == "cuda:2"
    assert result.sources["baseline.device"] == "--baseline-runtime-device"
    assert result.sources["subject.device"] == "--runtime-device"
    no_cli = resolve_runtime_profile(
        profile,
        explicit={},
        environment={"INVARLOCK_BASELINE_RUNTIME_DEVICE": "cuda:7"},
    )
    assert no_cli.arguments["baseline_device"] == "cuda:0"
    assert no_cli.arguments["subject_device"] == "cuda:1"


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
    assert result.arguments["baseline_device"] == "cuda:1"
    assert result.arguments["subject_device"] == "cpu"
    assert result.arguments["runtime_memory_mib"] == "2048"
    assert result.arguments["runtime_cpus"] == "4"
    assert result.sources["baseline.device"] == "INVARLOCK_BASELINE_RUNTIME_DEVICE"
    assert result.sources["subject.device"] == "default"


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
