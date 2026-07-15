from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.scripts._tensorrt_llm_fixture_support import (
    canary_payload as _canary_payload,
)
from tests.scripts._tensorrt_llm_fixture_support import (
    fixture,
)
from tests.scripts._tensorrt_llm_fixture_support import (
    identity as _identity,
)
from tests.scripts._tensorrt_llm_fixture_support import (
    qualification_summary as _qualification_summary,
)
from tests.scripts._tensorrt_llm_fixture_support import (
    valid_manifest as _valid_manifest,
)

_REAL_VALIDATE_HARDWARE = fixture._validate_hardware
_EXPECTED_ARTIFACT_SHA256 = fixture.artifact_identity_sha256(_identity())


@pytest.fixture(autouse=True)
def _stub_exact_base_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fixture,
        "_validate_hardware",
        lambda **_kwargs: (
            ("GPU-01234567-89ab-cdef-0123-456789abcdef", "9.0"),
            ("GPU-fedcba98-7654-3210-fedc-ba9876543210", "9.0"),
        ),
    )


def test_model_inventory_is_stable_and_rejects_links(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    first = fixture._model_inventory_sha256(model)
    assert first == fixture._model_inventory_sha256(model)
    (model / "alias").symlink_to("config.json")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="symlink"):
        fixture._model_inventory_sha256(model)


def test_parse_object_rejects_duplicate_keys_and_nonfinite() -> None:
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="duplicate"):
        fixture._parse_object(b'{"ok":true,"ok":false}', label="test")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="NaN"):
        fixture._parse_object(b'{"value":NaN}', label="test")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="strict JSON"):
        fixture._parse_object(b"not-json", label="test")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="strict JSON"):
        fixture._parse_object(b'vendor banner\n{"ok":true}', label="test")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="JSON object"):
        fixture._parse_object(b"[]", label="test")


def test_build_requires_reviewed_owned_model_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: "sha256:" + "a" * 64)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="reviewed inventory"):
        fixture.build_fixture(
            engine="docker",
            image="candidate",
            model=model,
            output=tmp_path / "result",
            selectors=("device=0", "device=1"),
            expected_model_inventory_sha256="f" * 64,
        )
    owned = tmp_path / "result" / ".inputs" / "model"
    assert owned != model
    assert (owned / "config.json").read_text(encoding="utf-8") == "{}"


def test_owned_snapshot_is_no_clobber_and_rejects_links(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_bytes(b"content")
    destination = tmp_path / "snapshot"
    fixture._snapshot_model_tree(source, destination)
    assert (destination / "file").read_bytes() == b"content"
    with pytest.raises(fixture.TensorRTLLMFixtureError):
        fixture._snapshot_model_tree(source, destination)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "file").symlink_to(source / "file")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="single-link regular"):
        fixture._snapshot_model_tree(linked, tmp_path / "linked-snapshot")
    with pytest.raises(
        fixture.TensorRTLLMFixtureError, match="source must be a directory"
    ):
        fixture._snapshot_model_tree(source / "file", tmp_path / "file-snapshot")
    directory_link = tmp_path / "directory-link"
    directory_link.mkdir()
    (directory_link / "child").symlink_to(source, target_is_directory=True)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="non-directory entry"):
        fixture._snapshot_model_tree(directory_link, tmp_path / "directory-snapshot")


def test_file_and_inventory_error_paths(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="non-empty"):
        fixture._sha256_file(empty)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="read safely"):
        fixture._sha256_file(tmp_path / "missing")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="unavailable"):
        fixture._model_inventory_sha256(tmp_path / "missing-model")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="directory"):
        fixture._model_inventory_sha256(empty)
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="empty"):
        fixture._model_inventory_sha256(model)
    original = model / "weights"
    original.write_bytes(b"x")
    os.link(original, model / "hardlink")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="hard link"):
        fixture._model_inventory_sha256(model)

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "weights").write_bytes(b"x")
    (clean / "linked-directory").symlink_to(clean, target_is_directory=True)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="symlink"):
        fixture._model_inventory_sha256(clean)

    special = tmp_path / "special"
    special.mkdir()
    os.mkfifo(special / "fifo")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="special file"):
        fixture._model_inventory_sha256(special)


def test_file_authentication_detects_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"content")
    real_fstat = fixture.os.fstat
    calls = 0

    def changed_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        metadata = real_fstat(descriptor)
        if calls == 2:
            return type(
                "Changed",
                (),
                {
                    "st_dev": metadata.st_dev,
                    "st_ino": metadata.st_ino,
                    "st_mode": metadata.st_mode,
                    "st_size": metadata.st_size,
                    "st_mtime_ns": metadata.st_mtime_ns + 1,
                },
            )()
        return metadata

    monkeypatch.setattr(fixture.os, "fstat", changed_fstat)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="changed"):
        fixture._sha256_file(path)


def test_manifest_and_canary_validation_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="cannot be read"):
        fixture._load_manifest(missing)
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="schema"):
        fixture._load_manifest(path)

    manifest = _valid_manifest(replace(_identity(), tokenizer_metadata_sha256="2" * 64))
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="bounded"):
        fixture._load_manifest(path)
    manifest["tokenizer_sha256"] = "bad"
    path.write_bytes(fixture._canonical_json(manifest))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="tokenizer_sha256"):
        fixture._load_manifest(path)
    manifest["tokenizer_sha256"] = "2" * 64
    manifest["candidate_image_digest"] = "latest"
    path.write_bytes(fixture._canonical_json(manifest))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="image digest"):
        fixture._load_manifest(path)

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="identity"):
        fixture._canary_one(
            engine="docker",
            image="candidate",
            image_digest="sha256:" + "a" * 64,
            selector="device=0",
            fixture=frozen,
            manifest={"selected_engine_identity": "bad"},
            expected_artifact_identity_sha256=_EXPECTED_ARTIFACT_SHA256,
        )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="engine digest"):
        fixture._canary_one(
            engine="docker",
            image="candidate",
            image_digest="sha256:" + "a" * 64,
            selector="device=0",
            fixture=frozen,
            manifest={"selected_engine_identity": {}},
            expected_artifact_identity_sha256=_EXPECTED_ARTIFACT_SHA256,
        )
    manifest = {
        "selected_engine_identity": {"engine_bundle_tree_sha256": "1" * 64},
        "tokenizer_sha256": "2" * 64,
        "expected_output_sha256": "3" * 64,
    }
    monkeypatch.setattr(fixture, "_run_captured", lambda *_a, **_k: (2, b"", b"bad"))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="canary failed"):
        fixture._canary_one(
            engine="docker",
            image="candidate",
            image_digest="sha256:" + "a" * 64,
            selector="device=0",
            fixture=frozen,
            manifest=manifest,
            expected_artifact_identity_sha256=_EXPECTED_ARTIFACT_SHA256,
        )
    monkeypatch.setattr(fixture, "_run_captured", lambda *_a, **_k: (0, b"{}", b""))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="open schema"):
        fixture._canary_one(
            engine="docker",
            image="candidate",
            image_digest="sha256:" + "a" * 64,
            selector="device=0",
            fixture=frozen,
            manifest=manifest,
            expected_artifact_identity_sha256=_EXPECTED_ARTIFACT_SHA256,
        )


def test_qualification_rejects_changed_bindings_and_cross_gpu_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    frozen = root / "fixture"
    (frozen / "engine").mkdir(parents=True)
    (frozen / "tokenizer.json").write_bytes(b"tokenizer")
    tokenizer_sha = fixture._sha256_file(frozen / "tokenizer.json")
    identity = replace(_identity(), tokenizer_metadata_sha256=tokenizer_sha)
    manifest = {
        "candidate_image_digest": "sha256:" + "a" * 64,
        "expected_output_sha256": "7" * 64,
        "selected_engine_identity": fixture.asdict(identity),
        "tokenizer_sha256": tokenizer_sha,
    }
    monkeypatch.setattr(fixture, "_load_manifest", lambda _path: manifest)
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: "sha256:" + "b" * 64)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="image differs"):
        fixture.qualify_two_gpu(
            engine="docker",
            image="candidate",
            fixture_root=root,
            output=tmp_path / "one.json",
            selectors=("device=0", "device=1"),
        )
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: "sha256:" + "a" * 64)
    manifest["tokenizer_sha256"] = "8" * 64
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="tokenizer binding"):
        fixture.qualify_two_gpu(
            engine="docker",
            image="candidate",
            fixture_root=root,
            output=tmp_path / "two.json",
            selectors=("device=0", "device=1"),
        )
    manifest["tokenizer_sha256"] = tokenizer_sha
    monkeypatch.setattr(
        fixture,
        "read_tensorrt_llm_artifact_identity",
        lambda *_a, **_k: replace(
            identity,
            bundle_name=f"tensorrt-llm-sha256-{'9' * 64}",
            engine_bundle_tree_sha256="9" * 64,
        ),
    )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="engine identity"):
        fixture.qualify_two_gpu(
            engine="docker",
            image="candidate",
            fixture_root=root,
            output=tmp_path / "three.json",
            selectors=("device=0", "device=1"),
        )
    monkeypatch.setattr(
        fixture, "read_tensorrt_llm_artifact_identity", lambda *_a, **_k: identity
    )
    first = _canary_payload()
    second = _canary_payload()
    second["runtime_provider_receipt_sha256"] = "f" * 64
    results = iter((first, second))
    monkeypatch.setattr(fixture, "_canary_one", lambda **_k: next(results))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="evidence differs"):
        fixture.qualify_two_gpu(
            engine="docker",
            image="candidate",
            fixture_root=root,
            output=tmp_path / "four.json",
            selectors=("device=0", "device=1"),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(backend_version="1.2.2"), "backend version"),
        (lambda value: value["build_recipe"].update(max_batch_size=2), "build recipe"),
        (
            lambda value: value["model"].update(repository="other/model"),
            "model binding",
        ),
        (lambda value: value["worker"].update(sha256="bad"), "worker binding"),
        (lambda value: value.update(engine_builds={}), "engine builds"),
        (
            lambda value: value["engine_builds"]["primary"].update(extra=True),
            "unexpected schema",
        ),
        (
            lambda value: value["engine_builds"]["primary"].update(
                builder_config_sha256="bad"
            ),
            "builder_config_sha256",
        ),
        (
            lambda value: value["engine_builds"]["primary"].update(
                target_compute_capability="bad"
            ),
            "identity is invalid",
        ),
        (
            lambda value: value["engine_builds"]["primary"].update(bundle_name="other"),
            "not canonical",
        ),
        (
            lambda value: value["engine_builds"]["primary"].update(
                bundle_name=f"tensorrt-llm-sha256-{'e' * 64}"
            ),
            "not canonical",
        ),
        (
            lambda value: value["engine_builds"]["primary"].update(
                tokenizer_metadata_sha256="f" * 64
            ),
            "tokenizer binding",
        ),
        (
            lambda value: value["selected_engine_identity"].update(
                bundle_name=f"tensorrt-llm-sha256-{'e' * 64}",
                engine_bundle_tree_sha256="e" * 64,
            ),
            "not the primary",
        ),
        (
            lambda value: value.update(engine_byte_reproduction="different"),
            "reproduction",
        ),
    ],
)
def test_nested_manifest_contract_is_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest = json.loads(json.dumps(_valid_manifest(_identity())))
    mutation(manifest)
    path = tmp_path / "manifest.json"
    path.write_bytes(fixture._canonical_json(manifest))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match=message):
        fixture._load_manifest(path)


def test_manifest_accepts_distinct_canonical_engine_builds(tmp_path: Path) -> None:
    primary = _identity()
    secondary = _identity(tree="6" * 64)
    manifest = _valid_manifest(primary)
    manifest["engine_builds"]["secondary"] = fixture.asdict(secondary)
    manifest["engine_byte_reproduction"] = "different"
    path = tmp_path / "fixture-manifest.json"
    path.write_bytes(fixture._canonical_json(manifest))

    loaded = fixture._load_manifest(path)

    assert loaded["engine_builds"]["primary"] == fixture.asdict(primary)
    assert loaded["engine_builds"]["secondary"] == fixture.asdict(secondary)
    assert loaded["engine_byte_reproduction"] == "different"


def test_canary_requires_exact_schema_and_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "selected_engine_identity": {"engine_bundle_tree_sha256": "1" * 64},
        "tokenizer_sha256": "4" * 64,
        "expected_output_sha256": "7" * 64,
    }

    def run(payload: dict[str, object]) -> None:
        monkeypatch.setattr(
            fixture,
            "_run_captured",
            lambda *_a, **_k: (0, fixture._canonical_json(payload), b""),
        )
        fixture._canary_one(
            engine="docker",
            image="sha256:" + "a" * 64,
            image_digest="sha256:" + "a" * 64,
            selector="device=0",
            fixture=tmp_path,
            manifest=manifest,
            expected_artifact_identity_sha256=_EXPECTED_ARTIFACT_SHA256,
        )

    extra = _canary_payload()
    extra["extra"] = True
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="open schema"):
        run(extra)
    missing_receipt_digest = _canary_payload()
    del missing_receipt_digest["runtime_provider_receipt_sha256"]
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="open schema"):
        run(missing_receipt_digest)
    wrong_output = _canary_payload(output="f" * 64)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="result is invalid"):
        run(wrong_output)
    wrong_artifact = _canary_payload(artifact="f" * 64)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="result is invalid"):
        run(wrong_artifact)
    bad_digest = _canary_payload()
    bad_digest["scoring_observation_sha256"] = "bad"
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="digest is invalid"):
        run(bad_digest)
    bad_receipt_digest = _canary_payload()
    bad_receipt_digest["runtime_provider_receipt_sha256"] = "bad"
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="digest is invalid"):
        run(bad_receipt_digest)


def test_canary_command_isolated_and_mounts_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command: list[str] = []

    def run(captured: list[str], **_kwargs: object):
        command.extend(captured)
        return (0, fixture._canonical_json(_canary_payload()), b"")

    monkeypatch.setattr(fixture, "_run_captured", run)
    fixture._canary_one(
        engine="docker",
        image="sha256:" + "a" * 64,
        image_digest="sha256:" + "a" * 64,
        selector="device=0",
        fixture=tmp_path,
        manifest={
            "expected_output_sha256": "7" * 64,
            "selected_engine_identity": {"engine_bundle_tree_sha256": "1" * 64},
            "tokenizer_sha256": "4" * 64,
        },
        expected_artifact_identity_sha256=_EXPECTED_ARTIFACT_SHA256,
    )

    assert "invarlock.runtime_providers.tensorrt_llm_canary" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command.count("--tmpfs") == 1
    assert (
        command[command.index("--tmpfs") + 1] == "/tmp:rw,noexec,nosuid,nodev,size=8g"
    )
    assert all(
        mount.endswith(":ro")
        for index, mount in enumerate(command)
        if index > 0 and command[index - 1] == "--volume"
    )


def test_promotion_tags_only_bound_immutable_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path = tmp_path / "qualification.json"
    summary_path.write_bytes(fixture._canonical_json(_qualification_summary()))
    inspections = iter(("sha256:" + "a" * 64, "sha256:" + "a" * 64))
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: next(inspections))
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **_kwargs: object):
        commands.append(command)
        return (0, b"", b"")

    monkeypatch.setattr(fixture, "_run_captured", run)
    result = fixture.promote_candidate(
        engine="docker",
        image="candidate:qualified",
        qualification_summary=summary_path,
        stable_tag="invarlock-runtime:tensorrt-llm-local",
    )
    assert result["ok"] is True
    assert commands == [
        (
            "docker",
            "image",
            "tag",
            "sha256:" + "a" * 64,
            "invarlock-runtime:tensorrt-llm-local",
        )
    ]
    assert "candidate:qualified" not in commands[0]


def test_promotion_fails_closed_on_tag_race_and_invalid_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "qualification.json"
    path.write_bytes(fixture._canonical_json(_qualification_summary()))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="stable image tag"):
        fixture.promote_candidate(
            engine="docker",
            image="candidate:qualified",
            qualification_summary=path,
            stable_tag="sha256:" + "a" * 64,
        )
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: "sha256:" + "b" * 64)
    with pytest.raises(
        fixture.TensorRTLLMFixtureError, match="changed after qualification"
    ):
        fixture.promote_candidate(
            engine="docker",
            image="candidate:qualified",
            qualification_summary=path,
            stable_tag="invarlock-runtime:tensorrt-llm-local",
        )
    invalid = _qualification_summary()
    invalid["extra"] = True
    path.write_bytes(fixture._canonical_json(invalid))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="invalid schema"):
        fixture.promote_candidate(
            engine="docker",
            image="candidate:qualified",
            qualification_summary=path,
            stable_tag="invarlock-runtime:tensorrt-llm-local",
        )
    invalid = _qualification_summary()
    invalid["candidate_image_digest"] = "latest"
    path.write_bytes(fixture._canonical_json(invalid))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="image digest"):
        fixture._load_qualification_summary(path)
    invalid = _qualification_summary()
    invalid["output_sha256"] = "bad"
    path.write_bytes(fixture._canonical_json(invalid))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="output_sha256"):
        fixture._load_qualification_summary(path)
    invalid = _qualification_summary()
    invalid["runtime_provider_receipt_sha256"] = "bad"
    path.write_bytes(fixture._canonical_json(invalid))
    with pytest.raises(
        fixture.TensorRTLLMFixtureError,
        match="runtime_provider_receipt_sha256",
    ):
        fixture._load_qualification_summary(path)
    invalid = _qualification_summary()
    del invalid["runtime_provider_receipt_sha256"]
    path.write_bytes(fixture._canonical_json(invalid))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="invalid schema"):
        fixture._load_qualification_summary(path)


def test_promotion_rejects_tag_failure_and_wrong_stable_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "qualification.json"
    path.write_bytes(fixture._canonical_json(_qualification_summary()))
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: "sha256:" + "a" * 64)
    monkeypatch.setattr(fixture, "_run_captured", lambda *_a, **_k: (2, b"", b"bad"))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="could not be promoted"):
        fixture.promote_candidate(
            engine="docker",
            image="candidate:qualified",
            qualification_summary=path,
            stable_tag="invarlock-runtime:tensorrt-llm-local",
        )
    inspections = iter(("sha256:" + "a" * 64, "sha256:" + "b" * 64))
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: next(inspections))
    monkeypatch.setattr(fixture, "_run_captured", lambda *_a, **_k: (0, b"", b""))
    with pytest.raises(
        fixture.TensorRTLLMFixtureError, match="stable tag does not bind"
    ):
        fixture.promote_candidate(
            engine="docker",
            image="candidate:qualified",
            qualification_summary=path,
            stable_tag="invarlock-runtime:tensorrt-llm-local",
        )
