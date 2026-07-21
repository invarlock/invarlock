from __future__ import annotations

import json
import os
import stat
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jsonschema
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock import trust_inputs as trust_inputs_module
from invarlock.public_contracts import (
    TRUST_INPUTS_FORMAT_VERSION,
    load_trust_inputs_schema,
)
from invarlock.trust_inputs import TrustInputsError, load_trust_inputs


def _digest(marker: str) -> str:
    return "sha256:" + marker * 64


def _material(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    trust_root = tmp_path / "trust"
    trust_root.mkdir()
    (trust_root / "policy.json").write_text("{}\n", encoding="utf-8")
    key = ed25519.Ed25519PrivateKey.generate()
    (trust_root / "verifier.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    payload: dict[str, object] = {
        "format": "invarlock/trust-inputs-v1",
        "policy": {"path": "policy.json"},
        "anchors": {
            "baseline_artifact_digest": _digest("a"),
            "subject_artifact_digest": _digest("b"),
            "schedule_digest": _digest("c"),
            "baseline_runtime_digest": _digest("d"),
            "subject_runtime_digest": _digest("e"),
            "evidence_signer_fingerprint": _digest("f"),
        },
        "verifier": {
            "identity": "invarlock-verifier/release",
            "signing_key_path": "verifier.pem",
        },
        "allow_installed_scorers": False,
    }
    return trust_root, payload


def test_source_packaged_and_loaded_trust_contracts_match(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = json.loads((root / "contracts/trust_inputs.schema.json").read_text())
    packaged = json.loads(
        (root / "src/invarlock/_data/contracts/trust_inputs.schema.json").read_text()
    )
    trust_root, payload = _material(tmp_path)

    assert TRUST_INPUTS_FORMAT_VERSION == "invarlock/trust-inputs-v1"
    assert source == packaged == load_trust_inputs_schema()
    jsonschema.Draft202012Validator.check_schema(source)
    jsonschema.validate(payload, source)
    assert trust_root.is_dir()


def test_profile_resolves_files_and_has_formatting_independent_digest(
    tmp_path: Path,
) -> None:
    trust_root, payload = _material(tmp_path)
    pretty = trust_root / "pretty.json"
    compact = trust_root / "compact.json"
    pretty.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    compact.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    loaded = load_trust_inputs(pretty)
    compact_loaded = load_trust_inputs(compact)

    assert loaded.policy_path == trust_root / "policy.json"
    assert loaded.policy_bytes == b"{}\n"
    assert loaded.verifier_signing_key_path == trust_root / "verifier.pem"
    assert (
        loaded.verifier_signing_key_bytes == (trust_root / "verifier.pem").read_bytes()
    )
    assert loaded.expected_artifact_digests == {
        "baseline": _digest("a"),
        "subject": _digest("b"),
    }
    assert loaded.expected_runtime_digests == {
        "baseline": _digest("d"),
        "subject": _digest("e"),
    }
    assert loaded.expected_schedule_digest == _digest("c")
    assert loaded.expected_signer_fingerprint == _digest("f")
    assert loaded.expected_request_digest is None
    assert loaded.verifier_identity == "invarlock-verifier/release"
    assert loaded.allow_installed_scorers is False
    assert loaded.profile_digest == compact_loaded.profile_digest
    assert loaded.profile_digest.startswith("sha256:")


def test_profile_loads_optional_independent_request_anchor(tmp_path: Path) -> None:
    trust_root, payload = _material(tmp_path)
    anchors = cast(dict[str, object], payload["anchors"])
    anchors["request_digest"] = _digest("1")
    profile = trust_root / "trust.json"
    profile.write_text(json.dumps(payload), encoding="utf-8")

    assert load_trust_inputs(profile).expected_request_digest == _digest("1")


def test_profile_can_replay_with_an_authenticated_public_key_override(
    tmp_path: Path,
) -> None:
    trust_root, payload = _material(tmp_path)
    profile = trust_root / "trust.json"
    profile.write_text(json.dumps(payload), encoding="utf-8")
    original = load_trust_inputs(profile)
    private_key = serialization.load_pem_private_key(
        original.verifier_signing_key_bytes,
        password=None,
    )
    assert isinstance(private_key, ed25519.Ed25519PrivateKey)
    public_key_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    original.verifier_signing_key_path.unlink()

    replay = load_trust_inputs(
        profile,
        verifier_key_bytes_override=public_key_bytes,
    )

    assert replay.profile_digest == original.profile_digest
    assert replay.verifier_signing_key_bytes == public_key_bytes
    with pytest.raises(TrustInputsError, match="exact bytes"):
        load_trust_inputs(
            profile,
            verifier_key_bytes_override="not-bytes",  # type: ignore[arg-type]
        )
    with pytest.raises(TrustInputsError, match="65536-byte"):
        load_trust_inputs(
            profile,
            verifier_key_bytes_override=b"x" * (64 * 1024 + 1),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(extra=True), "schema failed"),
        (
            lambda value: value["policy"].update(path="../policy.json"),
            "schema failed",
        ),
        (
            lambda value: value["anchors"].update(schedule_digest="sha256:bad"),
            "schema failed",
        ),
        (
            lambda value: value["anchors"].update(request_digest="sha256:bad"),
            "schema failed",
        ),
    ],
)
def test_profile_rejects_unknown_traversal_and_invalid_anchors(
    tmp_path: Path, mutation: object, message: str
) -> None:
    trust_root, payload = _material(tmp_path)
    assert callable(mutation)
    mutation(payload)
    profile = trust_root / "trust.json"
    profile.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrustInputsError, match=message):
        load_trust_inputs(profile)


def test_profile_rejects_duplicate_json_symlinks_and_missing_files(
    tmp_path: Path,
) -> None:
    trust_root, payload = _material(tmp_path)
    duplicate = trust_root / "duplicate.json"
    duplicate.write_text(
        '{"format":"invarlock/trust-inputs-v1","format":"invarlock/trust-inputs-v1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(TrustInputsError, match="duplicate"):
        load_trust_inputs(duplicate)

    profile = trust_root / "trust.json"
    profile.write_text(json.dumps(payload), encoding="utf-8")
    profile_link = trust_root / "trust-link.json"
    profile_link.symlink_to(profile)
    with pytest.raises(TrustInputsError, match="symlink"):
        load_trust_inputs(profile_link)

    policy = trust_root / "policy.json"
    policy.unlink()
    with pytest.raises(TrustInputsError, match="policy could not be opened"):
        load_trust_inputs(profile)


def test_profile_rejects_symlinked_parent_and_intermediate_references(
    tmp_path: Path,
) -> None:
    trust_root, payload = _material(tmp_path)
    profile = trust_root / "trust.json"
    profile.write_text(json.dumps(payload), encoding="utf-8")
    linked_parent = tmp_path / "linked-trust"
    linked_parent.symlink_to(trust_root, target_is_directory=True)

    with pytest.raises(TrustInputsError, match="non-symlink directory"):
        load_trust_inputs(linked_parent / profile.name)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "policy.json").write_text("{}\n", encoding="utf-8")
    (outside / "verifier.pem").write_bytes((trust_root / "verifier.pem").read_bytes())
    intermediate = trust_root / "material"
    intermediate.symlink_to(outside, target_is_directory=True)

    policy_link_payload = json.loads(json.dumps(payload))
    policy_link_payload["policy"]["path"] = "material/policy.json"
    profile.write_text(json.dumps(policy_link_payload), encoding="utf-8")
    with pytest.raises(TrustInputsError, match="policy could not be opened"):
        load_trust_inputs(profile)

    key_link_payload = json.loads(json.dumps(payload))
    key_link_payload["verifier"]["signing_key_path"] = "material/verifier.pem"
    profile.write_text(json.dumps(key_link_payload), encoding="utf-8")
    with pytest.raises(TrustInputsError, match="verifier signing key could not"):
        load_trust_inputs(profile)


def test_profile_allows_nested_real_directories_and_freezes_anchor_mappings(
    tmp_path: Path,
) -> None:
    trust_root, payload = _material(tmp_path)
    nested = trust_root / "nested" / "material"
    nested.mkdir(parents=True)
    (trust_root / "policy.json").replace(nested / "policy.json")
    (trust_root / "verifier.pem").replace(nested / "verifier.pem")
    payload["policy"]["path"] = "nested/material/policy.json"
    payload["verifier"]["signing_key_path"] = "nested/material/verifier.pem"
    profile = trust_root / "trust.json"
    profile.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_trust_inputs(profile)
    digest = loaded.profile_digest
    artifact_anchors = cast(Any, loaded.expected_artifact_digests)
    runtime_anchors = cast(Any, loaded.expected_runtime_digests)

    with pytest.raises(TypeError):
        artifact_anchors["baseline"] = _digest("9")
    with pytest.raises(TypeError):
        runtime_anchors["subject"] = _digest("8")
    assert loaded.policy_path == nested / "policy.json"
    assert loaded.verifier_signing_key_path == nested / "verifier.pem"
    assert loaded.expected_artifact_digests["baseline"] == _digest("a")
    assert loaded.expected_runtime_digests["subject"] == _digest("e")
    assert loaded.profile_digest == digest


def test_secure_descriptor_capabilities_and_path_validation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as context:
        context.delattr(trust_inputs_module.os, "O_DIRECTORY")
        with pytest.raises(TrustInputsError, match="loading is unavailable"):
            trust_inputs_module._directory_open_flags()
    with monkeypatch.context() as context:
        context.delattr(trust_inputs_module.os, "O_NOFOLLOW")
        with pytest.raises(TrustInputsError, match="loading is unavailable"):
            trust_inputs_module._file_open_flags()

    with pytest.raises(TrustInputsError, match="parent path is invalid"):
        trust_inputs_module._open_directory_without_links(
            Path("relative"), label="profile"
        )
    with pytest.raises(TrustInputsError, match="path is invalid"):
        trust_inputs_module._safe_relative_parts(7, label="policy")
    for unsafe in ("/absolute.json", "../policy.json", "a//policy.json"):
        with pytest.raises(TrustInputsError, match="path is unsafe"):
            trust_inputs_module._safe_relative_parts(unsafe, label="policy")


def test_secure_directory_open_rejects_unavailable_root_and_non_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as context:
        context.setattr(
            trust_inputs_module.os,
            "open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError()),
        )
        with pytest.raises(TrustInputsError, match="non-symlink directory"):
            trust_inputs_module._open_directory_without_links(
                Path("/trust"), label="profile"
            )

    with monkeypatch.context() as context:
        context.setattr(
            trust_inputs_module.os,
            "fstat",
            lambda _descriptor: SimpleNamespace(st_mode=stat.S_IFREG),
        )
        with pytest.raises(TrustInputsError, match="non-symlink directory"):
            trust_inputs_module._open_directory_without_links(
                Path("/"), label="profile"
            )


def test_descriptor_reader_rejects_wrong_type_static_and_dynamic_oversize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    too_large = tmp_path / "too-large"
    too_large.write_bytes(b"xx")
    root_fd = os.open(tmp_path, trust_inputs_module._directory_open_flags())
    try:
        with pytest.raises(TrustInputsError, match="real regular file"):
            trust_inputs_module._read_relative_regular_file(
                root_fd, (nested.name,), label="policy", max_bytes=4
            )
        with pytest.raises(TrustInputsError, match="size limit"):
            trust_inputs_module._read_relative_regular_file(
                root_fd, (too_large.name,), label="policy", max_bytes=1
            )

        real_fstat = os.fstat

        def hide_initial_size(descriptor: int) -> object:
            observed = real_fstat(descriptor)
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_size=0,
                st_mtime_ns=observed.st_mtime_ns,
                st_ctime_ns=observed.st_ctime_ns,
            )

        with monkeypatch.context() as context:
            context.setattr(trust_inputs_module.os, "fstat", hide_initial_size)
            with pytest.raises(TrustInputsError, match="size limit"):
                trust_inputs_module._read_relative_regular_file(
                    root_fd, (too_large.name,), label="policy", max_bytes=1
                )
    finally:
        os.close(root_fd)


def test_descriptor_reader_detects_change_and_handles_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stable = tmp_path / "stable"
    stable.write_bytes(b"x")
    empty = tmp_path / "empty"
    empty.touch()
    root_fd = os.open(tmp_path, trust_inputs_module._directory_open_flags())
    try:
        assert (
            trust_inputs_module._read_relative_regular_file(
                root_fd, (empty.name,), label="policy", max_bytes=1
            )
            == b""
        )
        real_fstat = os.fstat
        calls = 0

        def report_changed_identity(descriptor: int) -> object:
            nonlocal calls
            calls += 1
            observed = real_fstat(descriptor)
            if calls == 1:
                return observed
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_size=observed.st_size,
                st_mtime_ns=observed.st_mtime_ns,
                st_ctime_ns=observed.st_ctime_ns + 1,
            )

        with monkeypatch.context() as context:
            context.setattr(trust_inputs_module.os, "fstat", report_changed_identity)
            with pytest.raises(TrustInputsError, match="changed while being read"):
                trust_inputs_module._read_relative_regular_file(
                    root_fd, (stable.name,), label="policy", max_bytes=1
                )
    finally:
        os.close(root_fd)


def test_profile_rejects_non_object_json(tmp_path: Path) -> None:
    profile = tmp_path / "trust.json"
    profile.write_text("[]\n", encoding="utf-8")

    with pytest.raises(TrustInputsError, match="must decode to a JSON object"):
        load_trust_inputs(profile)


def test_profile_load_median_is_below_fifty_milliseconds(tmp_path: Path) -> None:
    trust_root, payload = _material(tmp_path)
    profile = trust_root / "trust.json"
    profile.write_text(json.dumps(payload), encoding="utf-8")
    load_trust_inputs(profile)

    durations: list[float] = []
    for _ in range(21):
        started = time.perf_counter()
        load_trust_inputs(profile)
        durations.append(time.perf_counter() - started)

    assert statistics.median(durations) < 0.05
