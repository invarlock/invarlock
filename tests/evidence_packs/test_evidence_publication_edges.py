from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from invarlock import evidence_pack_publication as publication
from invarlock.evidence_pack_contract import EvidencePackError, InputIdentity
from tests.evidence_packs.test_evidence_pack import _digest, _publish


def test_low_level_writer_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    publication._write_new(path, b"first")

    assert path.read_bytes() == b"first"
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(EvidencePackError, match="could not write"):
        publication._write_new(path, b"second")
    assert path.read_bytes() == b"first"


def test_signing_key_loader_rejects_malformed_and_non_ed25519_keys(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.pem"
    malformed.write_text("not a key", encoding="utf-8")
    with pytest.raises(EvidencePackError, match="could not load signing key"):
        publication._load_private_key(malformed)

    key = ec.generate_private_key(ec.SECP256R1())
    non_ed = tmp_path / "ec.pem"
    non_ed.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    with pytest.raises(EvidencePackError, match="must be Ed25519"):
        publication._load_private_key(non_ed)


def test_publish_destination_is_no_clobber(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    destination = tmp_path / "evidence"
    destination.mkdir()
    with pytest.raises(EvidencePackError, match="already exists"):
        publication._publish_directory_no_clobber(staging, destination)
    assert staging.is_dir()


def test_publish_maps_atomic_no_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    destination = tmp_path / "evidence"

    def denied(_source: Path, _destination: Path) -> None:
        raise publication.AtomicDirectoryPublicationError("denied")

    monkeypatch.setattr(publication, "publish_directory_no_replace", denied)
    with pytest.raises(EvidencePackError, match="atomically publish"):
        publication._publish_directory_no_clobber(staging, destination)


def test_publish_maps_atomic_destination_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    destination = tmp_path / "evidence"

    def raced(_source: Path, _destination: Path) -> None:
        raise publication.AtomicDirectoryExistsError("raced")

    monkeypatch.setattr(publication, "publish_directory_no_replace", raced)
    with pytest.raises(EvidencePackError, match="already exists"):
        publication._publish_directory_no_clobber(staging, destination)
    assert staging.is_dir()


def test_publication_rejects_symlink_parent_and_invalid_external_digests(
    tmp_path: Path,
) -> None:
    _pack, _policy, _fingerprint, _runtimes, _key, arguments = _publish(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(EvidencePackError, match="parent must not be a symlink"):
        publication.publish_comparison_evidence(linked_parent / "evidence", **arguments)

    bad_dataset = dict(arguments)
    bad_dataset["dataset"] = InputIdentity(
        _digest("e"), locator="schedule/runtime-behavioral-schedule.json"
    )
    with pytest.raises(EvidencePackError, match="canonical evaluation schedule"):
        publication.publish_comparison_evidence(tmp_path / "bad-dataset", **bad_dataset)

    bad_policy = dict(arguments)
    bad_policy["policy"] = InputIdentity(_digest("e"), locator="inputs/policy.json")
    with pytest.raises(EvidencePackError, match="policy identity"):
        publication.publish_comparison_evidence(tmp_path / "bad-policy", **bad_policy)


def test_publication_rejects_request_binding_drift(tmp_path: Path) -> None:
    _pack, _policy, _fingerprint, _runtimes, _key, arguments = _publish(tmp_path)

    mutations = [
        (
            lambda request: request["comparison"]["dataset"].update(
                source_sha256="f" * 64
            ),
            "does not match the canonical schedule identity",
        ),
        (
            lambda request: request["comparison"].update(policy="other"),
            "canonical policy identity",
        ),
        (
            lambda request: request["comparison"]["baseline"].update(artifact=[]),
            "not of type 'object'",
        ),
        (
            lambda request: request["comparison"]["baseline"]["artifact"].update(
                model_id="other.gguf"
            ),
            "model_id does not match",
        ),
    ]
    import copy

    for index, (mutate, message) in enumerate(mutations):
        candidate = dict(arguments)
        request = copy.deepcopy(arguments["normalized_request"])
        mutate(request)
        candidate["normalized_request"] = request
        with pytest.raises(EvidencePackError, match=message):
            publication.publish_comparison_evidence(
                tmp_path / f"binding-{index}", **candidate
            )

    locator_drift = dict(arguments)
    locator_drift["baseline"] = InputIdentity(
        arguments["baseline"].digest,  # type: ignore[union-attr]
        locator="artifact://other.gguf",
    )
    with pytest.raises(EvidencePackError, match="locator does not match"):
        publication.publish_comparison_evidence(
            tmp_path / "locator-drift", **locator_drift
        )


def test_checksum_ledger_is_deterministic() -> None:
    assert (
        publication._checksum_bytes({"b": b"two", "a": b"one"})
        == (
            f"{hashlib.sha256(b'one').hexdigest()}  a\n"
            f"{hashlib.sha256(b'two').hexdigest()}  b\n"
        ).encode()
    )
