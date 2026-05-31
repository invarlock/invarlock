from __future__ import annotations

from pathlib import Path

from tests.reporting._support_evidence_pack_paths import (
    _write_json,
    evidence_pack_mod,
)


def test_load_trust_store_fingerprints_error_and_payload_shapes(tmp_path: Path) -> None:
    missing = tmp_path / "missing-trust.json"
    fingerprints, errors, path = evidence_pack_mod.load_trust_store_fingerprints(
        missing
    )
    assert fingerprints == set()
    assert "not found" in errors[0]
    assert path == str(missing)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")
    fingerprints, errors, _ = evidence_pack_mod.load_trust_store_fingerprints(invalid)
    assert fingerprints == set()
    assert "not valid JSON" in errors[0]

    empty_list = tmp_path / "empty-list.json"
    _write_json(empty_list, [])
    fingerprints, errors, _ = evidence_pack_mod.load_trust_store_fingerprints(
        empty_list
    )
    assert fingerprints == set()
    assert errors == ["Evidence-pack trust store contains no trusted signers."]

    bad_dict = tmp_path / "bad-dict.json"
    _write_json(bad_dict, {"trusted_signers": "not-a-list"})
    fingerprints, errors, _ = evidence_pack_mod.load_trust_store_fingerprints(bad_dict)
    assert fingerprints == set()
    assert errors == ["Evidence-pack trust store trusted_signers must be a list."]

    scalar_payload = tmp_path / "scalar.json"
    _write_json(scalar_payload, "not-a-store")
    fingerprints, errors, _ = evidence_pack_mod.load_trust_store_fingerprints(
        scalar_payload
    )
    assert fingerprints == set()
    assert errors == ["Evidence-pack trust store must be a JSON object or list."]

    mixed = tmp_path / "mixed.json"
    valid = "sha256:" + ("a" * 64)
    _write_json(mixed, [{"fingerprint": valid.upper()}, 7, "bad"])
    fingerprints, errors, _ = evidence_pack_mod.load_trust_store_fingerprints(mixed)
    assert fingerprints == {valid}
    assert "entry 1 is not a string" in errors[0]
    assert "entry 2 is not a sha256 fingerprint" in errors[1]
