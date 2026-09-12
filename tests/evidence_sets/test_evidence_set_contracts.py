from pathlib import Path

import pytest

from invarlock.evidence_sets.contracts import EvidenceSetError, load_index


def test_unknown_index_fields_fail(tmp_path: Path):
    (tmp_path / "evidence-set.json").write_text(
        '{"format":"invarlock/evidence-set-v1","members":{},"accepted":true}'
    )
    with pytest.raises(EvidenceSetError, match="invalid"):
        load_index(tmp_path)


def test_index_symlink_fails(tmp_path: Path):
    (tmp_path / "outside.json").write_text("{}")
    (tmp_path / "evidence-set.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises((EvidenceSetError, ValueError)):
        load_index(tmp_path)


@pytest.mark.parametrize(
    "path", ["../outside", "/absolute", "a/../judge", "judge/child"]
)
def test_member_escape_or_overlap_rejected(tmp_path, path):
    import json

    from invarlock.evidence_sets.contracts import load_index
    from tests.evidence_sets.test_verification import fixture, write

    root, _ = fixture(tmp_path)
    value = json.loads((root / "evidence-set.json").read_text())
    value["members"]["deterministic"]["path"] = path
    write(root / "evidence-set.json", value)
    with pytest.raises(ValueError):
        load_index(root)


@pytest.mark.parametrize("filename", ["manifest.json", "envelope.json"])
def test_ambiguous_root_is_not_an_evidence_set(tmp_path, filename):
    from invarlock.evidence_sets.contracts import load_index, write_evidence_set_index
    from tests.evidence_sets.test_verification import fixture

    root, _ = fixture(tmp_path)
    (root / filename).write_text("{}")
    with pytest.raises(ValueError, match="conflicting"):
        load_index(root)
    with pytest.raises(ValueError, match="conflicting"):
        write_evidence_set_index(root, deterministic="deterministic", judge="judge")


@pytest.mark.parametrize("filename", ["evidence-set.json", "manifest.json"])
def test_nested_or_ambiguous_child_rejected(tmp_path, filename):
    from invarlock.evidence_sets.contracts import check_statements, load_index
    from tests.evidence_sets.test_verification import fixture

    root, _ = fixture(tmp_path)
    (root / "judge" / filename).write_text("{}")
    index, _ = load_index(root)
    with pytest.raises(ValueError, match="nested|conflicting"):
        check_statements(root, index)


def test_transport_writer_rejects_overlap_and_nonobject(tmp_path):
    from invarlock.evidence_sets.contracts import read_object, write_evidence_set_index
    from tests.evidence_sets.test_verification import fixture

    root, _ = fixture(tmp_path)
    (root / "judge/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="overlap"):
        write_evidence_set_index(root, deterministic="judge", judge="judge")
    value = tmp_path / "array.json"
    value.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        read_object(value)


@pytest.mark.parametrize(
    "name",
    ["evidence_set", "evidence_set_recipient_policy", "evidence_set_verification"],
)
def test_public_loaders_use_matching_packaged_closed_schemas(name):
    import json

    from jsonschema import Draft202012Validator

    from invarlock import public_contracts

    loader_name = f"load_{name}_schema"
    assert loader_name in public_contracts.__all__
    schema = getattr(public_contracts, loader_name)()
    Draft202012Validator.check_schema(schema)
    assert schema == json.loads(
        (Path(__file__).parents[2] / "contracts" / f"{name}.schema.json").read_text()
    )
    schema["additionalProperties"] = True
    assert getattr(public_contracts, loader_name)()["additionalProperties"] is False
