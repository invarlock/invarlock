"""Checked SDK serialization repair over synthetic frozen model ledgers."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"
sys.path.insert(0, str(HERE))
import bindings  # noqa: E402
import common  # noqa: E402
import recipient  # noqa: E402
import recover_harness  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "harness_recovery_fixture", Path(__file__).with_name("test_live_recipient.py")
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


def captured(tmp_path):
    protocol, args = FIXTURE.fixture(tmp_path, "lm-evaluation-harness")
    protocol["cases"][1]["metadata"]["only_second"] = "retained"
    args[0].write_bytes(common.encoded(protocol))
    args[1] = common.digest(protocol)
    for role in recipient.ROLES:
        directory = tmp_path / role
        role_protocol = {**protocol, "role": role}
        (directory / "protocol.json").write_bytes(common.encoded(role_protocol))
        native = common.read(directory / "native.json")
        for path in list((directory / "tasks").glob("*.response.json")):
            response = common.read(path)
            case = next(
                row
                for row in protocol["cases"]
                if row["id"] == response["request"]["case_id"]
            )
            old_stem = path.name.removesuffix(".response.json")
            response["request"]["protocol_digest"] = common.digest(role_protocol)
            execution = response["result"]["metadata"]["invarlock_model_execution"]
            execution["request"] = copy.deepcopy(response["request"])
            execution["protocol_digest"] = common.digest(role_protocol)
            path.unlink()
            (directory / "tasks" / (old_stem + ".request.json")).unlink()
            stem = common.digest(response["request"]).removeprefix("sha256:")
            common.write(
                directory / "tasks" / (stem + ".request.json"), response["request"]
            )
            common.write(directory / "tasks" / (stem + ".response.json"), response)
            sample = next(row for row in native if row["doc"]["id"] == case["id"])
            sample["doc"] = {
                **copy.deepcopy(case),
                "metadata": {
                    "family": "synthetic-test",
                    "only_second": case["metadata"].get("only_second"),
                },
            }
            sample["metadata"] = {
                **case["metadata"],
                **bindings.bind_result(
                    response["result"], case, "lm-evaluation-harness", "0.4.12"
                )["metadata"],
                "invarlock_id": case["id"],
            }
        (directory / "native.json").write_bytes(common.encoded(native))
        manifest = common.read(directory / "capture.json")
        manifest["protocol_digest"] = common.digest(role_protocol)
        manifest["native_sha256"] = (
            "sha256:"
            + hashlib.sha256((directory / "native.json").read_bytes()).hexdigest()
        )
        (directory / "capture.json").write_bytes(common.encoded(manifest))
        (directory / "sdk").mkdir()
        (directory / "sdk" / "original.jsonl").write_bytes(b'{"native_sdk_log":true}\n')
    return protocol, args


def rewrite(directory, native):
    path = directory / "native.json"
    path.write_bytes(common.encoded(native))
    manifest = common.read(directory / "capture.json")
    manifest["native_sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (directory / "capture.json").write_bytes(common.encoded(manifest))


@pytest.mark.parametrize("route", ["native-json", "envelope"])
def test_recovery_preserves_original_archive_and_enables_checked_import(
    tmp_path, route
):
    protocol, args = captured(tmp_path)
    with pytest.raises(ValueError, match="input_digest"):
        recipient.prepare(*args, route, tmp_path / "original-rejected")
    corrected = []
    for role in recipient.ROLES:
        source, output = tmp_path / role, tmp_path / (role + "-derived")
        before = {
            path.relative_to(source): path.read_bytes()
            for path in source.rglob("*")
            if path.is_file()
        }
        declaration = recover_harness.recover(args[0], args[1], source, role, output)
        assert (
            declaration["model_calls"] == 0
            and declaration["status"] == "derived_capture"
        )
        for name, raw in before.items():
            assert (source / name).read_bytes() == raw
            assert (output / "original" / name).read_bytes() == raw
            if name.parts[0] == "tasks":
                assert (output / name).read_bytes() == raw
        rows = common.read(output / "native.json")
        binding = rows[0]["metadata"]["invarlock_serialization_binding"]
        assert binding["nullable_metadata_fields"] == ["only_second"]
        assert binding["original_input"] == protocol["cases"][0]
        assert binding["original_likelihood"]["input_digest"] == common.digest(
            protocol["cases"][0]
        )
        assert rows[0]["metadata"]["invarlock_likelihood"][
            "input_digest"
        ] == common.digest(rows[0]["doc"])
        assert "invarlock_serialization_binding" not in rows[1]["metadata"]
        assert common.read(output / "capture.json")[
            "serialization_recovery"
        ] == common.digest(declaration)
        corrected.append(output)
    _, runs = recipient.prepare(
        args[0], args[1], *corrected, args[4], route, tmp_path / "recipient"
    )
    assert all(len(run["records"]) == 2 for run in runs)
    assert all(row["output"] == "A" for run in runs for row in run["records"])


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_null",
        "non_null",
        "existing_value",
        "bool_number",
        "missing",
        "input",
        "target",
    ],
)
def test_document_binding_rejects_every_change_beyond_declared_padding(mutation):
    case = {
        "id": "a",
        "input": "task",
        "expected": "reference",
        "metadata": {"count": 1},
    }
    cases = [case, {**case, "id": "b", "metadata": {"count": 1, "optional": "source"}}]
    document = {**copy.deepcopy(case), "metadata": {"count": 1, "optional": None}}
    if mutation == "extra_null":
        document["metadata"]["invented"] = None
    elif mutation == "non_null":
        document["metadata"]["optional"] = "invented"
    elif mutation == "existing_value":
        document["metadata"]["count"] = 2
    elif mutation == "bool_number":
        document["metadata"]["count"] = True
    elif mutation == "missing":
        document["metadata"].pop("count")
    elif mutation == "input":
        document["input"] = "changed"
    else:
        document["expected"] = "changed"
    with pytest.raises(ValueError, match="nullable metadata"):
        bindings.rebind_harness_metadata({}, case, cases, document)


@pytest.mark.parametrize(
    "fault",
    ["digest", "duplicate", "missing", "output", "binding", "symlink", "existing"],
)
def test_recovery_fails_closed_without_rewriting_source(tmp_path, fault):
    _, args = captured(tmp_path)
    source, output = tmp_path / "baseline", tmp_path / "derived"
    native = common.read(source / "native.json")
    if fault == "digest":
        args[1] = "sha256:" + "0" * 64
    elif fault == "duplicate":
        native[1] = copy.deepcopy(native[0])
    elif fault == "missing":
        native.pop()
    elif fault == "output":
        native[0]["filtered_resps"] = ["changed"]
    elif fault == "binding":
        native[0]["metadata"]["invarlock_likelihood"]["token_count"] += 1
    elif fault == "symlink":
        (source / "sdk" / "link").symlink_to(source / "native.json")
    else:
        output.mkdir()
    rewrite(source, native)
    original = (source / "native.json").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        recover_harness.recover(args[0], args[1], source, "baseline", output)
    assert (source / "native.json").read_bytes() == original
    if fault != "existing":
        assert not output.exists()


@pytest.mark.parametrize("fault", ["input", "reference", "source", "type", "capture"])
def test_rebinding_requires_the_original_likelihood_and_capture_binding(fault):
    case = {"id": "a", "input": "task", "expected": "reference", "metadata": {}}
    cases = [case, {**case, "id": "b", "metadata": {"tag": "second"}}]
    document = {**case, "metadata": {"tag": None}}
    facts = {
        "input_digest": common.digest(case),
        "reference_digest": common.digest(case["expected"]),
        "source": {"name": "lm-evaluation-harness", "version": "0.4.12"},
    }
    metadata = {"invarlock_likelihood": facts}
    if fault == "input":
        facts["input_digest"] = "changed"
    elif fault == "reference":
        facts["reference_digest"] = "changed"
    elif fault == "source":
        facts["source"]["version"] = "other"
    elif fault == "type":
        metadata["invarlock_likelihood"] = []
    else:
        metadata["invarlock_capture_binding"] = {
            "native_input": case,
            "input_projection": None,
        }
    with pytest.raises(ValueError, match="binding"):
        bindings.rebind_harness_metadata(metadata, case, cases, document)


def test_serialization_without_likelihood_preserves_absence_and_refuses_rebinding():
    case = {"id": "a", "input": "task", "expected": "reference", "metadata": {}}
    cases = [case, {**case, "id": "b", "metadata": {"tag": "second"}}]
    document = {**case, "metadata": {"tag": None}}
    original = {"retained": {"value": True}}
    result = bindings.rebind_harness_metadata(original, case, cases, document)
    assert "invarlock_likelihood" not in result
    assert result["invarlock_serialization_binding"]["original_likelihood"] is None
    result["retained"]["value"] = False
    assert original["retained"]["value"] is True
    with pytest.raises(ValueError, match="already exists"):
        bindings.rebind_harness_metadata(result, case, cases, document)
    with pytest.raises(ValueError, match="outside"):
        bindings.rebind_harness_metadata(
            original, {**case, "input": "changed"}, cases, document
        )


def test_recovery_refuses_archive_size_and_validation_races(tmp_path, monkeypatch):
    import invarlock.evaluation_record_contracts.contracts as contracts

    _, args = captured(tmp_path)
    source, output = tmp_path / "baseline", tmp_path / "derived"
    monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", 50000)
    for index in range(2):
        (source / "sdk" / f"large-{index}.log").write_bytes(b"x" * 25000)
    with pytest.raises(ValueError, match="archive exceeds"):
        recover_harness.recover(args[0], args[1], source, "baseline", output)
    assert not output.exists()
    for path in (source / "sdk").glob("large-*.log"):
        path.unlink()
    original_capture = recipient.capture

    def changed_after_validation(*args):
        result = original_capture(*args)
        with (source / "native.json").open("ab") as stream:
            stream.write(b" ")
        return result

    monkeypatch.setattr(recipient, "capture", changed_after_validation)
    with pytest.raises(ValueError, match="changed during"):
        recover_harness.recover(args[0], args[1], source, "baseline", output)
    assert not output.exists()


def test_recovery_cli_checks_recipient_identity_and_emits_derivation(
    tmp_path, monkeypatch, capsys
):
    _, args = captured(tmp_path)
    checked = []
    monkeypatch.setattr(recipient, "installed_identity", lambda: checked.append(True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_harness.py",
            "--protocol",
            str(args[0]),
            "--protocol-sha256",
            args[1],
            "--capture",
            str(tmp_path / "baseline"),
            "--role",
            "baseline",
            "--output",
            str(tmp_path / "derived"),
        ],
    )
    recover_harness.main()
    assert checked == [True]
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "derived_capture" and result["model_calls"] == 0
