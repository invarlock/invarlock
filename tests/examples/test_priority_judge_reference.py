"""Retained real judge sources, receipts and interruption checkpoints stay bound."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "examples/integrations/evaluator-live/references/priority-workflows"
REFERENCES = DIRECTORY.parent
SPEC = importlib.util.spec_from_file_location(
    "priority_judge_reference", DIRECTORY / "judge_replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


@pytest.fixture(scope="module")
def retained():
    return REPLAY.read_reference(DIRECTORY, REFERENCES)


@pytest.fixture(scope="module")
def materialized(tmp_path_factory, retained):
    output = tmp_path_factory.mktemp("priority-judge-members").resolve()
    for name, raw in retained[1].items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return output


def selected(retained, prefix):
    return next(
        pack
        for pack in retained[0]["reference"]["packs"]
        if pack["id"].startswith(prefix)
    )


def test_all_real_judge_packs_replay_with_original_insufficient_decisions(
    tmp_path, retained, monkeypatch
):
    monkeypatch.setattr(REPLAY, "read_reference", lambda *_args: retained)
    report = REPLAY.replay(
        tmp_path / "replay", directory=DIRECTORY, require_installed=False
    )
    assert report["original_receipts_replayed"] == 24
    assert report["counts"] == {
        "attempt_status": {"completed": 1288},
        "decisions": {"insufficient_evidence": 24},
        "lifecycle_entries": 4,
        "packs": 24,
        "planned_trials": 1472,
        "retained_attempts": 1288,
        "unadmitted_trials": 184,
    }
    for result in report["results"]:
        value = result["verification"]
        assert value["authenticated"] and value["verified"] and value["replayed"]
        assert not value["accepted"]
        assert value["decision"] == "insufficient_evidence"


def test_catalog_pins_companion_profiles_and_private_exclusions(retained):
    catalog, members, companions = retained
    assert (
        hashlib.sha256((DIRECTORY / "judge-reference.json").read_bytes()).hexdigest()
        == REPLAY.CATALOG_SHA256
    )
    assert len(catalog["reference"]["packs"]) == 24
    assert len(companions) == 2
    assert sum(pin["members"] for pin in catalog["archives"]) == len(members)
    for pin in catalog["archives"]:
        raw = (DIRECTORY / pin["path"]).read_bytes()
        assert len(raw) == pin["bytes"] < 10 * 1024**2
        assert REPLAY.digest(raw) == pin["sha256"]
    for name, raw in members.items():
        assert not name.endswith((".pem", ".safetensors", ".gguf", ".pickle"))
        assert not Path(name).name.startswith("admission-")
        assert b"-----BEGIN PRIVATE KEY-----" not in raw
    packs = catalog["reference"]["packs"]
    assert sum(pack["group"] == "http-primary" for pack in packs) == 8
    assert sum(bool(pack["lifecycle"]) for pack in packs) == 4
    assert sum(pack["group"] == "budget-controls" for pack in packs) == 4


def test_catalog_byte_tampering_rejected(tmp_path):
    (tmp_path / "judge-reference.json").write_bytes(
        (DIRECTORY / "judge-reference.json").read_bytes() + b" "
    )
    with pytest.raises(ValueError, match="catalog digest"):
        REPLAY.read_reference(tmp_path, REFERENCES)


def test_archive_physical_tampering_rejected(tmp_path, retained):
    (tmp_path / "judge-reference.json").write_bytes(
        (DIRECTORY / "judge-reference.json").read_bytes()
    )
    pin = retained[0]["archives"][0]
    raw = bytearray((DIRECTORY / pin["path"]).read_bytes())
    raw[-1] ^= 1
    (tmp_path / pin["path"]).write_bytes(raw)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        REPLAY.read_reference(tmp_path, REFERENCES)


@pytest.mark.parametrize(
    "change",
    ["format", "path", "overlap", "manifest", "inventory", "member", "map", "link"],
)
def test_archive_and_source_link_validation(tmp_path, retained, monkeypatch, change):
    catalog = copy.deepcopy(retained[0])
    members = dict(retained[1])
    companions = {name: dict(files) for name, files in retained[2].items()}
    if change == "format":
        catalog["format"] = "unknown"
    elif change == "path":
        catalog["archives"][0]["path"] = "../escape.zip"
    elif change == "manifest":
        catalog["manifest_sha256"] = "sha256:" + "0" * 64
    elif change == "inventory":
        members["unexpected.json"] = b"{}"
    elif change == "member":
        name = next(name for name in members if name != "judge-manifest.json")
        members[name] += b" "
    elif change == "map":
        catalog["reference"]["scope"] = "changed"
    elif change == "link":
        link = catalog["reference"]["packs"][0]["capture_links"][0]
        name = next(iter(link["files"]))
        companions[link["companion"]][link["directory"] + "/" + name] += b" "
    raw = json.dumps(catalog).encode()
    (tmp_path / "judge-reference.json").write_bytes(raw)
    monkeypatch.setattr(REPLAY, "CATALOG_SHA256", hashlib.sha256(raw).hexdigest())
    pins = catalog["archives"]
    by_pin = {
        pin["sha256"]: companions[name]
        for name, pin in catalog["reference"]["companions"].items()
    }
    by_pin[pins[0]["sha256"]] = members
    by_pin[pins[1]["sha256"]] = members if change == "overlap" else {}
    monkeypatch.setattr(
        REPLAY, "read_pinned", lambda _reader, _path, pin: by_pin[pin["sha256"]]
    )
    with pytest.raises(ValueError):
        REPLAY.read_reference(tmp_path, REFERENCES)


@pytest.mark.parametrize("key", ["bytes", "expanded_bytes", "members"])
def test_declared_archive_bounds_cannot_expand(key):
    pin = {"bytes": 1, "expanded_bytes": 1, "members": 1}
    pin[key] = 2**40
    with pytest.raises(ValueError, match="bounds"):
        REPLAY.read_pinned(None, Path("unused.zip"), pin)


@pytest.mark.parametrize(
    "change",
    [
        "frozen",
        "plan_bytes",
        "event_source",
        "event_shard",
        "missing_shard",
        "receipt_expectation",
    ],
)
def test_signed_pack_source_and_shard_tampering_rejected(
    materialized, retained, change
):
    pack = copy.deepcopy(selected(retained, "budget-controls-inspect-ai"))
    entry = materialized / pack["directory"]
    if change == "receipt_expectation":
        pack["expected"]["accepted"] = True
        with pytest.raises(ValueError, match="signed receipt"):
            REPLAY.verify_pack(entry, pack)
        return
    if change in {"frozen", "plan_bytes"}:
        path = entry / "frozen/plan.json"
    elif change == "event_source":
        path = entry / "sources" / Path(pack["event_sources"][0]["member"]).name
    else:
        path = next((entry / "checkpoints").glob("result-*.json"))
    raw = path.read_bytes()
    try:
        if change == "missing_shard":
            path.unlink()
        elif change == "frozen":
            value = json.loads(raw)
            value["changed"] = True
            path.write_text(json.dumps(value))
        elif change == "event_shard":
            value = json.loads(raw)
            value["event"]["changed"] = True
            path.write_text(json.dumps(value))
        else:
            path.write_bytes(raw + b" ")
        with pytest.raises(ValueError):
            REPLAY.verify_pack(entry, pack)
    finally:
        path.write_bytes(raw)


@pytest.mark.parametrize(
    "change",
    [
        "snapshot",
        "count",
        "not_resumable",
        "checkpoint_replaced",
        "checkpoint_bytes",
        "attempts",
    ],
)
def test_original_two_call_pause_and_resume_cannot_be_rewritten(
    materialized, retained, change
):
    pack = selected(retained, "local-primary-inspect-ai-native-json")
    entry = materialized / pack["directory"]
    measured = json.loads((entry / "evidence/measurements.json").read_bytes())
    path = entry / "lifecycle/lifecycle-resume-observation.json"
    if change == "not_resumable":
        path = entry / "lifecycle/lifecycle-stop-observation.json"
    elif change == "checkpoint_bytes":
        member = pack["lifecycle"]["checked_initial_members"][0]
        path = (
            entry / "lifecycle/collection.json"
            if member.endswith("/collection.json")
            else entry / "checkpoints" / Path(member).name
        )
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
        if change == "snapshot":
            value["admission_sha256"] = "sha256:" + "0" * 64
        elif change == "count":
            value["after"]["results"] -= 1
        elif change == "not_resumable":
            value["result"]["resumable"] = False
        elif change == "checkpoint_replaced":
            name = next(iter(value["before"]["files"]))
            value["after"]["files"][name] = {"bytes": 0, "sha256": "sha256:" + "0" * 64}
        elif change == "attempts":
            trial = next(t for t in measured["trials"] if t["attempts"])
            trial["attempts"][0]["attempt"] = 99
        if change == "checkpoint_bytes":
            path.write_bytes(raw + b" ")
        elif change != "attempts":
            path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            REPLAY.check_lifecycle(entry, pack, measured)
    finally:
        path.write_bytes(raw)


def test_original_sdk_capture_must_reproduce_canonical_frozen_judge_run(
    materialized, retained
):
    pack = selected(retained, "budget-controls-inspect-ai")
    path = materialized / pack["directory"] / "frozen/baseline_run.json"
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
        value["run_id"] = "different-run"
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="does not reproduce frozen judge run"):
            REPLAY.verify_capture_bindings(
                materialized, {"packs": [pack]}, retained[2], REFERENCES
            )
    finally:
        path.write_bytes(raw)


@pytest.mark.parametrize("change", ["identity", "counts", "lifecycle", "decisions"])
def test_replay_orchestration_refuses_changed_campaign_claims(
    tmp_path, retained, monkeypatch, change
):
    catalog = copy.deepcopy(retained[0])
    reference = catalog["reference"]
    if change == "identity":
        reference["packs"][0]["directory"] = "different"
    elif change == "counts":
        reference["counts"]["retained_attempts"] = 0
    elif change == "lifecycle":
        reference["counts"]["lifecycle_entries"] = 0
    else:
        reference["counts"]["decisions"] = {"pass": 24}
    monkeypatch.setattr(REPLAY, "read_reference", lambda *_args: (catalog, {}, {}))
    monkeypatch.setattr(REPLAY, "verify_capture_bindings", lambda *_args: None)
    monkeypatch.setattr(
        REPLAY,
        "verify_pack",
        lambda _entry, pack: {"verification": {"decision": "insufficient_evidence"}},
    )
    with pytest.raises(ValueError):
        REPLAY.replay(tmp_path / "replay", directory=DIRECTORY, require_installed=False)


def test_main_enforces_offline_installed_replay(tmp_path, monkeypatch, capsys):
    import invarlock.security

    seen = []
    monkeypatch.setattr(
        invarlock.security, "enforce_network_policy", lambda value: seen.append(value)
    )
    monkeypatch.setattr(
        REPLAY,
        "replay",
        lambda output, **kwargs: {"counts": {"packs": 24}, "results": []},
    )
    REPLAY.main(["--output", str(tmp_path / "output")])
    assert seen == [False]
    assert json.loads(capsys.readouterr().out) == {"counts": {"packs": 24}}


def test_installed_recipient_guard_runs_before_opening_reference(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def unavailable():
        raise ValueError("recipient installation required")

    monkeypatch.setattr(
        REPLAY, "shared", lambda _root: SimpleNamespace(require_installed=unavailable)
    )
    with pytest.raises(ValueError, match="installation required"):
        REPLAY.replay(tmp_path / "replay", directory=DIRECTORY)
