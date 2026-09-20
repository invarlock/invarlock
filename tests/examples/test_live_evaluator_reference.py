"""The real sentinel replays retained rejection and rejects altered archives."""

from __future__ import annotations

import copy
import importlib.util
import json
import stat
import sys
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/integrations/evaluator-live/references/mistral-7b-sentinel"
SPEC = importlib.util.spec_from_file_location(
    "live_evaluator_reference", REFERENCE / "replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


@pytest.fixture(scope="module")
def files():
    return REPLAY.read_reference(REFERENCE)


def test_complete_real_reference_replays_without_model_acceptance():
    result = REPLAY.replay(REFERENCE)
    assert result["ok"] and result["evaluator_count"] == 19
    assert result["signed_packs_replayed"] == 76
    assert result["new_model_or_judge_calls"] == 0
    assert not result["historical_policy_acceptance"]
    assert len({row["id"] for row in result["results"]}) == 76
    for row in result["results"]:
        assert row["integrity_ok"] and row["receipt_authenticated"]
        assert (row["decision"], row["policy_verdict"], row["verification_status"]) == (
            "regression",
            "fail",
            7,
        )


def test_catalog_sizes_pins_and_public_member_inventory(files):
    catalog = json.loads((REFERENCE / "reference.json").read_bytes())
    assert len(catalog["evaluators"]) == 19 and catalog["signed_packs"] == 76
    assert sum(row["members"] for row in catalog["archives"]) == len(files) == 5181
    for row in catalog["archives"]:
        raw = (REFERENCE / row["path"]).read_bytes()
        assert len(raw) == row["bytes"] < REPLAY.ARCHIVE_LIMIT
        assert REPLAY.sha(raw) == row["sha256"] == REPLAY.ARCHIVES[row["path"]]
        with zipfile.ZipFile(REFERENCE / row["path"]) as archive:
            assert (
                sum(info.file_size for info in archive.infolist())
                == row["expanded_bytes"]
            )
    for name, raw in files.items():
        path = Path(name)
        assert path.suffix not in {".pem", ".pickle", ".pyc", ".gguf", ".safetensors"}
        assert "admission" not in path.name and ".DS_Store" not in path.parts
        assert b"PRIVATE KEY-----" not in raw
    reference = json.loads(files["reference.json"])
    assert reference["format"] == "invarlock/evaluator-live-reference-v1"
    assert len(reference["captures"]) == 19 and len(reference["packs"]) == 76
    assert any("failure.json" in name for name in files)
    assert any(name.startswith("source-history/initial/") for name in files)
    assert any("recovery/harness/" in name for name in files)


def test_generated_temporary_root_resolves_system_alias(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    original = REPLAY.tempfile.TemporaryDirectory

    @contextmanager
    def through_alias(**kwargs):
        with original(dir=real, **kwargs) as directory:
            yield str(alias / Path(directory).name)

    monkeypatch.setattr(REPLAY.tempfile, "TemporaryDirectory", through_alias)
    assert REPLAY.replay(REFERENCE)["signed_packs_replayed"] == 76


def test_changed_archive_rejected_before_zip_inspection(tmp_path, monkeypatch):
    changed = tmp_path / "changed.zip"
    changed.write_bytes((REFERENCE / "exact-match.zip").read_bytes() + b"changed")
    monkeypatch.setattr(
        REPLAY.zipfile, "ZipFile", lambda *a, **k: pytest.fail("untrusted ZIP opened")
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        REPLAY.read_archive(
            changed,
            expected_sha256=REPLAY.ARCHIVES["exact-match.zip"],
            maximum_bytes=REPLAY.ARCHIVE_LIMIT,
            maximum_expanded_bytes=REPLAY.EXPANDED_LIMIT,
        )


@pytest.mark.parametrize(
    "kind",
    [
        "traversal",
        "absolute",
        "drive",
        "drive-relative",
        "backslash",
        "duplicate",
        "symlink",
        "expanded",
        "members",
    ],
)
def test_unsafe_authenticated_zip_members_are_rejected(tmp_path, kind):
    archive = tmp_path / "unsafe.zip"
    names = {
        "traversal": "../escape",
        "absolute": "/escape",
        "drive": "C:/escape",
        "drive-relative": "C:escape",
        "backslash": "one\\escape",
    }
    info = zipfile.ZipInfo(names.get(kind, "one"))
    info.create_system = 3
    info.external_attr = (
        stat.S_IFLNK if kind == "symlink" else stat.S_IFREG | 0o644
    ) << 16
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(info, b"xx")
        if kind in {"duplicate", "members"}:
            with pytest.warns(UserWarning) if kind == "duplicate" else nullcontext():
                second = zipfile.ZipInfo("one" if kind == "duplicate" else "two")
                second.external_attr = (stat.S_IFREG | 0o644) << 16
                stream.writestr(second, b"x")
    with pytest.raises(ValueError, match="unsafe|duplicate|limit"):
        REPLAY.read_archive(
            archive,
            expected_sha256=REPLAY.sha(archive.read_bytes()),
            maximum_bytes=1024,
            maximum_expanded_bytes=1 if kind == "expanded" else 100,
            maximum_members=1 if kind == "members" else 10,
        )


@pytest.mark.parametrize("fault", ["pin", "limit", "symlink", "size"])
def test_archive_source_and_bounds_fail_closed(tmp_path, fault):
    path = tmp_path / "archive.zip"
    if fault == "symlink":
        path.symlink_to(REFERENCE / "exact-match.zip")
    else:
        path.write_bytes(b"x" * 10)
    with pytest.raises(ValueError):
        REPLAY.read_archive(
            path,
            expected_sha256="bad" if fault == "pin" else "0" * 64,
            maximum_bytes=0 if fault == "limit" else 1 if fault == "size" else 100,
            maximum_expanded_bytes=100,
        )


@pytest.mark.parametrize("fault", ["overlap", "inventory", "hash"])
def test_archive_member_manifest_is_independently_checked(files, monkeypatch, fault):
    changed = dict(files)
    if fault == "inventory":
        changed["unexpected"] = b"x"
    elif fault == "hash":
        changed["reference.json"] += b" "
    calls = iter([changed, changed if fault == "overlap" else {}, {}])
    monkeypatch.setattr(REPLAY, "read_archive", lambda *a, **k: next(calls))
    with pytest.raises(ValueError, match="overlapping|inventory|hash"):
        REPLAY.read_reference(REFERENCE)


@pytest.mark.parametrize(
    "fault", ["evidence", "receipt", "trust", "expectation", "status"]
)
def test_changed_signed_pack_or_recipient_expectation_refuses_replay(
    tmp_path, files, fault
):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    row = copy.deepcopy(json.loads(files["reference.json"])["packs"][0])
    for name, raw in files.items():
        if name.startswith(row["directory"] + "/"):
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    directory = tmp_path / row["directory"]
    if fault == "evidence":
        path = directory / "evidence/records/subject.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif fault == "receipt":
        path = directory / "verification.receipt.json"
        value = json.loads(path.read_bytes())
        value["signature"]["value"] = "A" * 86 + "=="
        path.write_text(json.dumps(value))
    elif fault == "trust":
        path = directory / "trust.json"
        value = json.loads(path.read_bytes())
        value["verifier"]["identity"] = "different-recipient"
        path.write_text(json.dumps(value))
    elif fault == "expectation":
        row["expected"]["decision"] = "pass"
    else:
        row["expected_status"] = 0
    key = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    receipts = tmp_path / "new-receipts"
    receipts.mkdir()
    with pytest.raises(ValueError):
        REPLAY.verify_pack(tmp_path, row, receipts, key)


def test_command_requires_installed_recipient_and_installs_network_guard(
    monkeypatch, capsys
):
    import invarlock

    monkeypatch.setattr(invarlock, "__file__", str(ROOT / "src/invarlock/__init__.py"))
    with pytest.raises(ValueError, match="installed recipient"):
        REPLAY.require_installed()
    monkeypatch.setattr(
        invarlock,
        "__file__",
        str(Path(REPLAY.get_path("purelib")) / "invarlock/__init__.py"),
    )
    REPLAY.require_installed()
    guards = []
    monkeypatch.setattr(REPLAY.sys, "addaudithook", guards.append)
    monkeypatch.setattr(sys, "argv", ["replay.py"])
    monkeypatch.setattr(REPLAY, "replay", lambda directory: {"ok": True})
    assert REPLAY.main() == 0 and json.loads(capsys.readouterr().out)["ok"]
    assert guards == [REPLAY.block_network]
    REPLAY.block_network("open", ())
    with pytest.raises(RuntimeError, match="forbids outbound network"):
        REPLAY.block_network("socket.connect", ())
