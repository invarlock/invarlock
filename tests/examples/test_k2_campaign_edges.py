"""Adversarial file, protocol, CLI, and independent-recipient boundaries."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import struct
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from examples.qualification import k2_campaign as campaign
from tests.examples.test_k2_campaign import _capture, _ready_plan, _tensor_file


@pytest.mark.parametrize("data", [b'{"a":1,"a":2}', b'{"a":NaN}'])
def test_ambiguous_json_is_rejected(tmp_path, data):
    path = tmp_path / "data.json"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        campaign.read_json(path)


def test_json_bound_and_no_clobber(tmp_path):
    path = tmp_path / "data.json"
    campaign.write_json(path, {})
    with pytest.raises(FileExistsError):
        campaign.write_json(path, {"changed": True})
    with path.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="64 MiB"):
        campaign.read_json(path)


@pytest.mark.parametrize(
    "data", [b"", struct.pack("<Q", 0), struct.pack("<Q", 20_000_000)]
)
def test_tensor_header_bounds(data):
    with pytest.raises(ValueError, match="header"):
        campaign._tensor_inventory(io.BytesIO(data), len(data))


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "F32"},
        {"shape": [-1]},
        {"shape": [True]},
        {"shape": "two"},
        {"data_offsets": [1, 5]},
        {"data_offsets": [0, 8]},
        {"data_offsets": [False, 4]},
    ],
)
def test_tensor_layout_rejects_unsupported_or_ambiguous_values(change):
    value = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4], **change}
    header = json.dumps({"weight": value, "__metadata__": {"format": "pt"}}).encode()
    data = struct.pack("<Q", len(header)) + header + b"1234"
    with pytest.raises(ValueError, match="tensor"):
        campaign._tensor_inventory(io.BytesIO(data), len(data))


def test_tensor_bytes_must_be_fully_accounted_for():
    header = b"{}"
    data = struct.pack("<Q", len(header)) + header + b"extra"
    with pytest.raises(ValueError, match="uninterpreted"):
        campaign._tensor_inventory(io.BytesIO(data), len(data))
    with pytest.raises(ValueError, match="truncated"):
        campaign._stream_hash(io.BytesIO(b"a"), 2, hashlib.sha256())


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"weight": None},
        {"weight": {"dtype": "BF16", "shape": [1], "data_offsets": None}},
        {"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0]}},
    ],
)
def test_malformed_tensor_descriptors_fail_with_input_error(value):
    header = json.dumps(value).encode()
    data = struct.pack("<Q", len(header)) + header + b"xx"
    with pytest.raises(ValueError, match="tensor"):
        campaign._tensor_inventory(io.BytesIO(data), len(data))


def test_symlink_swap_after_inventory_cannot_escape_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    root.mkdir()
    path = root / "model.safetensors"
    item = _tensor_file(path)
    outside = tmp_path / "outside"
    outside.write_bytes(path.read_bytes())
    path_class = type(path)
    original = path_class.resolve

    def swap(selected, *args, **kwargs):
        if selected == path:
            selected.unlink()
            selected.symlink_to(outside)
        return original(selected, *args, **kwargs)

    monkeypatch.setattr(path_class, "resolve", swap)
    with pytest.raises(ValueError, match="regular"):
        campaign.measure_snapshot(root, [item])


def test_git_blob_metadata_is_rehashed_and_not_executed(tmp_path):
    item = _tensor_file(tmp_path / "model.safetensors")
    code = b"raise RuntimeError('must never execute')\n"
    (tmp_path / "model.py").write_bytes(code)
    extra = {
        "path": "model.py",
        "size_bytes": len(code),
        "sha256": None,
        "git_blob": hashlib.sha1(
            f"blob {len(code)}\0".encode() + code, usedforsecurity=False
        ).hexdigest(),
    }
    assert len(campaign.measure_snapshot(tmp_path, [item, extra])["files"]) == 2
    extra["git_blob"] = "0" * 40
    with pytest.raises(ValueError, match="identity"):
        campaign.measure_snapshot(tmp_path, [item, extra])


def test_duplicate_tensors_and_no_weights_are_rejected(tmp_path):
    a = _tensor_file(tmp_path / "a.safetensors")
    b = _tensor_file(tmp_path / "b.safetensors")
    with pytest.raises(ValueError, match="duplicate tensor"):
        campaign.measure_snapshot(tmp_path, [a, b])
    (tmp_path / "b.safetensors").unlink()
    (tmp_path / "a.safetensors").rename(tmp_path / "data.bin")
    a["path"] = "data.bin"
    with pytest.raises(ValueError, match="no BF16 tensors"):
        campaign.measure_snapshot(tmp_path, [a])


def test_snapshot_mutation_during_hashing_is_rejected(tmp_path, monkeypatch):
    item = _tensor_file(tmp_path / "model.safetensors")
    original = campaign._tensor_inventory

    def mutate(stream, size):
        result = original(stream, size)
        os.utime(tmp_path / item["path"], ns=(1, 1))
        return result

    monkeypatch.setattr(campaign, "_tensor_inventory", mutate)
    with pytest.raises(ValueError, match="changed during"):
        campaign.measure_snapshot(tmp_path, [item])


def test_tensor_shape_change_requires_separate_review():
    plan = _ready_plan()
    plan["model"]["candidate"]["materialized"]["tensors"]["weight"]["shape"] = [2]
    with pytest.raises(ValueError, match="shapes"):
        campaign.require_ready(plan)


@pytest.mark.parametrize(
    "mutation,message",
    [("identity", "unmeasured"), ("budget", "budget"), ("policy", "predeclared")],
)
def test_readiness_cannot_be_obtained_by_omitting_materialization_or_changing_policy(
    mutation, message
):
    plan = _ready_plan()
    if mutation == "identity":
        plan["model"]["baseline"]["materialized"] = None
    elif mutation == "budget":
        plan["budget"]["maximum_wall_seconds"] = True
    else:
        plan["policies"]["classification"]["metrics"][0]["candidate_minimum"] = 0
    with pytest.raises(ValueError, match=message):
        campaign.require_ready(plan)


@pytest.mark.parametrize("latency", [True, -1, float("inf"), "10"])
def test_latency_must_be_a_finite_measurement(latency):
    plan = _ready_plan()
    captured = _capture(plan, "baseline")
    captured["rows"][0]["latency_ms"] = latency
    with pytest.raises(ValueError, match="latency"):
        campaign.project_capture(plan, captured)


def test_native_settings_and_wrong_recipient_key_are_rejected():
    plan = _ready_plan()
    a, b = _capture(plan, "baseline"), _capture(plan, "candidate")
    changed = copy.deepcopy(a)
    changed["runtime"]["dtype"] = "float16"
    with pytest.raises(ValueError, match="runtime"):
        campaign.project_capture(plan, changed)
    with pytest.raises(ValueError, match="reversed"):
        campaign.publish(plan, b, a, Ed25519PrivateKey.generate())
    evidence = campaign.publish(plan, a, b, Ed25519PrivateKey.generate())
    expected = {
        "expected_plan": campaign.digest(plan),
        "expected_baseline_capture": campaign.digest(a),
        "expected_candidate_capture": campaign.digest(b),
    }
    key = Ed25519PrivateKey.generate().public_key()
    with pytest.raises(ValueError, match="signature"):
        campaign.verify(plan, a, b, evidence, key, **expected)
    with pytest.raises(ValueError, match="expected plan"):
        campaign.verify(
            plan, a, b, evidence, key, **{**expected, "expected_plan": "wrong"}
        )
    with pytest.raises(ValueError, match="cohorts"):
        campaign.verify(plan, a, b, {}, key, **expected)


def test_download_authenticates_only_enumerated_pinned_files(tmp_path, monkeypatch):
    source = tmp_path / "source.safetensors"
    item = _tensor_file(source)
    selected = {
        "model": {
            "repository": "fixture/model",
            "baseline": {"revision": "a" * 40, "files": [item]},
        }
    }
    monkeypatch.setattr(campaign, "select_plan", lambda model: selected)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    target = tmp_path / "model"
    assert (
        campaign.main(
            [
                "download",
                "--model",
                "fixture",
                "--role",
                "baseline",
                "--output",
                str(target),
            ]
        )
        == 0
    )
    assert calls == [
        {"repo_id": "fixture/model", "revision": "a" * 40, "filename": source.name}
    ]
    assert (target / source.name).read_bytes() == source.read_bytes()
    result = tmp_path / "measurement.json"
    assert (
        campaign.main(
            [
                "measure",
                "--model",
                "fixture",
                "--role",
                "baseline",
                "--snapshot",
                str(target),
                "--output",
                str(result),
            ]
        )
        == 0
    )
    assert campaign.read_json(result)["tensors"]
    monkeypatch.setattr(
        campaign.shutil, "disk_usage", lambda path: SimpleNamespace(free=0)
    )
    with pytest.raises(ValueError, match="disk"):
        campaign.download_snapshot("fixture", "baseline", tmp_path / "another")


def test_freeze_requires_validated_build_and_preserves_candidate_status(tmp_path):
    ready = _ready_plan()
    build = {
        "format": "invarlock/k2-runtime-build-v1",
        "status": "ready",
        "source_commit": ready["runtime"]["source_commit"],
        "reviewed_source_files": ready["runtime"]["reviewed_source_files"],
        **{key: ready["runtime"][key] for key in campaign._RUNTIME_BINDINGS},
    }
    campaign.write_json(tmp_path / "build.json", build)
    for role in campaign.ROLES:
        campaign.write_json(
            tmp_path / f"{role}.json", ready["model"][role]["materialized"]
        )
    args = [
        "freeze",
        "--model",
        "0.9b",
        "--runtime-build",
        str(tmp_path / "build.json"),
        "--baseline-measurement",
        str(tmp_path / "baseline.json"),
        "--candidate-measurement",
        str(tmp_path / "candidate.json"),
        "--maximum-wall-seconds",
        "3600",
        "--maximum-output-tokens",
        "400000",
        "--output",
        str(tmp_path / "frozen.json"),
    ]
    assert campaign.main(args) == 0
    frozen = campaign.read_json(tmp_path / "frozen.json")
    assert frozen["status"] == "candidate_not_qualified"
    assert frozen["runtime"]["build_manifest_digest"] == campaign.digest(build)
    build["status"] = "blocked"
    (tmp_path / "build.json").write_text(json.dumps(build))
    with pytest.raises(SystemExit) as error:
        campaign.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize(
    "outcome,expected", [("pass", 0), ("regression", 1), ("incomplete", 3)]
)
def test_cli_signed_journey_retains_all_outcomes(tmp_path, outcome, expected):
    plan = _ready_plan()
    left, right = (
        _capture(plan, "baseline"),
        _capture(plan, "candidate", wrong=outcome == "regression"),
    )
    if outcome == "incomplete":
        right["rows"][0]["error"] = "native timeout"
    key = Ed25519PrivateKey.generate()
    (tmp_path / "private.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    (tmp_path / "public.pem").write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    for name, value in [("plan", plan), ("baseline", left), ("candidate", right)]:
        campaign.write_json(tmp_path / f"{name}.json", value)
    common = [
        part
        for name in ("plan", "baseline", "candidate")
        for part in (f"--{name}", str(tmp_path / f"{name}.json"))
    ]
    assert (
        campaign.main(
            [
                "publish",
                *common,
                "--key",
                str(tmp_path / "private.pem"),
                "--output",
                str(tmp_path / "evidence.json"),
            ]
        )
        == expected
    )
    assert (
        campaign.main(
            [
                "verify",
                *common,
                "--key",
                str(tmp_path / "public.pem"),
                "--evidence",
                str(tmp_path / "evidence.json"),
                "--output",
                str(tmp_path / "verified.json"),
                "--expected-plan",
                campaign.digest(plan),
                "--expected-baseline-capture",
                campaign.digest(left),
                "--expected-candidate-capture",
                campaign.digest(right),
            ]
        )
        == 0
    )
    assert len(campaign.read_json(tmp_path / "verified.json")) == 3
    assert (
        campaign.main(
            [
                "report",
                "--evidence",
                str(tmp_path / "evidence.json"),
                "--output",
                str(tmp_path / "reports"),
            ]
        )
        == 0
    )
    for cohort in campaign.COHORTS:
        assert (
            "Release comparison"
            in (tmp_path / "reports" / f"{cohort}.html").read_text()
        )
        assert (
            (tmp_path / "reports" / f"{cohort}.xml").read_bytes().startswith(b"<?xml")
        )
