"""Package identity must never substitute for the evaluated model contents."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from examples.integrations import modelkit_handoff as handoff
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256


def _blob(store: Path, data: bytes) -> dict:
    digest = hashlib.sha256(data).hexdigest()
    store.mkdir(exist_ok=True)
    (store / digest).write_bytes(data)
    return {"digest": f"sha256:{digest}", "size": len(data)}


def _package(
    tmp_path: Path,
    *,
    description: str = "Synthetic serialization fixture; no inference",
    compressed: bool = False,
    members: list[tuple[str, bytes | str]] | None = None,
) -> tuple[Path, str, Path]:
    store = tmp_path / "blobs"
    model = tmp_path / "candidate"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_bytes(b'{"model_type":"fixture"}\n')
    (model / "model.safetensors").write_bytes(b"not executable weights\n")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name, value in members or [
            ("model/config.json", (model / "config.json").read_bytes()),
            ("model/model.safetensors", (model / "model.safetensors").read_bytes()),
        ]:
            info = tarfile.TarInfo(name)
            if isinstance(value, str):
                info.type = tarfile.SYMTYPE
                info.linkname = value
                tar.addfile(info)
            else:
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
    raw = archive.getvalue()
    layer = _blob(store, gzip.compress(raw) if compressed else raw)
    layer["mediaType"] = handoff.MODEL_TAR + ("+gzip" if compressed else "")
    config = _blob(
        store,
        json.dumps(
            {
                "manifestVersion": "1.0.0",
                "package": {"description": description},
                "model": {
                    "path": "model",
                    "digest": layer["digest"],
                    "diffId": "sha256:" + hashlib.sha256(raw).hexdigest(),
                },
            }
        ).encode(),
    )
    config["mediaType"] = handoff.CONFIG_MEDIA
    manifest = _blob(
        store,
        json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": handoff.MANIFEST_MEDIA,
                "artifactType": handoff.ARTIFACT_MEDIA,
                "config": config,
                "layers": [layer],
            }
        ).encode(),
    )
    return store, manifest["digest"], model


def _verify(store: Path, digest: str, model: Path, **kwargs):
    return handoff.verify_package_content(
        blobs=store,
        expected_package_digest=digest,
        candidate=model,
        expected_content_digest=checkpoint_tree_sha256(model),
        **kwargs,
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_package_and_actual_candidate_have_separate_identities(tmp_path, compressed):
    store, digest, model = _package(tmp_path, compressed=compressed)
    result = _verify(store, digest, model)
    assert result["package_digest"] == digest
    assert result["artifact_content_digest"] == checkpoint_tree_sha256(model)
    assert result["artifact_content_digest"] != digest
    assert result["model_path"] == "model"
    assert result["format"] == "invarlock/example-modelkit-content-v1"


def test_repacking_changes_package_identity_but_not_model_identity(tmp_path):
    store, first, model = _package(tmp_path)
    original = _verify(store, first, model)
    _, second, _ = _package(tmp_path, description="Repacked same contents")
    repacked = _verify(store, second, model)
    assert first != second
    assert original["artifact_content_digest"] == repacked["artifact_content_digest"]
    (store / first.removeprefix("sha256:")).write_bytes(
        (store / second.removeprefix("sha256:")).read_bytes()
    )
    with pytest.raises(handoff.ModelKitError, match="digest"):
        _verify(store, first, model)


@pytest.mark.parametrize("change", ["weights", "config", "extra", "missing"])
def test_actual_candidate_replacement_fails(tmp_path, change):
    store, digest, model = _package(tmp_path)
    if change == "weights":
        (model / "model.safetensors").write_bytes(b"replacement")
    elif change == "config":
        (model / "config.json").write_bytes(b"{}")
    elif change == "extra":
        (model / "tokenizer.json").write_bytes(b"{}")
    else:
        (model / "config.json").unlink()
    with pytest.raises(handoff.ModelKitError, match="content"):
        _verify(store, digest, model)


@pytest.mark.parametrize("reference", ["latest", "model:v1", "sha256:abc", "../x"])
def test_mutable_or_malformed_expected_package_reference_fails(tmp_path, reference):
    store, _, model = _package(tmp_path)
    with pytest.raises(handoff.ModelKitError, match="digest"):
        _verify(store, reference, model)


@pytest.mark.parametrize(
    "members",
    [
        [("../escape", b"x")],
        [("/absolute", b"x")],
        [("model/../escape", b"x")],
        [("other/file", b"x")],
        [("model/link", "../../escape")],
        [("model/file", b"one"), ("model/file", b"two")],
    ],
)
def test_unsafe_archives_fail_without_extracting_outside_workspace(tmp_path, members):
    store, digest, model = _package(tmp_path, members=members)
    with pytest.raises(handoff.ModelKitError):
        _verify(store, digest, model)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "limit", ["max_archive_bytes", "max_members", "max_model_bytes"]
)
def test_archive_resource_limits_are_enforced(tmp_path, limit):
    store, digest, model = _package(tmp_path, compressed=True)
    with pytest.raises(handoff.ModelKitError, match="limit"):
        _verify(store, digest, model, limits=handoff.Limits(**{limit: 1}))


def test_package_mapping_also_compares_files_excluded_from_checkpoint_identity(
    tmp_path,
):
    store, digest, model = _package(tmp_path)
    original = checkpoint_tree_sha256(model)
    (model / "logs").mkdir()
    (model / "logs" / "unexpected.py").write_bytes(b"extra package content")
    assert checkpoint_tree_sha256(model) == original
    with pytest.raises(handoff.ModelKitError, match="file inventory"):
        _verify(store, digest, model)


@pytest.fixture
def recipient_request(tmp_path):
    import shutil

    from examples.run_acceptance_handoff import run_handoff

    workspace = tmp_path / "transaction"
    run_handoff(workspace)
    incoming = workspace / "handoff"
    recipient = workspace / "recipient"
    sides = {}
    for role in ("baseline", "subject"):
        packaging = tmp_path / role
        packaging.mkdir()
        artifact = incoming / "artifacts" / role
        members = [
            (f"model/{p.name}", p.read_bytes()) for p in sorted(artifact.iterdir())
        ]
        store, digest, model = _package(packaging, members=members)
        shutil.rmtree(model)
        shutil.copytree(artifact, model)
        sides[role] = {
            "blobs": str(store),
            "package_digest": digest,
            "candidate": str(model),
            "content_digest": checkpoint_tree_sha256(model),
        }
    anchors = json.loads((recipient / "trust/technical-anchors.json").read_bytes())
    return {
        "format": "invarlock/example-modelkit-recipient-v1",
        "sides": sides,
        "evidence": str(incoming / "evidence"),
        "technical_policy": str(incoming / "policy/acceptance.json"),
        "technical_anchors": {
            key: anchors[key]
            for key in (
                "artifact_digests",
                "schedule_digest",
                "runtime_digests",
                "evidence_signer_fingerprint",
            )
        },
        "envelope": str(incoming / "acceptance.dsse.json"),
        "recipient_policy": str(recipient / "policy.json"),
        "trusted_public_keys": {
            anchors["envelope_signer_fingerprint"]: str(
                recipient / "trust/envelope-signer.public.pem"
            )
        },
    }


def test_point_of_use_verifies_actual_packages_evidence_and_current_acceptance(
    recipient_request,
):
    result = handoff.verify_point_of_use(
        recipient_request, now=datetime(2026, 7, 25, 12, 5, tzinfo=UTC)
    )
    assert result["accepted"] is True
    assert result["technical_integrity_ok"] is True
    assert result["technical_policy_verdict"] == "pass"
    assert result["envelope_evidence_bound"] is True
    assert result["exit_code"] == 0


def test_independent_recipient_fixture_satisfies_closed_request_schema(
    recipient_request,
):
    import jsonschema

    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/integrations/modelkit-handoff/recipient.schema.json"
        ).read_bytes()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(recipient_request)
    recipient_request["sides"]["subject"]["extra"] = "untrusted"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(recipient_request)


def test_stale_acceptance_preserves_valid_historical_technical_result(
    recipient_request,
):
    result = handoff.verify_point_of_use(
        recipient_request, now=datetime(2027, 7, 25, tzinfo=UTC)
    )
    assert result["accepted"] is False
    assert result["technical_integrity_ok"] is True
    assert result["technical_policy_verdict"] == "pass"
    assert result["exit_code"] == 1


def test_wrong_independent_runtime_anchor_cannot_be_overridden_by_envelope(
    recipient_request,
):
    recipient_request["technical_anchors"]["runtime_digests"]["subject"] = (
        "sha256:" + "a" * 64
    )
    result = handoff.verify_point_of_use(
        recipient_request, now=datetime(2026, 7, 25, 12, 5, tzinfo=UTC)
    )
    assert result["accepted"] is False
    assert result["technical_integrity_ok"] is False
    assert result["exit_code"] == 2


def test_independent_request_anchor_binds_complete_evaluation_context(
    recipient_request,
):
    import jsonschema

    from invarlock.evidence_pack_contract import canonical_json_bytes

    evidence = Path(recipient_request["evidence"])
    manifest = json.loads((evidence / "manifest.json").read_bytes())
    request_path = evidence / manifest["evidence"]["request"]["path"]
    normalized_request = json.loads(request_path.read_bytes())
    request_digest = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(normalized_request)).hexdigest()
    )
    recipient_request["technical_anchors"]["request_digest"] = request_digest
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/integrations/modelkit-handoff/recipient.schema.json"
        ).read_bytes()
    )
    jsonschema.Draft202012Validator(schema).validate(recipient_request)
    result = handoff.verify_point_of_use(
        recipient_request, now=datetime(2026, 7, 25, 12, 5, tzinfo=UTC)
    )
    assert result["accepted"] is True
    assert result["technical_integrity_ok"] is True

    # Package/content identities and both signatures remain valid. Only the
    # recipient's independently selected complete-request expectation changes.
    recipient_request["technical_anchors"]["request_digest"] = "sha256:" + "0" * 64
    rejected = handoff.verify_point_of_use(
        recipient_request, now=datetime(2026, 7, 25, 12, 5, tzinfo=UTC)
    )
    assert rejected["accepted"] is False
    assert rejected["technical_integrity_ok"] is False
    assert rejected["envelope_authenticated"] is True
    assert rejected["receipt_authenticated"] is True
    assert rejected["exit_code"] == 2
    assert any("independent request anchor" in error for error in rejected["errors"])


def _rewrite(store, digest, *, manifest_change=None, config_change=None):
    manifest = json.loads((store / digest[7:]).read_bytes())
    if config_change:
        config = json.loads((store / manifest["config"]["digest"][7:]).read_bytes())
        config_change(config)
        descriptor = _blob(store, json.dumps(config).encode())
        manifest["config"].update(descriptor)
    if manifest_change:
        manifest_change(manifest)
    return _blob(store, json.dumps(manifest).encode())["digest"]


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.pop("config"),
        lambda m: m.update(subject={}),
        lambda m: m.update(schemaVersion=True),
        lambda m: m.update(mediaType="unsupported"),
        lambda m: m["config"].update(mediaType=[]),
        lambda m: m["config"].update(size=-1),
        lambda m: m["config"].update(size=m["config"]["size"] + 1),
        lambda m: m.update(layers=m["layers"] * 2),
        lambda m: m["layers"][0].update(mediaType=handoff.MODEL_TAR + "+zstd"),
    ],
)
def test_unsupported_or_contradictory_manifest_fails(tmp_path, change):
    store, digest, model = _package(tmp_path)
    digest = _rewrite(store, digest, manifest_change=change)
    with pytest.raises(handoff.ModelKitError):
        _verify(store, digest, model)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(manifestVersion="2.0.0"),
        lambda c: c.update(code=[{"path": "arbitrary"}]),
        lambda c: c["model"].update(parts=[]),
        lambda c: c["model"].update(path=""),
        lambda c: c["model"].update(path="model\\nested"),
        lambda c: c["model"].update(digest="sha256:" + "e" * 64),
        lambda c: c["model"].update(diffId="sha256:" + "f" * 64),
    ],
)
def test_unsupported_or_contradictory_model_config_fails(tmp_path, change):
    store, digest, model = _package(tmp_path)
    digest = _rewrite(store, digest, config_change=change)
    with pytest.raises(handoff.ModelKitError):
        _verify(store, digest, model)


@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_ambiguous_json_fails(tmp_path, raw):
    store, _, model = _package(tmp_path)
    digest = _blob(store, raw)["digest"]
    with pytest.raises(handoff.ModelKitError, match="JSON"):
        _verify(store, digest, model)


@pytest.mark.parametrize("field", ["max_json_bytes", "max_blob_bytes"])
def test_blob_limits_are_enforced_before_loading(tmp_path, field):
    store, digest, model = _package(tmp_path)
    with pytest.raises(handoff.ModelKitError, match="limit"):
        _verify(store, digest, model, limits=handoff.Limits(**{field: 1}))


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_recipient_cannot_use_invalid_resource_limits(value):
    with pytest.raises(handoff.ModelKitError, match="limits"):
        handoff.Limits(max_members=value)


def test_missing_or_symlinked_blob_is_rejected(tmp_path):
    store, digest, model = _package(tmp_path)
    blob = store / digest[7:]
    other = store / "other"
    blob.rename(other)
    with pytest.raises(handoff.ModelKitError):
        _verify(store, digest, model)
    blob.symlink_to(other)
    with pytest.raises(handoff.ModelKitError):
        _verify(store, digest, model)


def test_candidate_mutation_during_package_read_is_detected(tmp_path, monkeypatch):
    store, digest, model = _package(tmp_path)
    original = handoff._model_descriptor

    def mutate(*args):
        value = original(*args)
        (model / "config.json").write_bytes(b"replacement")
        return value

    monkeypatch.setattr(handoff, "_model_descriptor", mutate)
    with pytest.raises(handoff.ModelKitError, match="changed"):
        _verify(store, digest, model)


def test_new_package_cannot_reuse_old_content_anchor(tmp_path):
    store, digest, model = _package(tmp_path)
    expected = checkpoint_tree_sha256(model)
    (model / "config.json").write_bytes(b"changed")
    with pytest.raises(handoff.ModelKitError, match="expected content"):
        handoff.verify_package_content(
            blobs=store,
            expected_package_digest=digest,
            candidate=model,
            expected_content_digest=expected,
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(format="unrecognized"),
        lambda r: r.update(trusted_public_keys={}),
        lambda r: r["sides"]["subject"].update(candidate=12),
        lambda r: r["technical_anchors"].update(schedule_digest="latest"),
        lambda r: r["technical_anchors"].update(request_digest="latest"),
        lambda r: r["technical_anchors"].update(request_digest=None),
    ],
)
def test_recipient_configuration_fails_closed(recipient_request, change):
    change(recipient_request)
    with pytest.raises(handoff.ModelKitError):
        handoff.verify_point_of_use(recipient_request)


def test_unknown_envelope_key_is_not_trusted_from_package(recipient_request):
    path = next(iter(recipient_request["trusted_public_keys"].values()))
    recipient_request["trusted_public_keys"] = {"sha256:" + "f" * 64: path}
    result = handoff.verify_point_of_use(recipient_request)
    assert result["accepted"] is False
    assert result["envelope_authenticated"] is False
    assert result["exit_code"] == 2


def test_cli_reports_current_rejection_and_invalid_request(
    recipient_request, tmp_path, capsys
):
    request_path = tmp_path / "recipient.json"
    request_path.write_text(json.dumps(recipient_request))
    assert handoff.main(["--request", str(request_path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["technical_integrity_ok"] is True
    assert result["accepted"] is False
    request_path.write_text('{"format": NaN}')
    assert handoff.main(["--request", str(request_path)]) == 2
    assert json.loads(capsys.readouterr().out)["accepted"] is False


def _archive(entries):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        for name, contents in entries:
            item = tarfile.TarInfo(name)
            if contents is None:
                item.type = tarfile.DIRTYPE
                archive.addfile(item)
            else:
                item.size = len(contents)
                archive.addfile(item, io.BytesIO(contents))
    data.seek(0)
    return data


def test_real_tar_parent_directories_and_nested_model_files(tmp_path):
    from pathlib import PurePosixPath

    archive = _archive(
        [
            ("assets/", None),
            ("assets/model/", None),
            ("assets/model/config.json", b"{}"),
        ]
    )
    files, total = handoff._extract(
        archive, tmp_path, PurePosixPath("assets/model"), handoff.Limits()
    )
    assert set(files) == {"config.json"}
    assert total == 2
    assert (tmp_path / "assets/model/config.json").read_bytes() == b"{}"


@pytest.mark.parametrize(
    "entries,limits",
    [
        ([("model", None)], handoff.Limits()),
        ([("model/a", b"a"), ("model/b", b"b")], handoff.Limits(max_members=1)),
        ([("model/large", b"too large")], handoff.Limits(max_model_bytes=1)),
        ([("model", b"not a directory")], handoff.Limits()),
    ],
)
def test_archive_limits_and_unsupported_empty_or_single_file_model(
    tmp_path, entries, limits
):
    from pathlib import PurePosixPath

    with pytest.raises(handoff.ModelKitError):
        handoff._extract(_archive(entries), tmp_path, PurePosixPath("model"), limits)


def test_named_pipe_blob_does_not_block_or_get_read(tmp_path):
    import os

    store, digest, model = _package(tmp_path)
    path = store / digest[7:]
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(handoff.ModelKitError, match="regular file"):
        _verify(store, digest, model)


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_operational_directories_cannot_hide_special_files(tmp_path, kind):
    import os

    store, digest, model = _package(tmp_path)
    hidden = model / "logs"
    hidden.mkdir()
    if kind == "symlink":
        (hidden / "elsewhere").symlink_to(tmp_path, target_is_directory=True)
    else:
        os.mkfifo(hidden / "pipe")
    with pytest.raises(handoff.ModelKitError):
        _verify(store, digest, model)


def test_blob_mutation_during_read_is_detected(tmp_path, monkeypatch):
    store, digest, model = _package(tmp_path)
    original = handoff._copy

    def mutate(source, target, maximum):
        result = original(source, target, maximum)
        if target is not None:
            (store / digest[7:]).write_bytes(b"changed after read")
        return result

    monkeypatch.setattr(handoff, "_copy", mutate)
    with pytest.raises(handoff.ModelKitError, match="changed"):
        _verify(store, digest, model)


def test_candidate_mutation_while_reading_its_inventory_is_detected(
    tmp_path, monkeypatch
):
    store, digest, model = _package(tmp_path)
    original = handoff._copy

    def mutate(source, target, maximum):
        result = original(source, target, maximum)
        if target is None:
            for path in model.iterdir():
                path.write_bytes(b"changed after read")
        return result

    monkeypatch.setattr(handoff, "_copy", mutate)
    with pytest.raises(handoff.ModelKitError, match="changed"):
        _verify(store, digest, model)


def test_candidate_swap_during_acceptance_is_detected(recipient_request, monkeypatch):
    original = handoff.verify_acceptance_attestation

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        candidate = Path(recipient_request["sides"]["subject"]["candidate"])
        (candidate / "model.safetensors").write_bytes(b"swapped at point of use")
        return result

    monkeypatch.setattr(handoff, "verify_acceptance_attestation", mutate)
    with pytest.raises(handoff.ModelKitError, match="changed"):
        handoff.verify_point_of_use(recipient_request)


def test_excluded_file_mutation_during_technical_verification_is_detected(
    recipient_request,
    monkeypatch,
):
    side = recipient_request["sides"]["subject"]
    candidate, store = Path(side["candidate"]), Path(side["blobs"])
    (candidate / "logs").mkdir()
    status = candidate / "logs/status.txt"
    status.write_bytes(b"original")
    archive = _archive(
        [
            ("model/" + p.relative_to(candidate).as_posix(), p.read_bytes())
            for p in sorted(candidate.rglob("*"))
            if p.is_file()
        ]
    ).getvalue()
    layer = _blob(store, archive)
    layer["mediaType"] = handoff.MODEL_TAR
    side["package_digest"] = _rewrite(
        store,
        side["package_digest"],
        config_change=lambda c: c["model"].update(
            digest=layer["digest"], diffId=layer["digest"]
        ),
        manifest_change=lambda m: m.update(layers=[layer]),
    )
    original = handoff.verify_comparison_evidence

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        status.write_bytes(b"replaced during technical verification")
        return result

    monkeypatch.setattr(handoff, "verify_comparison_evidence", mutate)
    with pytest.raises(handoff.ModelKitError, match="file inventory"):
        handoff.verify_point_of_use(
            recipient_request, now=datetime(2026, 7, 25, 12, 5, tzinfo=UTC)
        )


def test_unreadable_excluded_directory_cannot_disappear_from_inventory(tmp_path):
    import os

    if os.geteuid() == 0:
        pytest.skip("root can read files despite mode000")
    store, digest, model = _package(tmp_path)
    logs = model / "logs"
    logs.mkdir()
    (logs / "unpackaged.bin").write_bytes(b"must not be omitted")
    logs.chmod(0)
    try:
        with pytest.raises(handoff.ModelKitError, match="inventory"):
            _verify(store, digest, model)
    finally:
        logs.chmod(0o700)


@pytest.mark.parametrize(
    "header_type",
    [
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_SPARSE,
    ],
)
def test_extended_archive_headers_are_rejected_before_payload_parsing(
    tmp_path, header_type
):
    from pathlib import PurePosixPath

    item = tarfile.TarInfo("metadata")
    item.type = header_type
    item.size = 1024**3
    # There is deliberately no huge payload. Header validation must reject the
    # unsupported type before the parser tries to consume its declared contents.
    archive = io.BytesIO(item.tobuf())
    with pytest.raises(handoff.ModelKitError, match="header"):
        handoff._extract(archive, tmp_path, PurePosixPath("model"), handoff.Limits())


@pytest.mark.parametrize("raw", [b"short", b"\0" * 512, b"\0" * 1024 + b"unexpected"])
def test_incomplete_or_concatenated_archives_are_rejected(tmp_path, raw):
    from pathlib import PurePosixPath

    with pytest.raises(handoff.ModelKitError, match="header"):
        handoff._extract(
            io.BytesIO(raw), tmp_path, PurePosixPath("model"), handoff.Limits()
        )


def test_regular_archive_header_cannot_claim_missing_payload(tmp_path):
    from pathlib import PurePosixPath

    item = tarfile.TarInfo("model/weights")
    item.size = 10_000
    with pytest.raises(handoff.ModelKitError, match="available contents"):
        handoff._extract(
            io.BytesIO(item.tobuf()), tmp_path, PurePosixPath("model"), handoff.Limits()
        )


def test_directory_header_cannot_hide_file_contents(tmp_path):
    from pathlib import PurePosixPath

    item = tarfile.TarInfo("model")
    item.type = tarfile.DIRTYPE
    item.size = 512
    archive = io.BytesIO(item.tobuf() + b"hidden payload".ljust(512, b"\0"))
    with pytest.raises(handoff.ModelKitError, match="invalid archive header size"):
        handoff._extract(archive, tmp_path, PurePosixPath("model"), handoff.Limits())
    assert not (tmp_path / "model").exists()


def test_excluded_file_added_during_final_candidate_check_is_detected(
    tmp_path, monkeypatch
):
    store, digest, model = _package(tmp_path)
    original = handoff.checkpoint_tree_observation
    observations = []

    def mutate(path):
        observed = original(path)
        observations.append(observed)
        if len(observations) == 2:
            (model / "logs").mkdir()
            (model / "logs" / "new-content.txt").write_bytes(b"unpackaged content")
        return observed

    monkeypatch.setattr(handoff, "checkpoint_tree_observation", mutate)
    with pytest.raises(handoff.ModelKitError, match="file inventory changed"):
        _verify(store, digest, model)
    assert len(observations) == 2
    assert observations[0] == observations[1]
    assert checkpoint_tree_sha256(model) == observations[0].digest


def test_reading_an_operational_file_may_update_access_time(tmp_path):
    import os

    store, digest, model = _package(
        tmp_path,
        members=[
            ("model/config.json", b'{"model_type":"fixture"}\n'),
            ("model/model.safetensors", b"not executable weights\n"),
            ("model/logs/status.txt", b"unchanged operational content"),
        ],
    )
    (model / "logs").mkdir()
    status = model / "logs/status.txt"
    status.write_bytes(b"unchanged operational content")
    # Access time is allowed to change from reading; content/mtime/ctime and
    # inode substitutions remain guarded. This matters on Linux relatime mounts.
    os.utime(status, ns=(1_000_000_000, status.stat().st_mtime_ns))
    result = _verify(store, digest, model)
    assert result["model_file_count"] == 3
