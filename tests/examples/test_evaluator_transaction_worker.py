from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

import examples.integrations.evaluator_transaction.worker as oci
import examples.integrations.trust_material as trust_material
from examples.integrations.evaluator_transaction.build_attestation import (
    Level3BuildAttestationError,
    load_level3_build_attestation,
    make_level3_build_attestation,
    sign_level3_build_attestation,
    validate_level3_build_attestation,
    verify_level3_build_attestation,
    write_level3_build_attestation,
)
from examples.integrations.trust_material import (
    create_trust_material,
    load_external_key,
    read_external_file,
    validate_new_trust_root,
)
from invarlock.evidence_pack_contract import canonical_json_bytes

IMAGE = "sha256:" + "a" * 64
BASE = "sha256:" + "b" * 64
COMMIT = "c" * 40
SOURCE = "sha256:" + "d" * 64
LOCK = "sha256:" + "e" * 64
ENTRYPOINT = ("python", "/opt/evaluator.py", "worker")


def _attestation(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "evaluator": "inspect-ai",
        "evaluator_version": "0.3.254",
        "runtime_image_id": IMAGE,
        "base_image_id": BASE,
        "source_commit": COMMIT,
        "source_bundle_sha256": SOURCE,
        "lock_sha256": LOCK,
        "entrypoint": list(ENTRYPOINT),
        "base_layers": [BASE],
        "image_layers": [BASE, "sha256:" + "f" * 64],
        "config": {"Entrypoint": list(ENTRYPOINT), "Labels": {}},
    }
    values.update(overrides)
    config = values.pop("config")
    return make_level3_build_attestation(
        **values,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
    )


def test_build_attestation_is_canonical_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "build-attestation.json"
    payload = _attestation()
    signed = sign_level3_build_attestation(
        payload, ed25519.Ed25519PrivateKey.generate()
    )
    write_level3_build_attestation(path, signed)

    assert load_level3_build_attestation(path) == signed
    assert path.read_bytes() == canonical_json_bytes(signed)

    with pytest.raises(Level3BuildAttestationError, match="destination exists"):
        write_level3_build_attestation(path, signed)


def test_builder_signature_binds_the_complete_build_statement() -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    payload = _attestation()
    signed = sign_level3_build_attestation(payload, key)

    assert (
        verify_level3_build_attestation(
            signed,
            builder_public_key=key.public_key(),
            evaluator="inspect-ai",
            evaluator_version="0.3.254",
            runtime_image_id=IMAGE,
            base_image_id=BASE,
            source_commit=COMMIT,
            source_bundle_sha256=SOURCE,
            lock_sha256=LOCK,
            entrypoint=ENTRYPOINT,
        )
        == payload
    )

    tampered = dict(signed)
    tampered["statement"] = {**payload, "image_layers": [BASE]}
    with pytest.raises(Level3BuildAttestationError, match="does not verify"):
        verify_level3_build_attestation(
            tampered,
            builder_public_key=key.public_key(),
            evaluator="inspect-ai",
            evaluator_version="0.3.254",
            runtime_image_id=IMAGE,
            base_image_id=BASE,
            source_commit=COMMIT,
            source_bundle_sha256=SOURCE,
            lock_sha256=LOCK,
            entrypoint=ENTRYPOINT,
        )

    with pytest.raises(Level3BuildAttestationError, match="not trusted"):
        verify_level3_build_attestation(
            signed,
            builder_public_key=ed25519.Ed25519PrivateKey.generate().public_key(),
            evaluator="inspect-ai",
            evaluator_version="0.3.254",
            runtime_image_id=IMAGE,
            base_image_id=BASE,
            source_commit=COMMIT,
            source_bundle_sha256=SOURCE,
            lock_sha256=LOCK,
            entrypoint=ENTRYPOINT,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: "not-an-object", "must be an object"),
        (lambda value: {**value, "extra": True}, "unexpected fields"),
        (lambda value: {**value, "format_version": "wrong"}, "format is invalid"),
        (lambda value: {**value, "evaluator": "other"}, "evaluator is invalid"),
        (lambda value: {**value, "entrypoint": ["bad\x00"]}, "entrypoint must"),
        (lambda value: {**value, "base_layers": ["bad"]}, "base layer chain"),
        (lambda value: {**value, "config_sha256": "bad"}, "OCI config digest"),
    ],
)
def test_build_attestation_rejects_malformed_contract_fields(
    mutator: object, message: str
) -> None:
    payload = _attestation()
    invalid = mutator(payload)  # type: ignore[operator]
    with pytest.raises(Level3BuildAttestationError, match=message):
        validate_level3_build_attestation(
            invalid,
            evaluator="inspect-ai",
            evaluator_version="0.3.254",
            runtime_image_id=IMAGE,
            base_image_id=BASE,
            source_commit=COMMIT,
            source_bundle_sha256=SOURCE,
            lock_sha256=LOCK,
            entrypoint=ENTRYPOINT,
        )


def test_signed_build_attestation_rejects_invalid_envelopes_and_files(
    tmp_path: Path,
) -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    payload = _attestation()
    signed = sign_level3_build_attestation(payload, key)
    for invalid, message in (
        ([], "envelope is invalid"),
        ({**signed, "format_version": "wrong"}, "format is invalid"),
        ({**signed, "signature": {}}, "signature is invalid"),
        (
            {**signed, "signature": {**signed["signature"], "algorithm": "rsa"}},
            "algorithm is invalid",
        ),
        (
            {**signed, "signature": {**signed["signature"], "value": "%%%"}},
            "encoding is invalid",
        ),
    ):
        with pytest.raises(Level3BuildAttestationError, match=message):
            verify_level3_build_attestation(
                invalid,
                builder_public_key=key.public_key(),
                evaluator="inspect-ai",
                evaluator_version="0.3.254",
                runtime_image_id=IMAGE,
                base_image_id=BASE,
                source_commit=COMMIT,
                source_bundle_sha256=SOURCE,
                lock_sha256=LOCK,
                entrypoint=ENTRYPOINT,
            )
    unsigned = tmp_path / "unsigned.json"
    unsigned.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Level3BuildAttestationError, match="unsigned"):
        load_level3_build_attestation(unsigned)
    with pytest.raises(Level3BuildAttestationError, match="only signed"):
        write_level3_build_attestation(tmp_path / "invalid.json", payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entrypoint", ["/bin/sh"]),
        ("runtime_image_id", "sha256:" + "f" * 64),
        ("image_layers", ["not-a-digest"]),
    ],
)
def test_build_attestation_rejects_profile_or_identity_tampering(
    field: str, value: object
) -> None:
    payload = _attestation()
    payload[field] = value

    with pytest.raises(Level3BuildAttestationError):
        validate_level3_build_attestation(
            payload,
            evaluator="inspect-ai",
            evaluator_version="0.3.254",
            runtime_image_id=IMAGE,
            base_image_id=BASE,
            source_commit=COMMIT,
            source_bundle_sha256=SOURCE,
            lock_sha256=LOCK,
            entrypoint=ENTRYPOINT,
        )


def test_build_attestation_rejects_non_strings_and_version_mismatches() -> None:
    payload = _attestation()
    payload["config_sha256"] = None
    with pytest.raises(Level3BuildAttestationError, match="OCI config digest"):
        validate_level3_build_attestation(
            payload,
            evaluator="inspect-ai",
            evaluator_version="0.3.254",
            runtime_image_id=IMAGE,
            base_image_id=BASE,
            source_commit=COMMIT,
            source_bundle_sha256=SOURCE,
            lock_sha256=LOCK,
            entrypoint=ENTRYPOINT,
        )
    payload = _attestation()
    payload["evaluator_version"] = "wrong"
    with pytest.raises(Level3BuildAttestationError, match="version is invalid"):
        validate_level3_build_attestation(
            payload,
            evaluator="inspect-ai",
            evaluator_version="0.3.254",
            runtime_image_id=IMAGE,
            base_image_id=BASE,
            source_commit=COMMIT,
            source_bundle_sha256=SOURCE,
            lock_sha256=LOCK,
            entrypoint=ENTRYPOINT,
        )


def test_level3_command_keeps_control_file_private_and_output_bounded(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "transaction" / "result"
    output.parent.mkdir()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)

    command = oci.compose_evaluator_worker_command(
        engine="docker",
        image=IMAGE,
        entrypoint=ENTRYPOINT,
        worker_arguments=("--output", "/outputs/result"),
        model_source=model,
        dataset_source=dataset,
        output=output,
        control_root=control,
        environment={"INVARLOCK_TEST": "1"},
        timeout_seconds=30,
        output_limit_bytes=1024 * 1024,
    )

    cidfile = Path(command[command.index("--cidfile") + 1])
    assert cidfile.parent == control.resolve()
    assert str(output.parent) not in command
    tmpfs_values = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--tmpfs"
    ]
    assert any(
        value.startswith("/outputs:") and "size=1114112" in value
        for value in tmpfs_values
    )
    assert all(
        not (value.startswith("type=bind") and "target=/outputs" in value)
        for value in command
    )


def test_level3_worker_transfers_then_removes_preserved_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "transaction" / "result"
    observed_names: list[str] = []
    observed: list[list[str]] = []

    def fake_run_side_worker(
        command: list[str], *, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        assert timeout_seconds == 30
        observed_names.append(command[command.index("--name") + 1])
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_bounded_command(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        observed.append(command)
        if command[1] == "exec" and command[3] == "cat":
            return subprocess.CompletedProcess(command, 0, b"0", b"")
        if command[1] == "exec" and command[3] == "tar":
            archive_bytes = io.BytesIO()
            with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
                directory = tarfile.TarInfo("result")
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
                manifest = tarfile.TarInfo("result/run-manifest.json")
                manifest.size = 2
                archive.addfile(manifest, io.BytesIO(b"{}"))
            archive_path = kwargs["stdout_path"]
            assert isinstance(archive_path, Path)
            archive_path.write_bytes(archive_bytes.getvalue())
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(oci, "run_side_worker", fake_run_side_worker)
    monkeypatch.setattr(oci, "_run_bounded_command", fake_bounded_command)

    result = oci.run_evaluator_worker(
        engine="docker",
        image=IMAGE,
        entrypoint=ENTRYPOINT,
        worker_arguments=("--output", "/outputs/result"),
        model_source=model,
        dataset_source=dataset,
        output=output,
        timeout_seconds=30,
    )

    assert result.returncode == 0
    assert output.is_dir()
    assert len(observed_names) == 1
    container_name = observed_names[0]
    assert observed[0][1:4] == ["exec", container_name, "cat"]
    assert observed[1][1] == "exec"
    assert observed[2][1] == "rm"
    assert container_name in observed[1]


def test_level3_worker_returns_bounded_container_diagnostics_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "transaction" / "result"
    observed_name: list[str] = []

    def fake_run_side_worker(
        command: list[str], *, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        assert timeout_seconds == 30
        observed_name.append(command[command.index("--name") + 1])
        return subprocess.CompletedProcess(command, 0, "container-id", "")

    def fake_bounded_command(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        if command[1] == "exec":
            return subprocess.CompletedProcess(command, 0, b"7", b"")
        if command[1] == "logs":
            return subprocess.CompletedProcess(
                command, 0, b"native evaluator failed\n", b""
            )
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(oci, "run_side_worker", fake_run_side_worker)
    monkeypatch.setattr(oci, "_run_bounded_command", fake_bounded_command)

    result = oci.run_evaluator_worker(
        engine="docker",
        image=IMAGE,
        entrypoint=ENTRYPOINT,
        worker_arguments=("--output", "/outputs/result"),
        model_source=model,
        dataset_source=dataset,
        output=output,
        timeout_seconds=30,
    )

    assert result.returncode == 7
    assert result.stdout == ""
    assert result.stderr == "native evaluator failed\n"
    assert len(observed_name) == 1


def test_level3_output_archive_rejects_links(tmp_path: Path) -> None:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        directory = tarfile.TarInfo("result")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        link = tarfile.TarInfo("result/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc"
        archive.addfile(link)

    with pytest.raises(oci.OciEvaluationError, match="unsafe entry"):
        oci._extract_output_archive(
            archive_bytes.getvalue(),
            staging_root=tmp_path / "staging",
            output_name="result",
            max_bytes=1024,
        )


def test_trust_material_uses_new_private_root_and_external_key_snapshot(
    tmp_path: Path,
) -> None:
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    evidence_key = tmp_path / "evidence.pem"
    key = ed25519.Ed25519PrivateKey.generate()
    evidence_key.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    verifier_key = tmp_path / "verifier.pem"
    verifier = ed25519.Ed25519PrivateKey.generate()
    verifier_bytes = verifier.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    verifier_key.write_bytes(verifier_bytes)

    loaded_path, loaded_bytes, fingerprint = load_external_key(
        evidence_key,
        transaction_root=transaction,
        label="evidence signing key",
    )
    assert loaded_path == evidence_key
    assert loaded_bytes == evidence_key.read_bytes()
    assert fingerprint
    assert (
        read_external_file(verifier_key, label="verifier signing key") == verifier_bytes
    )

    trust_root = validate_new_trust_root(
        tmp_path / "trust-root", transaction_root=transaction
    )
    material = create_trust_material(
        transaction_root=transaction,
        evidence_key=evidence_key,
        verifier_key_bytes=verifier_bytes,
        evidence_fingerprint=fingerprint,
        verifier_fingerprint="verifier-fingerprint",
        trust_root=trust_root,
        policy_bytes=b'{"resolved_policy":{}}\n',
        verifier_identity="test-verifier",
        anchors={"schedule": "sha256:" + "a" * 64},
    )
    assert material.trust_root == trust_root
    assert material.verifier_key.read_bytes() == verifier_bytes
    assert material.independent_policy.read_bytes() == b'{"resolved_policy":{}}\n'
    assert material.trusted_inputs.read_text(encoding="utf-8").endswith("\n")


def test_trust_material_rejects_symlinked_external_files_and_existing_roots(
    tmp_path: Path,
) -> None:
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    key = tmp_path / "key.pem"
    key.write_bytes(b"not-a-key")
    key_link = tmp_path / "key-link.pem"
    key_link.symlink_to(key)
    with pytest.raises(ValueError, match="without following links"):
        read_external_file(key_link, label="key")
    with pytest.raises(ValueError, match="must contain"):
        load_external_key(key, transaction_root=transaction, label="key")
    public_key = ed25519.Ed25519PrivateKey.generate().public_key()
    public_path = tmp_path / "public.pem"
    public_path.write_bytes(
        public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    with pytest.raises(ValueError, match="must contain"):
        load_external_key(public_path, transaction_root=transaction, label="public")

    existing = tmp_path / "existing-root"
    existing.mkdir()
    with pytest.raises(ValueError, match="must be new"):
        validate_new_trust_root(existing, transaction_root=transaction)

    root_link = tmp_path / "root-link"
    root_link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="must be new"):
        validate_new_trust_root(root_link, transaction_root=transaction)


def test_trust_material_secure_directory_and_path_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    with pytest.raises(ValueError, match="outside"):
        trust_material._outside(transaction, transaction / "inside", label="key")
    original_directory = trust_material.os.O_DIRECTORY
    monkeypatch.setattr(trust_material.os, "O_DIRECTORY", None)
    with pytest.raises(ValueError, match="directory access"):
        trust_material._directory_flags()
    monkeypatch.setattr(trust_material.os, "O_DIRECTORY", original_directory)

    parent = trust_material.os.open(transaction, trust_material.os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="unsafe path"):
            trust_material._read_file_at(parent, "../escape", label="key")
        with pytest.raises(ValueError, match="file path is unsafe"):
            trust_material._write_new_file_at(parent, "", b"x", mode=0o600)
    finally:
        trust_material.os.close(parent)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("engine", "runc", "engine"),
        ("image", "latest", "immutable digest"),
        ("entrypoint", (), "entrypoint"),
        ("worker_arguments", ("bad\x00",), "arguments"),
        ("timeout_seconds", 0, "timeout"),
        ("output_limit_bytes", 0, "output limit"),
        ("environment", {"BAD=KEY": "value"}, "environment"),
    ],
)
def test_compose_worker_rejects_untrusted_boundary_values(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "transaction" / "result"
    output.parent.mkdir()
    control = tmp_path / "control"
    control.mkdir()
    values: dict[str, object] = {
        "engine": "docker",
        "image": IMAGE,
        "entrypoint": ENTRYPOINT,
        "worker_arguments": ("--output", "/outputs/result"),
        "model_source": model,
        "dataset_source": dataset,
        "output": output,
        "control_root": control,
        "timeout_seconds": 30,
        "output_limit_bytes": 1024,
    }
    values[field] = value

    with pytest.raises(oci.OciEvaluationError, match=match):
        oci.compose_evaluator_worker_command(**values)  # type: ignore[arg-type]


def test_compose_worker_rejects_symlink_and_existing_destinations(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    control = tmp_path / "control"
    control.mkdir()
    output = tmp_path / "transaction" / "result"
    output.parent.mkdir()
    base = {
        "engine": "docker",
        "image": IMAGE,
        "entrypoint": ENTRYPOINT,
        "worker_arguments": ("--output", "/outputs/result"),
        "model_source": model,
        "dataset_source": dataset,
        "output": output,
        "control_root": control,
        "timeout_seconds": 30,
    }

    (tmp_path / "model-link").symlink_to(model, target_is_directory=True)
    with pytest.raises(oci.OciEvaluationError, match="not a symlink"):
        oci.compose_evaluator_worker_command(
            **{**base, "model_source": tmp_path / "model-link"}
        )

    existing_output = output
    existing_output.mkdir()
    with pytest.raises(oci.OciEvaluationError, match="must be new"):
        oci.compose_evaluator_worker_command(**base)

    existing_output.rmdir()
    (control / "container.cid").write_text("x", encoding="ascii")
    with pytest.raises(oci.OciEvaluationError, match="ID destination"):
        oci.compose_evaluator_worker_command(**base)


@pytest.mark.parametrize("member_name", ["../escape", "/result/escape", "other/file"])
def test_output_archive_rejects_paths_outside_the_requested_result(
    tmp_path: Path, member_name: str
) -> None:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))

    with pytest.raises(oci.OciEvaluationError, match="unsafe path"):
        oci._extract_output_archive(
            archive_bytes.getvalue(),
            staging_root=tmp_path / "staging",
            output_name="result",
            max_bytes=1024,
        )


def test_output_archive_rejects_empty_special_duplicate_and_oversized_payloads(
    tmp_path: Path,
) -> None:
    empty = io.BytesIO()
    with tarfile.open(fileobj=empty, mode="w"):
        pass
    with pytest.raises(oci.OciEvaluationError, match="empty"):
        oci._extract_output_archive(
            empty.getvalue(),
            staging_root=tmp_path / "empty",
            output_name="result",
            max_bytes=1024,
        )

    special = io.BytesIO()
    with tarfile.open(fileobj=special, mode="w") as archive:
        member = tarfile.TarInfo("result/device")
        member.type = tarfile.CHRTYPE
        archive.addfile(member)
    with pytest.raises(oci.OciEvaluationError, match="unsafe entry"):
        oci._extract_output_archive(
            special.getvalue(),
            staging_root=tmp_path / "special",
            output_name="result",
            max_bytes=1024,
        )

    duplicate = io.BytesIO()
    with tarfile.open(fileobj=duplicate, mode="w") as archive:
        directory = tarfile.TarInfo("result")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for _ in range(2):
            member = tarfile.TarInfo("result/value")
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(oci.OciEvaluationError, match="duplicate"):
        oci._extract_output_archive(
            duplicate.getvalue(),
            staging_root=tmp_path / "duplicate",
            output_name="result",
            max_bytes=1024,
        )

    with pytest.raises(oci.OciEvaluationError, match="size limit"):
        oci._extract_output_archive(
            b"x" * (1024 + oci._LEVEL3_STATUS_RESERVE_BYTES + 1),
            staging_root=tmp_path / "oversized",
            output_name="result",
            max_bytes=1024,
        )


def test_worker_command_bounds_real_timeout_and_output(tmp_path: Path) -> None:
    with pytest.raises(oci.OciEvaluationError, match="limit exceeded"):
        oci._run_bounded_command(
            [sys.executable, "-c", "print('x' * 100)"],
            timeout_seconds=10,
            stdout_limit=8,
        )
    with pytest.raises(oci.OciEvaluationError, match="timed out"):
        oci._run_bounded_command(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            timeout_seconds=1,
            stdout_limit=1024,
        )


def test_worker_mount_permissions_and_bounded_file_output(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(oci.OciEvaluationError, match="unavailable"):
        oci._artifact_mount_source(missing, label="missing")
    bad_path = tmp_path / "bad,source"
    bad_path.mkdir()
    with pytest.raises(oci.OciEvaluationError, match="represented"):
        oci._artifact_mount_source(bad_path, label="bad")
    link = tmp_path / "link"
    link.symlink_to(bad_path, target_is_directory=True)
    with pytest.raises(oci.OciEvaluationError, match="symlink"):
        oci._artifact_mount_source(link, label="link")

    unreadable = tmp_path / "unreadable"
    unreadable.write_text("secret", encoding="utf-8")
    unreadable.chmod(0)
    try:
        with pytest.raises(oci.OciEvaluationError, match="not readable"):
            oci._assert_worker_readable(
                unreadable, user="65532:65532", label="unreadable"
            )
    finally:
        unreadable.chmod(0o644)

    destination = tmp_path / "stdout.bin"
    completed = oci._run_bounded_command(
        [sys.executable, "-c", "print('bounded')"],
        timeout_seconds=10,
        stdout_limit=1024,
        stdout_path=destination,
    )
    assert completed.returncode == 0
    assert destination.read_bytes() == b"bounded\n"


def test_worker_process_and_container_handle_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ["docker", "run", "--cidfile", "/tmp/cid", "--name", "worker-name"]
    assert oci._worker_cidfile(command) == Path("/tmp/cid")
    assert oci._worker_container_name(command) == "worker-name"
    assert oci._worker_container_handle(command, None) == "worker-name"
    assert oci._worker_cidfile(["docker", "run"]) is None
    assert oci._worker_container_name(["docker", "run", "--name", "bad/name"]) is None


def test_worker_archive_path_and_stream_error_boundaries(tmp_path: Path) -> None:
    archive_path = tmp_path / "output.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        directory = tarfile.TarInfo("result")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        member = tarfile.TarInfo("result/value")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    staging = tmp_path / "staging"
    assert oci._extract_output_archive(
        archive_path, staging_root=staging, output_name="result", max_bytes=1024
    ).is_dir()

    with pytest.raises(oci.OciEvaluationError, match="unavailable"):
        oci._extract_output_archive(
            tmp_path / "missing.tar",
            staging_root=tmp_path / "missing-staging",
            output_name="result",
            max_bytes=1024,
        )
    link = tmp_path / "archive-link"
    link.symlink_to(archive_path)
    with pytest.raises(oci.OciEvaluationError, match="unsafe"):
        oci._extract_output_archive(
            link,
            staging_root=tmp_path / "link-staging",
            output_name="result",
            max_bytes=1024,
        )
    with pytest.raises(oci.OciEvaluationError, match="could not be extracted"):
        oci._extract_output_archive(
            b"not-a-tar",
            staging_root=tmp_path / "invalid-staging",
            output_name="result",
            max_bytes=1024,
        )

    oversized = io.BytesIO()
    with tarfile.open(fileobj=oversized, mode="w") as archive:
        directory = tarfile.TarInfo("result")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        member = tarfile.TarInfo("result/value")
        payload = b"x" * 2048
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(oci.OciEvaluationError, match="size limit"):
        oci._extract_output_archive(
            oversized.getvalue(),
            staging_root=tmp_path / "oversized-content",
            output_name="result",
            max_bytes=1024,
        )


def test_worker_container_controls_and_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_command(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(oci, "_run_bounded_command", fake_command)
    oci._container_control("docker", "stop", "worker")
    oci._container_control("docker", "kill", "worker")
    oci._remove_worker_container("docker", "worker")
    assert calls == [
        ["docker", "stop", "--time", "5", "worker"],
        ["docker", "kill", "worker"],
        ["docker", "rm", "--force", "--volumes", "worker"],
    ]

    def failing_command(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 4, b"", b"cleanup failed")

    monkeypatch.setattr(oci, "_run_bounded_command", failing_command)
    with pytest.raises(oci.OciEvaluationError, match="cleanup failed"):
        oci._remove_worker_container("docker", "worker")
    monkeypatch.setattr(
        oci,
        "_run_bounded_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            oci.OciEvaluationError("control")
        ),
    )
    oci._container_control("docker", "stop", "worker")


def test_worker_returns_engine_failures_and_rejects_missing_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "records.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "transaction" / "result"

    monkeypatch.setattr(
        oci,
        "run_side_worker",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 9, "", "engine failed"
        ),
    )
    monkeypatch.setattr(oci, "_worker_container_handle", lambda *_args: "worker")
    monkeypatch.setattr(
        oci,
        "_run_bounded_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""),
    )
    failed = oci.run_evaluator_worker(
        engine="docker",
        image=IMAGE,
        entrypoint=ENTRYPOINT,
        worker_arguments=("--output", "/outputs/result"),
        model_source=model,
        dataset_source=dataset,
        output=output,
        timeout_seconds=30,
    )
    assert failed.returncode == 9

    monkeypatch.setattr(
        oci,
        "run_side_worker",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    monkeypatch.setattr(oci, "_worker_container_handle", lambda *_args: None)
    with pytest.raises(oci.OciEvaluationError, match="container handle"):
        oci.run_evaluator_worker(
            engine="docker",
            image=IMAGE,
            entrypoint=ENTRYPOINT,
            worker_arguments=("--output", "/outputs/result"),
            model_source=model,
            dataset_source=dataset,
            output=output,
            timeout_seconds=30,
        )
