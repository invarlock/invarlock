#!/usr/bin/env python3
"""Compare an official compact Q8 GGUF with a pinned llama.cpp Q5 derivative."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from examples.integrations import launch
from examples.integrations.run import ExamplePaths, _paths, _write_private_key
from examples.integrations.trust_material import (
    create_trust_material,
    load_external_key,
    validate_new_trust_root,
)
from invarlock.core.runtime_provider import artifact_identity_sha256
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evidence_pack_contract import canonical_json_bytes, normalize_digest
from invarlock.runtime_providers.gguf_identity import read_gguf_artifact_identity

_MODEL_REPOSITORY = "ggml-org/Qwen3.5-0.8B-GGUF"
_MODEL_REVISION = "8fea620810c4afa23dd6443f999a48574c1611a3"
_APT_SNAPSHOT = "20260701T000000Z"
_MAX_DOWNLOAD_BYTES = 900 * 1024 * 1024
_QUANTIZATION = "Q5_K_M"
_RECORDS = Path(__file__).with_name("gguf-llama-cpp") / "records.json"
_MINIMUM_SIDE_ACCURACY = 0.40
_PINNED_COMPACT_ONE_TOKEN_TARGET_IDS = {
    " Africa": 9871,
    " Asia": 13229,
    " Atlantic": 21678,
    " Berlin": 19241,
    " Cairo": 50779,
    " Canberra": 66463,
    " English": 6163,
    " Europe": 4357,
    " Everest": 83489,
    " Jupiter": 48017,
    " Lisbon": 77916,
    " Madrid": 23327,
    " Mars": 20403,
    " May": 3114,
    " Nairobi": 93190,
    " Nile": 73583,
    " Ottawa": 31106,
    " Pacific": 15979,
    " Paris": 11751,
    " Rome": 21047,
    " Tokyo": 25358,
    " blue": 6105,
    " book": 2236,
    " carbon": 12141,
    " child": 1623,
    " closed": 7629,
    " cold": 8981,
    " eight": 7810,
    " energy": 4649,
    " euro": 17146,
    " fifty": 31347,
    " four": 2943,
    " freezing": 40818,
    " gold": 6414,
    " gravity": 22525,
    " hundred": 7493,
    " night": 3603,
    " nine": 11292,
    " oxygen": 22817,
    " seven": 7840,
    " six": 4590,
    " slow": 6103,
    " small": 2526,
    " ten": 5600,
    " twelve": 28279,
    " vapor": 36405,
    " water": 2919,
    " yen": 55421,
}


@dataclass(frozen=True, slots=True)
class ModelDownload:
    role: str
    filename: str
    byte_length: int
    sha256: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{_MODEL_REPOSITORY}/resolve/"
            f"{_MODEL_REVISION}/{self.filename}"
        )


@dataclass(frozen=True, slots=True)
class PendingTrust:
    """Trust inputs waiting for the independently checked request identity."""

    anchors: Mapping[str, str]
    policy_bytes: bytes
    external: bool
    trust_root: Path | None
    verifier_key_bytes: bytes | None
    evidence_fingerprint: str
    verifier_fingerprint: str


_OFFICIAL_MODEL = ModelDownload(
    role="baseline",
    filename="Qwen3.5-0.8B-Q8_0.gguf",
    byte_length=833_592_096,
    sha256="37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f",
)


def _container_command(
    engine: str,
    image: str,
    *,
    user: str,
    entrypoint: str,
    mounts: tuple[str, ...] = (),
    environment: tuple[str, ...] = (),
) -> list[str]:
    command = [
        engine,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--user",
        user,
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=1g",
    ]
    for value in environment:
        command.extend(("--env", value))
    for value in mounts:
        command.extend(("--mount", value))
    return [*command, "--entrypoint", entrypoint, image]


def _mount_source(path: Path) -> str:
    rendered = str(path)
    if any(character in rendered for character in (",", "\n", "\r", "\x00")):
        raise ValueError("workspace path cannot be represented in an OCI mount")
    return rendered


def _copy_pinned_stream(source: object, destination: Path, spec: ModelDownload) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    byte_length = 0
    try:
        with temporary.open("xb") as output:
            while True:
                chunk = source.read(1024 * 1024)  # type: ignore[attr-defined]
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise RuntimeError("model download did not return bytes")
                byte_length += len(chunk)
                if byte_length > _MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("model download exceeds the byte limit")
                digest.update(chunk)
                output.write(chunk)
        if byte_length != spec.byte_length or digest.hexdigest() != spec.sha256:
            raise RuntimeError(
                f"downloaded {spec.role} GGUF does not match its pinned identity"
            )
        temporary.chmod(0o644)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _download_model(destination: Path, spec: ModelDownload) -> None:
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "invarlock-gguf-example/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            _copy_pinned_stream(response, destination, spec)
    except OSError as exc:
        raise RuntimeError(f"could not download the pinned {spec.role} GGUF") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_models(
    repository: Path,
    model_root: Path,
    *,
    container_engine: str,
    image_id: str,
) -> dict[str, Path]:
    model_root.mkdir(parents=True)
    baseline = model_root / _OFFICIAL_MODEL.filename
    _download_model(baseline, _OFFICIAL_MODEL)
    subject = model_root / "Qwen3.5-0.8B-Q5_K_M.gguf"
    uid = os.getuid() if hasattr(os, "getuid") else 65532
    gid = os.getgid() if hasattr(os, "getgid") else 65532
    launch._run(
        [
            *_container_command(
                container_engine,
                image_id,
                user=f"{uid}:{gid}",
                entrypoint="/opt/llama.cpp/llama-quantize",
                mounts=(
                    f"type=bind,src={_mount_source(baseline)},dst=/inputs/source.gguf,readonly",
                    f"type=bind,src={_mount_source(model_root)},dst=/output",
                ),
            ),
            "--allow-requantize",
            "/inputs/source.gguf",
            f"/output/{subject.name}",
            _QUANTIZATION,
        ],
        cwd=repository,
    )
    if not subject.is_file() or subject.stat().st_size <= 0:
        raise RuntimeError("pinned llama.cpp did not create the Q5 comparison artifact")
    subject.chmod(0o644)
    if _sha256_file(subject) == _OFFICIAL_MODEL.sha256:
        raise RuntimeError(
            "Q5 comparison artifact is identical to the official Q8 source"
        )
    return {"baseline": baseline, "subject": subject}


def _inspect_image_id(repository: Path, *, container_engine: str, image: str) -> str:
    completed = launch._run(
        [container_engine, "image", "inspect", "--format", "{{.Id}}", image],
        cwd=repository,
        capture_output=True,
    )
    image_id = completed.stdout.strip()
    if (
        not image_id.startswith("sha256:")
        or len(image_id) != 71
        or any(character not in "0123456789abcdef" for character in image_id[7:])
    ):
        raise RuntimeError("container inspection did not return a sha256 image ID")
    return image_id


def _build_runtime_image(
    repository: Path, build_root: Path, *, container_engine: str
) -> str:
    commit = launch._require_committed_checkout(repository)
    source_bundle = build_root / "source.tar"
    source = launch._run(
        [
            sys.executable,
            str(repository / "scripts/qualification_source.py"),
            "create",
            "--repository",
            str(repository),
            "--commit",
            commit,
            "--output",
            str(source_bundle),
        ],
        cwd=repository,
        capture_output=True,
    )
    source_digest = json.loads(source.stdout).get("source_bundle_sha256")
    if not isinstance(source_digest, str):
        raise RuntimeError("source-bundle creation did not return its digest")
    epoch = launch._git(repository, "show", "-s", "--format=%ct", commit)
    image = f"invarlock-example-gguf:{commit[:12]}"
    launch._run(
        [
            "make",
            "-C",
            "addins/gguf",
            "build",
            f"PYTHON={sys.executable}",
            f"CONTAINER_ENGINE={container_engine}",
            f"SOURCE_COMMIT={commit}",
            f"SOURCE_BUNDLE={source_bundle}",
            f"SOURCE_BUNDLE_SHA256={source_digest}",
            f"SOURCE_DATE_EPOCH={epoch}",
            f"LLAMA_CPP_APT_SNAPSHOT={_APT_SNAPSHOT}",
            "LLAMA_CPP_BUILD_JOBS=8",
            f"IMAGE={image}",
            f"BUILD_STATEMENT={build_root / 'runtime-build.json'}",
        ],
        cwd=repository,
    )
    image_id = launch._load_runtime_build_statement(
        build_root / "runtime-build.json",
        image=image,
        source_commit=commit,
        source_bundle_sha256=source_digest,
    )
    launch._verify_runtime_image_identity(
        repository=repository,
        container_engine=container_engine,
        runtime_image_id=image_id,
        source_commit=commit,
        source_bundle_sha256=source_digest,
    )
    return image_id


def _stage_backend(
    repository: Path,
    runtime_root: Path,
    *,
    container_engine: str,
    image_id: str,
) -> None:
    backend = runtime_root / "backend"
    backend.mkdir()
    uid = os.getuid() if hasattr(os, "getuid") else 65532
    gid = os.getgid() if hasattr(os, "getgid") else 65532
    launch._run(
        [
            *_container_command(
                container_engine,
                image_id,
                user=f"{uid}:{gid}",
                entrypoint="/bin/sh",
                mounts=(f"type=bind,src={_mount_source(backend)},dst=/output",),
            ),
            "-eu",
            "-c",
            (
                "cp /opt/llama.cpp/llama-completion /output/llama-completion; "
                "cp /opt/llama.cpp/source/llama.cpp-b10015.tar.gz "
                "/output/llama.cpp-b10015.tar.gz; "
                "chmod 0755 /output/llama-completion; "
                "chmod 0644 /output/llama.cpp-b10015.tar.gz"
            ),
        ],
        cwd=repository,
    )


def _inspect_spec(
    repository: Path,
    model: Path,
    *,
    container_engine: str,
    image_id: str,
) -> dict[str, object]:
    code = (
        "from pathlib import Path; import json; "
        "from invarlock_addins.gguf.provider import LlamaCppProvider; "
        "from invarlock_addins.gguf.session import LlamaCppRuntimeBindings; "
        "binding=LlamaCppRuntimeBindings(gguf_path=Path('/inputs/model.gguf'),"
        "executable_path=Path('/opt/llama.cpp/llama-completion'),"
        "source_archive_path=Path('/opt/llama.cpp/source/llama.cpp-b10015.tar.gz')); "
        "spec=LlamaCppProvider().inspect_runtime_spec(binding,seed=0,"
        "context_length=64,batch_size=1,max_output_tokens=1,timeout_seconds=30); "
        "print(json.dumps({'model_id':spec.model_id,'settings':dict(spec.settings)},"
        "sort_keys=True,separators=(',',':')))"
    )
    completed = launch._run(
        [
            *_container_command(
                container_engine,
                image_id,
                user="65532:65532",
                entrypoint="python",
                mounts=(
                    "type=bind,src="
                    f"{_mount_source(model)},dst=/inputs/model.gguf,readonly",
                ),
                environment=(
                    "HOME=/tmp",
                    "INVARLOCK_CONTAINER_EXECUTION=1",
                    f"INVARLOCK_RUNTIME_IMAGE={image_id}",
                    f"INVARLOCK_RUNTIME_IMAGE_DIGEST={image_id}",
                ),
            ),
            "-c",
            code,
        ],
        cwd=repository,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    if set(payload) != {"model_id", "settings"}:
        raise RuntimeError("GGUF inspection returned an unexpected payload")
    return payload


def _load_records() -> list[dict[str, str]]:
    payload = json.loads(_RECORDS.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 50:
        raise RuntimeError("the GGUF example must contain exactly 50 records")
    records: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for index, value in enumerate(payload):
        if not isinstance(value, dict) or set(value) != {"expected", "id", "prompt"}:
            raise RuntimeError(f"GGUF example record {index} has an invalid shape")
        if any(not isinstance(value[name], str) for name in value):
            raise RuntimeError(f"GGUF example record {index} must contain text")
        record = {name: value[name] for name in ("expected", "id", "prompt")}
        if not all(record.values()) or record["id"] in identifiers:
            raise RuntimeError(f"GGUF example record {index} is empty or duplicated")
        if (
            not record["expected"].startswith(" ")
            or record["expected"].strip() != record["expected"][1:]
            or any(character.isspace() for character in record["expected"][1:])
            or record["expected"] not in _PINNED_COMPACT_ONE_TOKEN_TARGET_IDS
        ):
            raise RuntimeError(
                f"GGUF example record {index} must use one maintained target word"
            )
        identifiers.add(record["id"])
        records.append(record)
    return records


def _prepare_transaction(
    root: Path,
    *,
    runtime_root: Path,
    models: Mapping[str, Path],
    specs: Mapping[str, Mapping[str, object]],
    image_id: str,
    evidence_signing_key: Path | None = None,
    verifier_signing_key: Path | None = None,
    trust_root: Path | None = None,
    ephemeral_trust_root: bool = True,
) -> tuple[ExamplePaths, PendingTrust]:
    external_trust = any(
        value is not None
        for value in (evidence_signing_key, verifier_signing_key, trust_root)
    )
    if external_trust and not all(
        value is not None
        for value in (evidence_signing_key, verifier_signing_key, trust_root)
    ):
        raise ValueError(
            "evidence key, verifier key, and trust root must be supplied together"
        )
    if external_trust and ephemeral_trust_root:
        raise ValueError("external trust material cannot use ephemeral mode")
    if not external_trust and not ephemeral_trust_root:
        raise ValueError(
            "caller-owned evidence/verifier keys and trust root are required"
        )
    paths = _paths(
        root,
        evidence_key=(
            Path(os.path.abspath(evidence_signing_key.expanduser()))
            if external_trust
            else None
        ),
        trust_root=(
            Path(os.path.abspath(trust_root.expanduser())) if external_trust else None
        ),
    )
    (paths.evaluation / "inputs").mkdir(parents=True)
    evidence_key_bytes: bytes | None = None
    verifier_key_bytes: bytes | None = None
    if external_trust:
        assert evidence_signing_key is not None
        assert verifier_signing_key is not None
        assert trust_root is not None
        trust_root = validate_new_trust_root(trust_root, transaction_root=root)
        evidence_key_path, evidence_key_bytes, evidence_signer = load_external_key(
            evidence_signing_key,
            transaction_root=root,
            label="evidence signing key",
        )
        _verifier_key_path, verifier_key_bytes, verifier = load_external_key(
            verifier_signing_key,
            transaction_root=root,
            label="verifier signing key",
        )
        if evidence_signer == verifier:
            raise ValueError("evidence and verifier signing keys must be distinct")
        paths = _paths(
            root,
            evidence_key=evidence_key_path,
            trust_root=Path(os.path.abspath(trust_root.expanduser()))
            if trust_root is not None
            else None,
        )
    else:
        paths.independent_policy.parent.mkdir(parents=True)
        paths.evidence_key.parent.mkdir(parents=True)
        paths.verifier_key.parent.mkdir(parents=True)
    paths.receipt.parent.mkdir(parents=True, exist_ok=True)

    records = _load_records()
    dataset_bytes = b"".join(canonical_json_bytes(record) for record in records)
    dataset = paths.evaluation / "inputs" / "records.jsonl"
    dataset.write_bytes(dataset_bytes)
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    schedule = prepare_local_evaluation_schedule_bytes(
        LocalDatasetRequest(
            path=dataset,
            sha256=dataset_sha256,
            name="qwen35-0.8b-q8-to-q5",
            split="validation",
            input_field="prompt",
            expected_output_field="expected",
            id_field="id",
        ),
        dataset_bytes,
    )

    policy_bytes = canonical_json_bytes(
        {
            "resolved_policy": {
                "metrics": {
                    "exact_match": {
                        "delta_min_pp": -15.0,
                        "maximum_interval_width_pp": 20.0,
                        "minimum_record_count": 50,
                        "minimum_side_accuracy": _MINIMUM_SIDE_ACCURACY,
                    }
                }
            }
        }
    )
    request_policy = paths.evaluation / "inputs" / "acceptance.json"
    request_policy.write_bytes(policy_bytes)
    if not external_trust:
        paths.independent_policy.write_bytes(policy_bytes)

    def side(role: str) -> dict[str, object]:
        spec = specs[role]
        model_id = spec.get("model_id")
        settings = spec.get("settings")
        if not isinstance(model_id, str) or not isinstance(settings, dict):
            raise RuntimeError(f"{role} GGUF inspection payload is invalid")
        if role == "baseline":
            locator = (
                f"hf://{_MODEL_REPOSITORY}@{_MODEL_REVISION}#{_OFFICIAL_MODEL.filename}"
            )
        else:
            locator = (
                f"derived://{_MODEL_REPOSITORY}@{_MODEL_REVISION}#"
                f"llama.cpp-b10015-{_QUANTIZATION.lower()}@sha256:"
                f"{_sha256_file(models[role])}"
            )
        return {
            "artifact": {
                "path": str(models[role].relative_to(paths.evaluation)),
                "model_id": model_id,
                "locator": locator,
            },
            "runtime": {"provider": "llama_cpp", "settings": settings},
        }

    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": side("baseline"),
            "subject": side("subject"),
            "dataset": {
                "path": "inputs/records.jsonl",
                "sha256": dataset_sha256,
                "format": "jsonl",
                "name": "qwen35-0.8b-q8-to-q5",
                "split": "validation",
                "input_field": "prompt",
                "expected_output_field": "expected",
                "id_field": "id",
            },
            "policy": "inputs/acceptance.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {"mode": "run"},
        "observations": [
            {
                "id": "qwen3-gguf-requantization",
                "kind": "artifact_transformation",
                "scope": "subject",
                "path": "inputs/subject-transformation.json",
            }
        ],
        "output": {"evidence": "evidence"},
    }
    transformation = {
        "format": "invarlock/example-gguf-transformation-v1",
        "source": {
            "repository": _MODEL_REPOSITORY,
            "revision": _MODEL_REVISION,
            "filename": _OFFICIAL_MODEL.filename,
            "sha256": _OFFICIAL_MODEL.sha256,
        },
        "tool": {
            "name": "llama.cpp",
            "source_commit": "12127defda4f41b7679cb2477a4b0d65ee6a0c8f",
            "source_tag": "b10015",
        },
        "quantization": _QUANTIZATION,
        "subject": {
            "filename": models["subject"].name,
            "sha256": _sha256_file(models["subject"]),
            "byte_length": models["subject"].stat().st_size,
        },
    }
    (paths.evaluation / "inputs" / "subject-transformation.json").write_bytes(
        canonical_json_bytes(transformation)
    )
    paths.request.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")

    if external_trust:
        assert evidence_key_bytes is not None
        assert verifier_key_bytes is not None
    else:
        evidence_signer = _write_private_key(paths.evidence_key)
        verifier = _write_private_key(paths.verifier_key)
        paths.evidence_key.with_suffix(".fingerprint").write_text(
            evidence_signer + "\n", encoding="ascii"
        )
        paths.verifier_key.with_suffix(".fingerprint").write_text(
            verifier + "\n", encoding="ascii"
        )
    artifact_anchors = {
        role: "sha256:" + artifact_identity_sha256(read_gguf_artifact_identity(model))
        for role, model in models.items()
    }
    anchors = {
        "baseline_artifact_digest": artifact_anchors["baseline"],
        "subject_artifact_digest": artifact_anchors["subject"],
        "schedule_digest": f"sha256:{schedule.schedule_sha256}",
        "baseline_runtime_digest": image_id,
        "subject_runtime_digest": image_id,
        "evidence_signer_fingerprint": evidence_signer,
    }
    if runtime_root != models["baseline"].parents[1]:
        raise RuntimeError("GGUF runtime resources do not share one closed root")
    return paths, PendingTrust(
        anchors=anchors,
        policy_bytes=policy_bytes,
        external=external_trust,
        trust_root=trust_root,
        verifier_key_bytes=verifier_key_bytes,
        evidence_fingerprint=evidence_signer,
        verifier_fingerprint=verifier,
    )


def _materialize_trust(
    paths: ExamplePaths, pending: PendingTrust, request_digest: str
) -> None:
    anchored_request = normalize_digest(
        request_digest, label="independent request anchor"
    )
    anchors = {**pending.anchors, "request_digest": anchored_request}
    if pending.external:
        if pending.trust_root is None or pending.verifier_key_bytes is None:
            raise RuntimeError("external GGUF trust material is incomplete")
        material = create_trust_material(
            transaction_root=paths.root,
            evidence_key=paths.evidence_key,
            verifier_key_bytes=pending.verifier_key_bytes,
            evidence_fingerprint=pending.evidence_fingerprint,
            verifier_fingerprint=pending.verifier_fingerprint,
            trust_root=pending.trust_root,
            policy_bytes=pending.policy_bytes,
            verifier_identity="invarlock-example/gguf-llama-cpp-verifier",
            anchors=anchors,
        )
        if material.trusted_inputs != paths.trusted_inputs:
            raise ValueError("external trust material resolved to an unexpected root")
        return
    paths.trusted_inputs.write_bytes(
        canonical_json_bytes(
            {
                "format": "invarlock/trust-inputs-v1",
                "policy": {"path": "policy/acceptance.json"},
                "anchors": anchors,
                "verifier": {
                    "identity": "invarlock-example/gguf-llama-cpp-verifier",
                    "signing_key_path": "keys/verifier.pem",
                },
                "allow_installed_scorers": False,
            }
        )
    )


def _execute(
    repository: Path,
    paths: ExamplePaths,
    *,
    runtime_root: Path,
    container_engine: str,
    image_id: str,
    pending_trust: PendingTrust | None = None,
) -> None:
    bindings = {
        "INVARLOCK_GGUF_RESOURCE_ROOT": str(runtime_root),
        "INVARLOCK_GGUF_BACKEND_EXECUTABLE": "backend/llama-completion",
        "INVARLOCK_GGUF_BACKEND_SOURCE": "backend/llama.cpp-b10015.tar.gz",
    }
    base = [sys.executable, "-m", "invarlock"]
    environment = dict(os.environ)
    environment.update(bindings)
    evaluation = [
        *base,
        "evaluate",
        str(paths.request),
        "--signing-key",
        str(paths.evidence_key),
        "--container-engine",
        container_engine,
        "--runtime-image",
        image_id,
        "--runtime-image-digest",
        image_id,
        "--runtime-device",
        "cpu",
        "--json",
    ]
    preflight = launch._run(
        [*evaluation, "--preflight"],
        cwd=repository,
        capture_output=True,
        environment=environment,
    )
    try:
        preflight_result = json.loads(preflight.stdout)
        request_digest = preflight_result["request_digest"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("GGUF preflight did not return a request identity") from exc
    if (
        not isinstance(request_digest, str)
        or not request_digest.startswith("sha256:")
        or len(request_digest) != 71
    ):
        raise RuntimeError("GGUF preflight returned an invalid request identity")
    if pending_trust is not None:
        _materialize_trust(paths, pending_trust, request_digest)
    else:
        try:
            trust_profile = json.loads(paths.trusted_inputs.read_text(encoding="utf-8"))
            anchors = trust_profile["anchors"]
            paths.trusted_inputs.relative_to(paths.root)
        except (KeyError, TypeError, json.JSONDecodeError, OSError, ValueError) as exc:
            raise RuntimeError("GGUF trust profile anchors are invalid") from exc
        if not isinstance(anchors, dict):
            raise RuntimeError("GGUF trust profile anchors are invalid")
        anchors["request_digest"] = request_digest
        paths.trusted_inputs.write_bytes(canonical_json_bytes(trust_profile))
    launch._run(evaluation, cwd=repository, environment=environment)
    launch._run(
        [
            *base,
            "verify",
            str(paths.evidence),
            "--trust-profile",
            str(paths.trusted_inputs),
            "--receipt",
            str(paths.receipt),
            "--json",
        ],
        cwd=repository,
    )
    launch._run(
        [
            *base,
            "report",
            str(paths.evidence),
            "--html",
            str(paths.html_report),
        ],
        cwd=repository,
    )
    try:
        report = json.loads(
            (paths.evidence / "reports" / "evaluation.report.json").read_text(
                encoding="utf-8"
            )
        )
        receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "GGUF comparison is missing verified transaction outputs"
        ) from exc
    if not isinstance(report, dict) or not isinstance(receipt, dict):
        raise RuntimeError("GGUF comparison returned invalid transaction outputs")
    comparison = report.get("comparison")
    comparison_value = comparison.get("value") if isinstance(comparison, dict) else None
    statement = receipt.get("statement")
    receipt_verdict = statement.get("verdict") if isinstance(statement, dict) else None
    if (
        report.get("verdict") != "pass"
        or report.get("metric") != "exact_match"
        or isinstance(comparison_value, bool)
        or not isinstance(comparison_value, (int, float))
        or not isinstance(receipt_verdict, dict)
        or receipt_verdict.get("ok") is not True
        or receipt_verdict.get("integrity_ok") is not True
        or receipt_verdict.get("policy_verdict") != "pass"
        or not paths.html_report.is_file()
        or paths.html_report.stat().st_size == 0
    ):
        raise RuntimeError("GGUF comparison did not produce verified passing evidence")
    for role in ("baseline", "subject"):
        side = report.get(role)
        mean_score = side.get("mean_score") if isinstance(side, dict) else None
        if (
            isinstance(mean_score, bool)
            or not isinstance(mean_score, (int, float))
            or mean_score < _MINIMUM_SIDE_ACCURACY
        ):
            raise RuntimeError(
                f"the {role} GGUF model solved fewer than 40% of the maintained "
                "causal-cloze records"
            )
    print(f"PASS subject exact-match delta: {comparison_value:.2f} percentage points")
    print(f"Evidence: {paths.evidence}")
    print(f"Receipt: {paths.receipt}")
    print(f"Report: {paths.html_report}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--container-engine", choices=("docker", "podman"), default="docker"
    )
    parser.add_argument(
        "--runtime-image",
        help="Reuse a locally available GGUF runtime image; default builds current source.",
    )
    parser.add_argument("--evidence-signing-key", type=Path)
    parser.add_argument("--verifier-signing-key", type=Path)
    parser.add_argument("--trust-root", type=Path)
    parser.add_argument(
        "--ephemeral-trust-root",
        action="store_true",
        help="Use disposable generated keys; never use this mode for acceptance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    trust_values = (
        arguments.evidence_signing_key,
        arguments.verifier_signing_key,
        arguments.trust_root,
    )
    provided_trust = any(value is not None for value in trust_values)
    external_trust = all(value is not None for value in trust_values)
    if provided_trust and not external_trust:
        print(
            "FAIL --evidence-signing-key, --verifier-signing-key, and "
            "--trust-root must be supplied together",
            file=sys.stderr,
        )
        return 2
    if not external_trust and not arguments.ephemeral_trust_root:
        print(
            "FAIL caller-owned --evidence-signing-key, --verifier-signing-key, "
            "and --trust-root are required; use --ephemeral-trust-root only for "
            "a disposable non-acceptance demo",
            file=sys.stderr,
        )
        return 2
    if external_trust and arguments.ephemeral_trust_root:
        print(
            "FAIL --ephemeral-trust-root cannot be combined with caller-owned trust",
            file=sys.stderr,
        )
        return 2
    repository = Path(__file__).resolve().parents[2]
    if arguments.workspace is None:
        workspace = Path(tempfile.mkdtemp(prefix="invarlock-gguf-")).resolve(
            strict=True
        )
    else:
        workspace = Path(os.path.abspath(arguments.workspace.expanduser()))
        if workspace.exists() or workspace.is_symlink():
            print(f"FAIL workspace already exists: {workspace}", file=sys.stderr)
            return 2
        workspace.parent.mkdir(parents=True, exist_ok=True)
        workspace.mkdir()
    try:
        build_root = workspace / "build"
        build_root.mkdir()
        if arguments.runtime_image is None:
            image_id = _build_runtime_image(
                repository,
                build_root,
                container_engine=arguments.container_engine,
            )
        else:
            image_id = _inspect_image_id(
                repository,
                container_engine=arguments.container_engine,
                image=arguments.runtime_image,
            )
        transaction = workspace / "transaction"
        transaction.mkdir()
        runtime_root = transaction / "evaluation" / "runtime"
        runtime_root.mkdir(parents=True)
        models = _stage_models(
            repository,
            runtime_root / "models",
            container_engine=arguments.container_engine,
            image_id=image_id,
        )
        _stage_backend(
            repository,
            runtime_root,
            container_engine=arguments.container_engine,
            image_id=image_id,
        )
        specs = {
            role: _inspect_spec(
                repository,
                model,
                container_engine=arguments.container_engine,
                image_id=image_id,
            )
            for role, model in models.items()
        }
        paths, pending_trust = _prepare_transaction(
            transaction,
            runtime_root=runtime_root,
            models=models,
            specs=specs,
            image_id=image_id,
            evidence_signing_key=arguments.evidence_signing_key,
            verifier_signing_key=arguments.verifier_signing_key,
            trust_root=arguments.trust_root,
            ephemeral_trust_root=arguments.ephemeral_trust_root,
        )
        _execute(
            repository,
            paths,
            runtime_root=runtime_root,
            container_engine=arguments.container_engine,
            image_id=image_id,
            pending_trust=pending_trust,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"Complete GGUF integration workspace: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
