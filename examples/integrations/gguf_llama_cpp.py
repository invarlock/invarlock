#!/usr/bin/env python3
"""Compare an official Qwen3 Q8 GGUF with a pinned llama.cpp Q5 derivative."""

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
from invarlock.core.runtime_provider import artifact_identity_sha256
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    prepare_local_evaluation_schedule_bytes,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.runtime_providers.gguf_identity import read_gguf_artifact_identity

_MODEL_REPOSITORY = "Qwen/Qwen3-0.6B-GGUF"
_MODEL_REVISION = "23749fefcc72300e3a2ad315e1317431b06b590a"
_APT_SNAPSHOT = "20260701T000000Z"
_MAX_DOWNLOAD_BYTES = 700 * 1024 * 1024
_QUANTIZATION = "Q5_K_M"
_RECORDS = Path(__file__).with_name("gguf-llama-cpp") / "records.json"
_MINIMUM_SIDE_ACCURACY = 0.40
_PINNED_QWEN3_ONE_TOKEN_TARGET_IDS = {
    " Africa": 10174,
    " Asia": 13622,
    " Atlantic": 22375,
    " Berlin": 19846,
    " Cairo": 52550,
    " Canberra": 68790,
    " English": 6364,
    " Europe": 4505,
    " Everest": 86478,
    " Jupiter": 49689,
    " Lisbon": 80701,
    " Madrid": 24081,
    " Mars": 21048,
    " May": 3217,
    " Nairobi": 96525,
    " Nile": 76190,
    " Ottawa": 32166,
    " Pacific": 16462,
    " Paris": 12095,
    " Rome": 21718,
    " Tokyo": 26194,
    " blue": 6303,
    " book": 2311,
    " carbon": 12499,
    " child": 1682,
    " closed": 7877,
    " cold": 9255,
    " eight": 8063,
    " energy": 4802,
    " euro": 17672,
    " fifty": 32417,
    " four": 3040,
    " freezing": 42218,
    " gold": 6623,
    " gravity": 23249,
    " hundred": 7739,
    " night": 3729,
    " nine": 11627,
    " oxygen": 23552,
    " seven": 8094,
    " six": 4743,
    " slow": 6301,
    " small": 2613,
    " ten": 5779,
    " twelve": 29235,
    " vapor": 37652,
    " water": 3015,
    " yen": 57340,
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


_OFFICIAL_MODEL = ModelDownload(
    role="baseline",
    filename="Qwen3-0.6B-Q8_0.gguf",
    byte_length=639_446_688,
    sha256="9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
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
    subject = model_root / "Qwen3-0.6B-Q5_K_M.gguf"
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
    return _inspect_image_id(repository, container_engine=container_engine, image=image)


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
            or record["expected"] not in _PINNED_QWEN3_ONE_TOKEN_TARGET_IDS
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
) -> ExamplePaths:
    paths = _paths(root)
    (paths.evaluation / "inputs").mkdir(parents=True)
    paths.independent_policy.parent.mkdir(parents=True)
    paths.evidence_key.parent.mkdir(parents=True)
    paths.verifier_key.parent.mkdir(parents=True)

    records = _load_records()
    dataset_bytes = b"".join(canonical_json_bytes(record) for record in records)
    dataset = paths.evaluation / "inputs" / "records.jsonl"
    dataset.write_bytes(dataset_bytes)
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    schedule = prepare_local_evaluation_schedule_bytes(
        LocalDatasetRequest(
            path=dataset,
            sha256=dataset_sha256,
            name="qwen3-0.6b-q8-to-q5",
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
                    }
                }
            }
        }
    )
    request_policy = paths.evaluation / "inputs" / "acceptance.json"
    request_policy.write_bytes(policy_bytes)
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
                "name": "qwen3-0.6b-q8-to-q5",
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
    paths.trusted_inputs.write_bytes(
        canonical_json_bytes(
            {
                "format": "invarlock/trust-inputs-v1",
                "policy": {"path": "policy/acceptance.json"},
                "anchors": {
                    "baseline_artifact_digest": artifact_anchors["baseline"],
                    "subject_artifact_digest": artifact_anchors["subject"],
                    "schedule_digest": f"sha256:{schedule.schedule_sha256}",
                    "baseline_runtime_digest": image_id,
                    "subject_runtime_digest": image_id,
                    "evidence_signer_fingerprint": evidence_signer,
                },
                "verifier": {
                    "identity": "invarlock-example/gguf-llama-cpp-verifier",
                    "signing_key_path": "keys/verifier.pem",
                },
                "allow_installed_scorers": False,
            }
        )
    )
    if runtime_root != models["baseline"].parents[1]:
        raise RuntimeError("GGUF runtime resources do not share one closed root")
    return paths


def _execute(
    repository: Path,
    paths: ExamplePaths,
    *,
    runtime_root: Path,
    container_engine: str,
    image_id: str,
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
        trust_profile = json.loads(paths.trusted_inputs.read_text(encoding="utf-8"))
        anchors = trust_profile["anchors"]
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("GGUF preflight did not return a request identity") from exc
    if (
        not isinstance(request_digest, str)
        or not request_digest.startswith("sha256:")
        or len(request_digest) != 71
    ):
        raise RuntimeError("GGUF preflight returned an invalid request identity")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    if arguments.workspace is None:
        workspace = Path(tempfile.mkdtemp(prefix="invarlock-gguf-")).resolve(
            strict=True
        )
    else:
        workspace = arguments.workspace.expanduser().resolve()
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
        paths = _prepare_transaction(
            transaction,
            runtime_root=runtime_root,
            models=models,
            specs=specs,
            image_id=image_id,
        )
        _execute(
            repository,
            paths,
            runtime_root=runtime_root,
            container_engine=arguments.container_engine,
            image_id=image_id,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"Complete GGUF integration workspace: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
