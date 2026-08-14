from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from examples.integrations import gguf_llama_cpp as example
from invarlock.core.runtime_provider import GGUFArtifactIdentity


def _completed(
    command: list[str], stdout: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_records_are_closed_unique_and_sufficient() -> None:
    records = example._load_records()

    assert len(records) == 50
    assert len({record["id"] for record in records}) == 50
    assert all(record["prompt"] and record["expected"] for record in records)
    assert all(record["id"].startswith("qwen3-") for record in records)
    assert all(record["expected"].startswith(" ") for record in records)
    assert all(
        not any(character.isspace() for character in record["expected"][1:])
        for record in records
    )
    assert {record["expected"] for record in records} == set(
        example._PINNED_COMPACT_ONE_TOKEN_TARGET_IDS
    )
    assert all(
        isinstance(token_id, int) and token_id > 0
        for token_id in example._PINNED_COMPACT_ONE_TOKEN_TARGET_IDS.values()
    )


def test_records_reject_unreviewed_target_token_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = json.loads(example._RECORDS.read_text(encoding="utf-8"))
    records[0]["expected"] = " tokenizer"
    changed = tmp_path / "records.json"
    changed.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(example, "_RECORDS", changed)

    with pytest.raises(RuntimeError, match="maintained target word"):
        example._load_records()


def test_official_qwen35_08b_gguf_identity() -> None:
    assert example._MODEL_REPOSITORY == "ggml-org/Qwen3.5-0.8B-GGUF"
    assert example._MODEL_REVISION == "8fea620810c4afa23dd6443f999a48574c1611a3"
    assert example._OFFICIAL_MODEL.filename == "Qwen3.5-0.8B-Q8_0.gguf"
    assert example._OFFICIAL_MODEL.byte_length == 833_592_096
    assert example._OFFICIAL_MODEL.sha256 == (
        "37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f"
    )
    dockerfile = (
        Path(__file__).resolve().parents[2] / "addins/gguf/runtime/Dockerfile"
    ).read_text(encoding="utf-8")
    assert "--target llama-completion llama-quantize" in dockerfile
    assert "COPY --from=llama-cpp-build /opt/llama.cpp/llama-quantize" in dockerfile


def test_pinned_download_accepts_exact_bytes_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"GGUF fixture bytes"
    spec = example.ModelDownload(
        role="baseline",
        filename="model.gguf",
        byte_length=len(payload),
        sha256=example.hashlib.sha256(payload).hexdigest(),
    )
    destination = tmp_path / "model.gguf"

    example._copy_pinned_stream(io.BytesIO(payload), destination, spec)
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o644

    bad = example.ModelDownload(
        role="subject",
        filename="bad.gguf",
        byte_length=len(payload),
        sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="pinned identity"):
        example._copy_pinned_stream(io.BytesIO(payload), tmp_path / "bad.gguf", bad)
    assert not (tmp_path / "bad.gguf.partial").exists()

    monkeypatch.setattr(example, "_MAX_DOWNLOAD_BYTES", 4)
    with pytest.raises(RuntimeError, match="byte limit"):
        example._copy_pinned_stream(io.BytesIO(payload), tmp_path / "large.gguf", spec)
    assert not (tmp_path / "large.gguf.partial").exists()


def test_download_failure_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise OSError("offline")

    monkeypatch.setattr(example.urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="could not download the pinned baseline"):
        example._download_model(tmp_path / "model.gguf", example._OFFICIAL_MODEL)


def test_stage_models_derives_q5_with_the_pinned_networkless_quantizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_download(destination: Path, spec: example.ModelDownload) -> None:
        destination.write_bytes(b"official-q8")

    monkeypatch.setattr(example, "_download_model", fake_download)
    monkeypatch.setattr(
        example.launch,
        "_run",
        lambda command, **_kwargs: (
            commands.append(command),
            (tmp_path / "models/Qwen3.5-0.8B-Q5_K_M.gguf").write_bytes(
                b"derived-q5"
            ),
            _completed(command),
        )[-1],
    )
    models = example._stage_models(
        tmp_path,
        tmp_path / "models",
        container_engine="docker",
        image_id="sha256:" + "a" * 64,
    )

    assert models["baseline"].read_bytes() != models["subject"].read_bytes()
    command = commands[0]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--entrypoint") + 1] == (
        "/opt/llama.cpp/llama-quantize"
    )
    assert "--allow-requantize" in command
    assert command[-1] == "Q5_K_M"


def test_container_command_closes_the_native_runtime_boundary() -> None:
    command = example._container_command(
        "docker",
        "sha256:" + "a" * 64,
        user="65532:65532",
        entrypoint="python",
        mounts=("type=bind,src=/model,dst=/inputs/model,readonly",),
        environment=("INVARLOCK_CONTAINER_EXECUTION=1",),
    )

    assert command[:3] == ["docker", "run", "--rm"]
    assert ["--network", "none"] == command[3:5]
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--user") + 1] == "65532:65532"
    assert command[-3:] == ["--entrypoint", "python", "sha256:" + "a" * 64]

    with pytest.raises(ValueError, match="OCI mount"):
        example._mount_source(Path("bad,path"))


def test_image_inspection_and_runtime_build_use_immutable_current_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    commands: list[list[str]] = []
    image_id = "sha256:" + "d" * 64
    monkeypatch.setattr(
        example.launch, "_require_committed_checkout", lambda _root: "c" * 40
    )
    monkeypatch.setattr(example.launch, "_git", lambda *_args: "1234567890")

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "qualification_source.py" in " ".join(command):
            return _completed(command, json.dumps({"source_bundle_sha256": image_id}))
        if command[:3] == ["make", "-C", "addins/gguf"]:
            statement_path = next(
                Path(value.split("=", 1)[1])
                for value in command
                if value.startswith("BUILD_STATEMENT=")
            )
            statement_path.write_text(
                json.dumps(
                    {
                        "base_image": None,
                        "build_arguments": {},
                        "dockerfile": {
                            "path": "runtime/Dockerfile",
                            "sha256": image_id,
                        },
                        "format_version": "invarlock/runtime-image-build-v1",
                        "image": "invarlock-example-gguf:" + "c" * 12,
                        "ok": True,
                        "platform": None,
                        "runtime_image_id": image_id,
                        "source_bundle_sha256": image_id,
                        "source_commit": "c" * 40,
                    }
                ),
                encoding="utf-8",
            )
            return _completed(command)
        if command[1:3] == ["image", "inspect"]:
            return _completed(
                command, image_id + "\t" + ("c" * 40) + "\t" + image_id + "\n"
            )
        return _completed(command)

    monkeypatch.setattr(example.launch, "_run", fake_run)
    assert (
        example._build_runtime_image(repository, build, container_engine="docker")
        == image_id
    )
    make = next(
        command for command in commands if command[:3] == ["make", "-C", "addins/gguf"]
    )
    assert "SOURCE_COMMIT=" + "c" * 40 in make
    assert "LLAMA_CPP_APT_SNAPSHOT=20260701T000000Z" in make
    assert any(value.startswith("BUILD_STATEMENT=") for value in make)

    monkeypatch.setattr(
        example.launch,
        "_run",
        lambda *args, **kwargs: _completed(args[0], "mutable-tag\n"),
    )
    with pytest.raises(RuntimeError, match="sha256 image ID"):
        example._inspect_image_id(repository, container_engine="docker", image="bad")


def test_inspection_runs_in_the_authenticated_networkless_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    image_id = "sha256:" + "e" * 64
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _completed(command, '{"model_id":"gguf-sha256-x.gguf","settings":{}}')

    monkeypatch.setattr(example.launch, "_run", fake_run)
    assert example._inspect_spec(
        tmp_path, model, container_engine="docker", image_id=image_id
    ) == {"model_id": "gguf-sha256-x.gguf", "settings": {}}
    command = commands[0]
    assert command[command.index("--network") + 1] == "none"
    assert f"INVARLOCK_RUNTIME_IMAGE_DIGEST={image_id}" in command
    assert command[command.index("--entrypoint") + 1] == "python"

    monkeypatch.setattr(
        example.launch,
        "_run",
        lambda *args, **kwargs: _completed(args[0], '{"unexpected":true}'),
    )
    with pytest.raises(RuntimeError, match="unexpected payload"):
        example._inspect_spec(
            tmp_path, model, container_engine="docker", image_id=image_id
        )


def test_transaction_binds_distinct_gguf_artifacts_schedule_policy_and_signers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "transaction"
    runtime = root / "evaluation" / "runtime"
    models = {
        role: runtime / "models" / f"{role}.gguf" for role in ("baseline", "subject")
    }
    for role, path in models.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode("ascii"))

    def identity(path: Path) -> GGUFArtifactIdentity:
        digest = "a" * 64 if path == models["baseline"] else "b" * 64
        return GGUFArtifactIdentity(
            artifact_name=f"gguf-sha256-{digest}.gguf",
            sha256=digest,
            byte_length=8,
            gguf_metadata_sha256="c" * 64,
            tensor_inventory_sha256="d" * 64,
            tokenizer_metadata_sha256="e" * 64,
        )

    monkeypatch.setattr(example, "read_gguf_artifact_identity", identity)
    specs = {
        role: {"model_id": identity(path).artifact_name, "settings": {}}
        for role, path in models.items()
    }
    image_id = "sha256:" + "f" * 64
    paths = example._prepare_transaction(
        root,
        runtime_root=runtime,
        models=models,
        specs=specs,
        image_id=image_id,
    )

    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    trust = json.loads(paths.trusted_inputs.read_text(encoding="utf-8"))
    assert request["comparison"]["metric"] == "exact_match"
    assert request["comparison"]["baseline"]["runtime"]["provider"] == "llama_cpp"
    assert request["comparison"]["subject"]["runtime"]["provider"] == "llama_cpp"
    assert request["comparison"]["dataset"]["name"] == "qwen3-0.6b-q8-to-q5"
    assert request["comparison"]["baseline"]["artifact"]["locator"].startswith(
        "hf://ggml-org/Qwen3.5-0.8B-GGUF@8fea6208"
    )
    assert request["comparison"]["subject"]["artifact"]["locator"].startswith(
        "derived://ggml-org/Qwen3.5-0.8B-GGUF@8fea6208"
    )
    assert request["observations"][0]["path"] == ("inputs/subject-transformation.json")
    transformation = json.loads(
        (paths.evaluation / "inputs/subject-transformation.json").read_text()
    )
    assert transformation["source"]["sha256"] == example._OFFICIAL_MODEL.sha256
    assert transformation["quantization"] == "Q5_K_M"
    policy = json.loads(paths.independent_policy.read_text(encoding="utf-8"))
    assert policy["resolved_policy"]["metrics"]["exact_match"] == {
        "delta_min_pp": -15.0,
        "maximum_interval_width_pp": 20.0,
        "minimum_record_count": 50,
        "minimum_side_accuracy": example._MINIMUM_SIDE_ACCURACY,
    }
    assert (
        trust["anchors"]["baseline_artifact_digest"]
        != trust["anchors"]["subject_artifact_digest"]
    )
    assert trust["anchors"]["baseline_runtime_digest"] == image_id
    assert paths.evidence_key.stat().st_mode & 0o777 == 0o600
    assert paths.verifier_key.stat().st_mode & 0o777 == 0o600


def test_execute_uses_public_commands_and_caller_owned_backend_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = example._paths(tmp_path / "transaction")
    report = paths.evidence / "reports" / "evaluation.report.json"
    report.parent.mkdir(parents=True)
    paths.trusted_inputs.parent.mkdir(parents=True, exist_ok=True)
    paths.trusted_inputs.write_text(json.dumps({"anchors": {}}), encoding="utf-8")
    report.write_text(
        json.dumps(
            {
                "baseline": {"mean_score": 0.5},
                "comparison": {"value": 0.0},
                "metric": "exact_match",
                "subject": {"mean_score": 0.5},
                "verdict": "pass",
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    observed_binding: dict[str, str] = {}

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool = False,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "evaluate" in command:
            assert environment is not None
            observed_binding["root"] = environment["INVARLOCK_GGUF_RESOURCE_ROOT"]
            observed_binding["backend"] = environment[
                "INVARLOCK_GGUF_BACKEND_EXECUTABLE"
            ]
            if "--preflight" in command:
                return _completed(
                    command,
                    json.dumps({"request_digest": "sha256:" + "1" * 64}),
                )
        if "verify" in command:
            paths.receipt.parent.mkdir(parents=True, exist_ok=True)
            paths.receipt.write_text(
                json.dumps(
                    {
                        "statement": {
                            "verdict": {
                                "integrity_ok": True,
                                "ok": True,
                                "policy_verdict": "pass",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
        if "report" in command:
            paths.html_report.write_text("<html></html>", encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr(example.launch, "_run", fake_run)
    image_id = "sha256:" + "f" * 64
    runtime = tmp_path / "runtime"
    example._execute(
        tmp_path,
        paths,
        runtime_root=runtime,
        container_engine="docker",
        image_id=image_id,
    )

    assert [command[3] for command in commands] == [
        "evaluate",
        "evaluate",
        "verify",
        "report",
    ]
    assert "--preflight" in commands[0]
    assert "--preflight" not in commands[1]
    assert observed_binding == {
        "root": str(runtime),
        "backend": "backend/llama-completion",
    }
    assert "--trust-profile" in commands[2]
    assert "--html" in commands[3]

    report.write_text(
        json.dumps(
            {"comparison": {"value": 0.0}, "metric": "exact_match", "verdict": "fail"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="verified passing evidence"):
        example._execute(
            tmp_path,
            paths,
            runtime_root=runtime,
            container_engine="docker",
            image_id=image_id,
        )

    report.write_text(
        json.dumps(
            {
                "baseline": {"mean_score": 0.39},
                "comparison": {"value": 0.0},
                "metric": "exact_match",
                "subject": {"mean_score": 0.5},
                "verdict": "pass",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="fewer than 40%"):
        example._execute(
            tmp_path,
            paths,
            runtime_root=runtime,
            container_engine="docker",
            image_id=image_id,
        )


@pytest.mark.parametrize(
    ("report_payload", "receipt_payload", "error"),
    [
        (
            {
                "baseline": {"mean_score": 0.5},
                "comparison": {"value": 0.0},
                "metric": "exact_match",
                "subject": {"mean_score": 0.5},
                "verdict": "pass",
            },
            {},
            "verified passing evidence",
        ),
        (
            {
                "baseline": {"mean_score": 0.5},
                "comparison": {"value": True},
                "metric": "exact_match",
                "subject": {"mean_score": 0.5},
                "verdict": "pass",
            },
            {
                "statement": {
                    "verdict": {
                        "integrity_ok": True,
                        "ok": True,
                        "policy_verdict": "pass",
                    }
                }
            },
            "verified passing evidence",
        ),
        (
            {
                "baseline": {"mean_score": True},
                "comparison": {"value": 0.0},
                "metric": "exact_match",
                "subject": {"mean_score": 0.5},
                "verdict": "pass",
            },
            {
                "statement": {
                    "verdict": {
                        "integrity_ok": True,
                        "ok": True,
                        "policy_verdict": "pass",
                    }
                }
            },
            "fewer than 40%",
        ),
    ],
)
def test_execute_rejects_false_green_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_payload: dict[str, object],
    receipt_payload: dict[str, object],
    error: str,
) -> None:
    paths = example._paths(tmp_path / "transaction")
    report = paths.evidence / "reports" / "evaluation.report.json"
    report.parent.mkdir(parents=True)
    paths.trusted_inputs.parent.mkdir(parents=True, exist_ok=True)
    paths.trusted_inputs.write_text(json.dumps({"anchors": {}}), encoding="utf-8")
    report.write_text(json.dumps(report_payload), encoding="utf-8")

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "evaluate" in command and "--preflight" in command:
            return _completed(
                command,
                json.dumps({"request_digest": "sha256:" + "1" * 64}),
            )
        if "verify" in command:
            paths.receipt.parent.mkdir(parents=True, exist_ok=True)
            paths.receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
        if "report" in command:
            paths.html_report.write_text("<html></html>", encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr(example.launch, "_run", fake_run)
    with pytest.raises(RuntimeError, match=error):
        example._execute(
            tmp_path,
            paths,
            runtime_root=tmp_path / "runtime",
            container_engine="docker",
            image_id="sha256:" + "f" * 64,
        )


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_main_rejects_existing_workspace(tmp_path: Path, kind: str) -> None:
    existing = tmp_path / "existing"
    missing = tmp_path / "missing-workspace"
    if kind == "directory":
        existing.mkdir()
    else:
        existing.symlink_to(missing, target_is_directory=True)
    assert example.main(["--workspace", str(existing), "--ephemeral-trust-root"]) == 2
    assert not missing.exists()


def test_default_workspace_is_canonical_before_source_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    observed: list[Path] = []
    monkeypatch.setattr(example.tempfile, "mkdtemp", lambda **_kwargs: str(alias))

    def stop(_repository: Path, build_root: Path, *, container_engine: str) -> str:
        observed.append(build_root)
        raise RuntimeError("stop after canonical workspace check")

    monkeypatch.setattr(example, "_build_runtime_image", stop)
    assert example.main(["--ephemeral-trust-root"]) == 2
    assert observed == [real.resolve() / "build"]


def test_main_reuses_an_immutable_image_and_completes_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    digest = "sha256:" + "a" * 64
    observed: list[str] = []
    monkeypatch.setattr(
        example,
        "_inspect_image_id",
        lambda *_args, **_kwargs: observed.append("image") or digest,
    )

    def stage_models(
        _repository: Path,
        model_root: Path,
        **_kwargs: object,
    ) -> dict[str, Path]:
        model_root.mkdir(parents=True)
        paths = {role: model_root / f"{role}.gguf" for role in ("baseline", "subject")}
        for role, path in paths.items():
            path.write_bytes(role.encode())
        observed.append("models")
        return paths

    monkeypatch.setattr(example, "_stage_models", stage_models)
    monkeypatch.setattr(
        example,
        "_stage_backend",
        lambda *_args, **_kwargs: observed.append("backend"),
    )
    monkeypatch.setattr(
        example,
        "_inspect_spec",
        lambda _repository, model, **_kwargs: {
            "model_id": model.stem,
            "settings": {},
        },
    )
    paths = object()
    monkeypatch.setattr(
        example,
        "_prepare_transaction",
        lambda *_args, **_kwargs: observed.append("prepare") or paths,
    )
    monkeypatch.setattr(
        example,
        "_execute",
        lambda *_args, **_kwargs: observed.append("execute"),
    )

    assert (
        example.main(
            [
                "--workspace",
                str(workspace),
                "--runtime-image",
                digest,
                "--ephemeral-trust-root",
            ]
        )
        == 0
    )
    assert observed == ["image", "models", "backend", "prepare", "execute"]


def test_main_reports_runtime_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        example,
        "_build_runtime_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("build failed")),
    )
    assert (
        example.main(
            [
                "--workspace",
                str(tmp_path / "workspace"),
                "--ephemeral-trust-root",
            ]
        )
        == 2
    )
    assert "build failed" in capsys.readouterr().err


def test_pinned_download_rejects_non_byte_streams(tmp_path: Path) -> None:
    class TextStream:
        def read(self, _size: int) -> str:
            return "not bytes"

    destination = tmp_path / "model.gguf"
    with pytest.raises(RuntimeError, match="did not return bytes"):
        example._copy_pinned_stream(TextStream(), destination, example._OFFICIAL_MODEL)
    assert not destination.with_suffix(".gguf.partial").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("count", "exactly 50 records"),
        ("shape", "invalid shape"),
        ("text", "must contain text"),
        ("duplicate", "empty or duplicated"),
    ],
)
def test_records_reject_malformed_or_duplicate_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    records = json.loads(example._RECORDS.read_text(encoding="utf-8"))
    if mutation == "count":
        records.pop()
    elif mutation == "shape":
        records[0] = {"id": "missing-fields"}
    elif mutation == "text":
        records[0]["prompt"] = 1
    else:
        records[0]["id"] = records[1]["id"]
    changed = tmp_path / f"records-{mutation}.json"
    changed.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(example, "_RECORDS", changed)

    with pytest.raises(RuntimeError, match=message):
        example._load_records()


@pytest.mark.parametrize("artifact", ["missing", "identical"])
def test_quantization_requires_a_distinct_created_subject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    payload = b"official"
    official = example.ModelDownload(
        role="baseline",
        filename="official.gguf",
        byte_length=len(payload),
        sha256=example.hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(example, "_OFFICIAL_MODEL", official)
    monkeypatch.setattr(
        example,
        "_download_model",
        lambda destination, _spec: destination.write_bytes(payload),
    )

    def quantize(command: list[str], **_kwargs: object) -> object:
        if artifact == "identical":
            (tmp_path / "models/Qwen3.5-0.8B-Q5_K_M.gguf").write_bytes(payload)
        return _completed(command)

    monkeypatch.setattr(example.launch, "_run", quantize)
    message = "did not create" if artifact == "missing" else "identical"
    with pytest.raises(RuntimeError, match=message):
        example._stage_models(
            tmp_path,
            tmp_path / "models",
            container_engine="docker",
            image_id="sha256:" + "a" * 64,
        )


def test_runtime_build_requires_source_bundle_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        example.launch, "_require_committed_checkout", lambda _root: "c" * 40
    )
    monkeypatch.setattr(
        example.launch,
        "_run",
        lambda command, **_kwargs: _completed(command, "{}"),
    )
    with pytest.raises(RuntimeError, match="did not return its digest"):
        example._build_runtime_image(
            tmp_path, tmp_path / "build", container_engine="docker"
        )


def test_execute_rejects_non_object_transaction_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = example._paths(tmp_path / "transaction")
    report = paths.evidence / "reports/evaluation.report.json"
    report.parent.mkdir(parents=True)
    report.write_text("[]", encoding="utf-8")
    paths.receipt.parent.mkdir(parents=True)
    paths.receipt.write_text("{}", encoding="utf-8")
    paths.html_report.write_text("<html></html>", encoding="utf-8")
    paths.trusted_inputs.write_text(json.dumps({"anchors": {}}), encoding="utf-8")

    def complete(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "evaluate" in command and "--preflight" in command:
            return _completed(
                command,
                json.dumps({"request_digest": "sha256:" + "1" * 64}),
            )
        return _completed(command)

    monkeypatch.setattr(
        example.launch,
        "_run",
        complete,
    )

    with pytest.raises(RuntimeError, match="invalid transaction outputs"):
        example._execute(
            tmp_path,
            paths,
            runtime_root=tmp_path / "runtime",
            container_engine="docker",
            image_id="sha256:" + "a" * 64,
        )
