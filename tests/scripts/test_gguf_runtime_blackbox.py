from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path.cwd() / "scripts" / "release" / "gguf_runtime_blackbox.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gguf_runtime_blackbox", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


blackbox = _load_script()


def _result_payload(*, image_digest: str, output: str | None = None) -> bytes:
    observed_output = blackbox.EXPECTED_OUTPUT if output is None else output
    value = {
        "cli_journey": {
            "artifact_identity_sha256": blackbox.ARTIFACT_IDENTITY_SHA256,
            "binding_sha256": "1" * 64,
            "execution_settings_sha256": blackbox.CLI_EXECUTION_SETTINGS_SHA256,
            "format_version": blackbox.CLI_JOURNEY_FORMAT,
            "observation": {},
            "observation_sha256": blackbox.CLI_SCORING_OBSERVATION_SHA256,
            "policy_digest": "sha256:" + "2" * 64,
            "policy_file_sha256": "3" * 64,
            "portable_artifact_count": 17,
            "provider_receipt": {},
            "provider_receipt_sha256": "4" * 64,
            "schedule_sha256": blackbox.CLI_SCHEDULE_SHA256,
            "verification": {
                "baseline_score": 1.0,
                "regression": 0.0,
                "subject_score": 1.0,
                "verdict": "pass",
            },
        },
        "fixture": {
            "byte_length": blackbox.FIXTURE_BYTE_LENGTH,
            "repository": blackbox.FIXTURE_REPOSITORY,
            "revision": blackbox.FIXTURE_REVISION,
            "sha256": blackbox.FIXTURE_SHA256,
        },
        "format_version": blackbox.RESULT_FORMAT,
        "image_digest": image_digest,
        "observation": {
            "aggregate_source_sha256": blackbox.AGGREGATE_SOURCE_SHA256,
            "artifact_identity_sha256": blackbox.ARTIFACT_IDENTITY_SHA256,
            "format_version": "invarlock/runtime-scoring-observation-v1",
            "provider_name": "llama_cpp",
            "records": [
                {
                    "error_code": None,
                    "input_sha256": hashlib.sha256(
                        blackbox.PROMPT.encode("utf-8")
                    ).hexdigest(),
                    "logprob_sum": None,
                    "output_sha256": hashlib.sha256(
                        observed_output.encode("utf-8")
                    ).hexdigest(),
                    "output_text": observed_output,
                    "record_id": blackbox.RECORD_ID,
                    "status": "ok",
                    "token_count": None,
                    "utf8_byte_count": None,
                }
            ],
            "schedule_sha256": blackbox.SCHEDULE_SHA256,
        },
        "receipt": {
            "artifact_identity": {
                "artifact_format": "gguf",
                "artifact_name": f"gguf-sha256-{blackbox.FIXTURE_SHA256}.gguf",
                "byte_length": blackbox.FIXTURE_BYTE_LENGTH,
                "format_version": "invarlock/model-artifact-identity-v1",
                "gguf_metadata_sha256": blackbox.FIXTURE_METADATA_SHA256,
                "sha256": blackbox.FIXTURE_SHA256,
                "tensor_inventory_sha256": (blackbox.FIXTURE_TENSOR_INVENTORY_SHA256),
                "tokenizer_metadata_sha256": (
                    blackbox.FIXTURE_TOKENIZER_METADATA_SHA256
                ),
            },
            "backend": {
                "binary_sha256": "a" * 64,
                "build_sha256": None,
                "name": "llama.cpp",
                "source_sha256": blackbox.LLAMA_CPP_SOURCE_SHA256,
                "version": (
                    "version: 10015 "
                    f"({blackbox.LLAMA_CPP_SOURCE_COMMIT}) "
                    "built with Test for Linux x86_64"
                ),
            },
            "capabilities": {
                "metrics": ["exact_match"],
                "provider_name": "llama_cpp",
                "supported_claim_sets": ["invarlock-runtime-behavioral-regression-v1"],
                "tasks": ["text_causal"],
            },
            "device": {"device_kind": "cpu"},
            "execution_settings": {
                "allow_network": False,
                "batch_size": 32,
                "context_length": 256,
                "max_output_tokens": 16,
                "seed": 7,
                "timeout_seconds": 120,
            },
            "format_version": "invarlock/runtime-provider-receipt-v1",
            "outer_image_digest": image_digest,
            "plugin": {
                "distribution": "invarlock",
                "name": "llama_cpp",
                "provider_abi": "1",
            },
            "scoring_observation_sha256": blackbox.SCORING_OBSERVATION_SHA256,
        },
    }
    cli_observation = blackbox._expected_observation(
        schedule_sha256=blackbox.CLI_SCHEDULE_SHA256
    )
    cli_receipt = json.loads(json.dumps(value["receipt"]))
    cli_receipt["execution_settings"]["batch_size"] = 1
    cli_receipt["scoring_observation_sha256"] = blackbox.CLI_SCORING_OBSERVATION_SHA256
    value["cli_journey"]["observation"] = cli_observation
    value["cli_journey"]["provider_receipt"] = cli_receipt
    value["cli_journey"]["provider_receipt_sha256"] = hashlib.sha256(
        blackbox._canonical_json(cli_receipt)
    ).hexdigest()
    return blackbox._canonical_json(value) + b"\n"


def test_fixture_and_schedule_are_independently_pinned() -> None:
    schedule = (
        b"invarlock/gguf-runtime-blackbox-schedule-v1\x00"
        b"stories15m-q4-0-release-canary\x00"
        b"Once upon a time\x00"
        b", there was a little girl named Lily. She loved to play outside and"
    )

    assert blackbox.FIXTURE_REPOSITORY == "ggml-org/tiny-llamas"
    assert blackbox.FIXTURE_REVISION == "99dd1a73db5a37100bd4ae633f4cfce6560e1567"
    assert blackbox.FIXTURE_BYTE_LENGTH == 19_077_344
    assert blackbox.FIXTURE_SHA256 == (
        "6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04"
    )
    assert hashlib.sha256(schedule).hexdigest() == blackbox.SCHEDULE_SHA256

    cli_schedule = blackbox._canonical_json(
        {
            "dataset_identity": {
                "config_name": None,
                "dataset_name": None,
                "provider": "local_manifest",
                "revision": None,
                "split": "release-canary",
            },
            "format_version": "invarlock/runtime-behavioral-schedule-v1",
            "records": [
                {
                    "expected_output": blackbox.EXPECTED_OUTPUT,
                    "input_sha256": hashlib.sha256(
                        blackbox.PROMPT.encode("utf-8")
                    ).hexdigest(),
                    "input_text": blackbox.PROMPT,
                    "record_id": blackbox.RECORD_ID,
                }
            ],
        }
    )
    assert hashlib.sha256(cli_schedule).hexdigest() == blackbox.CLI_SCHEDULE_SHA256
    cli_observation = blackbox._canonical_json(
        blackbox._expected_observation(schedule_sha256=blackbox.CLI_SCHEDULE_SHA256)
    )
    assert (
        hashlib.sha256(cli_observation).hexdigest()
        == blackbox.CLI_SCORING_OBSERVATION_SHA256
    )
    settings = blackbox._canonical_json(
        {
            "allow_network": False,
            "batch_size": 1,
            "context_length": 256,
            "max_output_tokens": 16,
            "seed": 7,
            "timeout_seconds": 120,
        }
    )
    assert (
        hashlib.sha256(settings).hexdigest() == blackbox.CLI_EXECUTION_SETTINGS_SHA256
    )


def test_fixture_validation_fails_closed_without_leaking_path(tmp_path: Path) -> None:
    fixture = tmp_path / "private-host-fixture.gguf"
    with fixture.open("wb") as handle:
        handle.truncate(blackbox.FIXTURE_BYTE_LENGTH)

    with pytest.raises(blackbox.GGUFBlackBoxError) as error:
        blackbox._validate_fixture(fixture)

    assert str(fixture) not in str(error.value)
    assert "digest" in str(error.value)


def test_image_inspection_requires_actual_digest_and_pinned_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64
    inspection = [
        {
            "Id": digest,
            "Config": {
                "Labels": {
                    "dev.invarlock.runtime-provider": "llama_cpp",
                    "dev.invarlock.llama-cpp.source-commit": (
                        blackbox.LLAMA_CPP_SOURCE_COMMIT
                    ),
                    "dev.invarlock.llama-cpp.source-sha256": (
                        blackbox.LLAMA_CPP_SOURCE_SHA256
                    ),
                }
            },
        }
    ]
    monkeypatch.setattr(
        blackbox,
        "_run_captured",
        lambda *_args, **_kwargs: (0, json.dumps(inspection).encode(), b""),
    )

    assert blackbox._inspect_image("docker", "mutable:tag") == digest

    inspection[0]["Config"]["Labels"]["dev.invarlock.llama-cpp.source-commit"] = (
        "0" * 40
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match="labels"):
        blackbox._inspect_image("docker", "mutable:tag")


def test_container_command_is_immutable_offline_and_source_free(tmp_path: Path) -> None:
    digest = "sha256:" + "b" * 64
    model = tmp_path / "model.gguf"
    script = tmp_path / "blackbox.py"

    command = blackbox._container_command(
        engine="docker",
        image_digest=digest,
        model_path=model,
        script_path=script,
    )
    rendered = " ".join(command)

    assert command.count("--mount") == 2
    assert "--network none" in rendered
    assert "--read-only" in command
    assert "--cap-drop ALL" in rendered
    assert "--security-opt no-new-privileges" in rendered
    assert "INVARLOCK_ALLOW_HOST_EXECUTION=0" in command
    assert "INVARLOCK_ALLOW_NETWORK=0" in command
    assert "INVARLOCK_ALLOW_REMOTE_CODE=0" in command
    assert "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0" in command
    assert "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE=0" in command
    assert "/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777" in command
    assert (
        f"type=bind,src={model},dst=/fixtures/stories15M-q4_0.gguf,readonly" in command
    )
    assert (
        "type=bind,"
        f"src={script},dst=/opt/invarlock-blackbox/gguf_runtime_blackbox.py,readonly"
        in command
    )
    assert digest in command
    assert "PYTHONPATH" not in rendered
    assert "src=/src" not in rendered
    assert "src=/project" not in rendered

    with pytest.raises(blackbox.GGUFBlackBoxError, match="mount source"):
        blackbox._container_command(
            engine="docker",
            image_digest=digest,
            model_path=tmp_path / "unsafe,model.gguf",
            script_path=script,
        )


def test_result_validation_rejects_wrong_output_and_image() -> None:
    digest = "sha256:" + "c" * 64

    canonical = blackbox._validate_result_payload(
        _result_payload(image_digest=digest), image_digest=digest
    )
    assert canonical == _result_payload(image_digest=digest)[:-1]

    with pytest.raises(blackbox.GGUFBlackBoxError, match="output"):
        blackbox._validate_result_payload(
            _result_payload(image_digest=digest, output="normalized but wrong"),
            image_digest=digest,
        )
    with pytest.raises(blackbox.GGUFBlackBoxError, match="image digest"):
        blackbox._validate_result_payload(
            _result_payload(image_digest=digest),
            image_digest="sha256:" + "d" * 64,
        )


def test_result_validation_requires_the_complete_cli_journey() -> None:
    digest = "sha256:" + "c" * 64
    payload = json.loads(_result_payload(image_digest=digest))
    payload["cli_journey"]["portable_artifact_count"] = 16

    with pytest.raises(blackbox.GGUFBlackBoxError, match="artifact inventory"):
        blackbox._validate_result_payload(
            blackbox._canonical_json(payload) + b"\n",
            image_digest=digest,
        )


def test_portable_json_requires_canonical_path_free_regular_file(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(blackbox._canonical_json({"digest": "a" * 64}))

    payload, decoded = blackbox._portable_json(artifact)

    assert payload == blackbox._canonical_json(decoded)
    artifact.write_bytes(blackbox._canonical_json({"path": "/tmp/private"}))
    with pytest.raises(blackbox.GGUFBlackBoxError, match="runtime path"):
        blackbox._portable_json(artifact)


def test_installed_cli_helper_requires_canonical_versioned_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success = (
        blackbox._canonical_json({"format_version": "example-v1", "ok": True}) + b"\n"
    )
    monkeypatch.setattr(
        blackbox,
        "_run_captured",
        lambda *_args, **_kwargs: (0, success, b""),
    )

    assert (
        blackbox._run_installed_cli(("build-schedule",), expected_format="example-v1")[
            "ok"
        ]
        is True
    )

    failure = (
        blackbox._canonical_json(
            {
                "errors": ["output already exists"],
                "format_version": "example-v1",
                "ok": False,
            }
        )
        + b"\n"
    )
    monkeypatch.setattr(
        blackbox,
        "_run_captured",
        lambda *_args, **_kwargs: (2, failure, b""),
    )
    assert (
        blackbox._run_installed_cli(
            ("build-schedule",),
            expected_format="example-v1",
            expect_success=False,
        )["ok"]
        is False
    )


def test_host_runs_two_fresh_containers_and_requires_identical_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"placeholder")
    digest = "sha256:" + "e" * 64
    payload = _result_payload(image_digest=digest)[:-1]
    validations: list[Path] = []
    runs: list[str] = []
    monkeypatch.setattr(
        blackbox, "_validate_fixture", lambda path: validations.append(path)
    )
    monkeypatch.setattr(blackbox, "_inspect_image", lambda _engine, _image: digest)

    def run_once(**kwargs: object) -> bytes:
        runs.append(str(kwargs["image_digest"]))
        return payload

    monkeypatch.setattr(blackbox, "_run_container_once", run_once)

    result = blackbox._run_host(
        engine="docker", image="invarlock-runtime:gguf-local", model_path=model
    )

    assert len(validations) == 3
    assert runs == [digest, digest]
    assert result["runs"] == 2
    assert result["evidence_sha256"] == hashlib.sha256(payload).hexdigest()


def test_host_rejects_non_deterministic_canonical_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"placeholder")
    digest = "sha256:" + "f" * 64
    payloads = iter((b"first", b"second"))
    monkeypatch.setattr(blackbox, "_validate_fixture", lambda _path: None)
    monkeypatch.setattr(blackbox, "_inspect_image", lambda _engine, _image: digest)
    monkeypatch.setattr(
        blackbox, "_run_container_once", lambda **_kwargs: next(payloads)
    )

    with pytest.raises(blackbox.GGUFBlackBoxError, match="byte-identical"):
        blackbox._run_host(engine="docker", image="local:tag", model_path=model)


def test_public_failure_does_not_echo_private_host_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    private_path = tmp_path / "private-model-name.gguf"

    assert blackbox.main(["--model", str(private_path)]) == 2

    captured = capsys.readouterr()
    assert str(private_path) not in captured.err
    assert captured.out == ""
