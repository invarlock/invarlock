from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import typer
from click.termui import strip_ansi
from typer.testing import CliRunner

from invarlock.cli.commands import runtime_behavior as command_module
from invarlock.core.runtime_provider import (
    GGUFArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeExecutionSettings,
)
from invarlock.policy_pack import verify_policy_pack
from invarlock.reporting.validation.runtime_behavioral_claim import (
    runtime_execution_settings_sha256,
)

_SHA256 = "a" * 64
_IMAGE_DIGEST = f"sha256:{'b' * 64}"


class _LlamaProvider:
    name = "llama_cpp"

    def __init__(self) -> None:
        self.validated = False
        self.opened_context: object | None = None
        self.closed = False
        self.inspections = 0

    def validate_config(self, spec: object) -> None:
        self.validated = True

    def identify_artifact(self, spec: object) -> GGUFArtifactIdentity:
        return GGUFArtifactIdentity(
            artifact_name="model.gguf",
            sha256=_SHA256,
            byte_length=1,
            gguf_metadata_sha256="c" * 64,
            tensor_inventory_sha256="d" * 64,
            tokenizer_metadata_sha256="e" * 64,
        )

    def inspect_runtime_spec(
        self, bindings: object, **settings: int
    ) -> ModelRuntimeSpec:
        self.inspections += 1
        return ModelRuntimeSpec(
            provider_name=self.name,
            model_id="gguf-sha256-" + _SHA256 + ".gguf",
            settings={
                "artifact_byte_length": 1,
                "artifact_sha256": _SHA256,
                "backend_binary_sha256": "f" * 64,
                "backend_source_sha256": "1" * 64,
                "backend_version": "version: test built with test",
                "batch_size": settings["batch_size"],
                "context_length": settings["context_length"],
                "gguf_metadata_sha256": "c" * 64,
                "max_output_tokens": settings["max_output_tokens"],
                "seed": settings["seed"],
                "tensor_inventory_sha256": "d" * 64,
                "timeout_seconds": settings["timeout_seconds"],
                "tokenizer_metadata_sha256": "e" * 64,
            },
        )

    def open(self, spec: object, context: object) -> object:
        self.opened_context = context

        def _close() -> None:
            self.closed = True

        return SimpleNamespace(close=_close)


def _run_side_args(tmp_path: Path) -> list[str]:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "batch_size": 1,
                "context_length": 512,
                "max_output_tokens": 8,
                "seed": 7,
                "timeout_seconds": 60,
            }
        ),
        encoding="utf-8",
    )
    return [
        "run-side",
        "--role",
        "baseline",
        "--provider",
        "llama_cpp",
        "--model-id",
        "local-model",
        "--settings",
        str(settings),
        "--artifact",
        str(tmp_path / "model.gguf"),
        "--backend-executable",
        str(tmp_path / "llama-completion"),
        "--backend-source",
        str(tmp_path / "llama.cpp.tar"),
        "--container-image-digest",
        _IMAGE_DIGEST,
        "--schedule",
        str(tmp_path / "schedule.json"),
        "--policy-pack",
        str(tmp_path / "policy-pack.json"),
        "--out",
        str(tmp_path / "baseline-side"),
        "--json",
    ]


def _write_behavioral_policy_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_text = "Return the literal answer."
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "dataset_identity": {
                    "config_name": None,
                    "dataset_name": None,
                    "provider": "local_manifest",
                    "revision": None,
                    "split": "validation",
                },
                "format_version": "invarlock/runtime-behavioral-schedule-v1",
                "records": [
                    {
                        "expected_output": "answer",
                        "input_sha256": hashlib.sha256(
                            input_text.encode("utf-8")
                        ).hexdigest(),
                        "input_text": input_text,
                        "record_id": "example-1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _binding(path: Path, *, provider: str, artifact_format: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "artifact_format": artifact_format,
                    "artifact_identity_sha256": "a" * 64,
                    "execution_settings_sha256": "b" * 64,
                    "outer_image_digest": f"sha256:{'c' * 64}",
                    "provider_name": provider,
                }
            ),
            encoding="utf-8",
        )

    baseline = tmp_path / "baseline-binding.json"
    subject = tmp_path / "subject-binding.json"
    _binding(baseline, provider="llama_cpp", artifact_format="gguf")
    _binding(subject, provider="tensorrt_llm", artifact_format="tensorrt_llm_engine")
    return schedule, baseline, subject


def test_runtime_behavior_help_exposes_three_stage_journey() -> None:
    result = CliRunner().invoke(command_module.runtime_behavior_app, ["--help"])

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "run-side" in output
    assert "verify-pair" in output
    assert "build-policy" in output
    assert "build-schedule" in output
    assert "prepare-binding" in output
    assert "inspect-inputs" in output

    group = typer.main.get_command(command_module.runtime_behavior_app)
    assert isinstance(group, click.Group)
    run_command = group.get_command(click.Context(group), "run-side")
    assert run_command is not None
    declared_options = {
        option
        for parameter in run_command.params
        if isinstance(parameter, click.Option)
        for option in parameter.opts
    }
    for option in (
        "--role",
        "--provider",
        "--settings",
        "--artifact",
        "--container-image-digest",
        "--schedule",
        "--policy-pack",
        "--out",
    ):
        assert option in declared_options
    for removed_option in (
        "--device-kind",
        "--device-name",
        "--compute-capability",
        "--driver-version",
    ):
        assert removed_option not in declared_options


def test_prepare_binding_help_explains_provider_specific_and_boundary_inputs() -> None:
    result = CliRunner().invoke(
        command_module.runtime_behavior_app,
        ["prepare-binding", "--help"],
    )

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "same strict container boundary" in output
    assert "Required for llama_cpp" in output
    assert "Required for tensorrt_llm" in output
    assert "INVARLOCK_RUNTIME_IMAGE_DIGEST" in output
    assert "INVARLOCK_RUNTIME_IMAGE" in output
    assert "INVARLOCK_CONTAINER_EXECUTION=1" in output


def test_inspect_inputs_derives_reusable_settings_and_does_not_clobber(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _LlamaProvider()
    monkeypatch.setattr(command_module, "_provider", lambda name: provider)
    output = tmp_path / "derived-settings.json"
    args = [
        "inspect-inputs",
        "--provider",
        "llama_cpp",
        "--artifact",
        str(tmp_path / "private-model.gguf"),
        "--backend-executable",
        str(tmp_path / "private-llama-completion"),
        "--backend-source",
        str(tmp_path / "private-source.tar"),
        "--seed",
        "7",
        "--context-length",
        "512",
        "--batch-size",
        "1",
        "--max-output-tokens",
        "8",
        "--timeout-seconds",
        "60",
        "--out",
        str(output),
        "--json",
    ]

    result = CliRunner().invoke(command_module.runtime_behavior_app, args)

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    settings = json.loads(output.read_bytes())
    assert response["format_version"] == (
        command_module.RUNTIME_BEHAVIOR_INSPECT_INPUTS_CLI_FORMAT
    )
    assert response["model_id"] == "gguf-sha256-" + _SHA256 + ".gguf"
    assert response["artifact_format"] == "gguf"
    assert settings["artifact_sha256"] == _SHA256
    assert settings["backend_binary_sha256"] == "f" * 64
    assert settings["context_length"] == 512
    assert str(tmp_path) not in output.read_text(encoding="utf-8")

    repeated = CliRunner().invoke(command_module.runtime_behavior_app, args)
    assert repeated.exit_code == 2
    assert "output already exists" in json.loads(repeated.stdout)["errors"][0]
    assert provider.inspections == 1


def test_build_schedule_derives_input_digests_and_does_not_clobber(
    tmp_path: Path,
) -> None:
    records = tmp_path / "records.json"
    dataset = tmp_path / "dataset.json"
    output = tmp_path / "schedule.json"
    records.write_text(
        json.dumps(
            [
                {
                    "record_id": "example-1",
                    "input_text": "Return the literal answer.",
                    "expected_output": "answer",
                }
            ]
        ),
        encoding="utf-8",
    )
    dataset.write_text(
        json.dumps(
            {
                "provider": "local_manifest",
                "dataset_name": None,
                "config_name": None,
                "revision": None,
                "split": "validation",
            }
        ),
        encoding="utf-8",
    )
    args = [
        "build-schedule",
        "--records",
        str(records),
        "--dataset-identity",
        str(dataset),
        "--out",
        str(output),
        "--json",
    ]

    result = CliRunner().invoke(command_module.runtime_behavior_app, args)

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    schedule = json.loads(output.read_bytes())
    assert response["format_version"] == (
        command_module.RUNTIME_BEHAVIOR_BUILD_SCHEDULE_CLI_FORMAT
    )
    assert response["record_count"] == 1
    assert (
        schedule["records"][0]["input_sha256"]
        == hashlib.sha256(b"Return the literal answer.").hexdigest()
    )

    repeated = CliRunner().invoke(command_module.runtime_behavior_app, args)
    assert repeated.exit_code == 2
    assert "output already exists" in json.loads(repeated.stdout)["errors"][0]


def test_build_schedule_rejects_prehashed_or_unknown_record_fields(
    tmp_path: Path,
) -> None:
    records = tmp_path / "records.json"
    dataset = tmp_path / "dataset.json"
    records.write_text(
        json.dumps(
            [
                {
                    "record_id": "example-1",
                    "input_text": "prompt",
                    "input_sha256": "a" * 64,
                    "expected_output": "answer",
                }
            ]
        ),
        encoding="utf-8",
    )
    dataset.write_text(
        json.dumps(
            {
                "provider": "local_manifest",
                "dataset_name": None,
                "config_name": None,
                "revision": None,
                "split": "validation",
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        command_module.runtime_behavior_app,
        [
            "build-schedule",
            "--records",
            str(records),
            "--dataset-identity",
            str(dataset),
            "--out",
            str(tmp_path / "schedule.json"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "unknown input_sha256" in json.loads(result.stdout)["errors"][0]


def test_prepare_binding_validates_native_inputs_and_does_not_clobber(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _LlamaProvider()
    monkeypatch.setattr(command_module, "_provider", lambda name: provider)
    args = _run_side_args(tmp_path)
    args[0] = "prepare-binding"
    for option in ("--role", "--schedule", "--policy-pack"):
        index = args.index(option)
        del args[index : index + 2]
    args[args.index("--out") + 1] = str(tmp_path / "baseline-binding.json")

    result = CliRunner().invoke(command_module.runtime_behavior_app, args)

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    binding_path = tmp_path / "baseline-binding.json"
    binding = json.loads(binding_path.read_bytes())
    expected_settings_sha256 = runtime_execution_settings_sha256(
        RuntimeExecutionSettings(
            seed=7,
            context_length=512,
            batch_size=1,
            max_output_tokens=8,
            timeout_seconds=60,
            allow_network=False,
        )
    )
    assert response["format_version"] == (
        command_module.RUNTIME_BEHAVIOR_PREPARE_BINDING_CLI_FORMAT
    )
    assert binding == {
        "artifact_format": "gguf",
        "artifact_identity_sha256": response["artifact_identity_sha256"],
        "execution_settings_sha256": expected_settings_sha256,
        "outer_image_digest": _IMAGE_DIGEST,
        "provider_name": "llama_cpp",
    }
    assert response["execution_settings_sha256"] == expected_settings_sha256
    assert str(tmp_path) not in binding_path.read_text(encoding="utf-8")
    assert provider.opened_context is not None
    assert provider.closed is True

    repeated = CliRunner().invoke(command_module.runtime_behavior_app, args)
    assert repeated.exit_code == 2
    assert "output already exists" in json.loads(repeated.stdout)["errors"][0]


def test_build_policy_writes_directed_v3_without_clobber(tmp_path: Path) -> None:
    schedule, baseline, subject = _write_behavioral_policy_inputs(tmp_path)
    output = tmp_path / "acceptance-policy-pack.json"
    args = [
        "build-policy",
        "--schedule",
        str(schedule),
        "--baseline-binding",
        str(baseline),
        "--subject-binding",
        str(subject),
        "--minimum-subject-score",
        "0.95",
        "--maximum-regression",
        "0.01",
        "--evidence-surface",
        "behavior",
        "--evidence-surface",
        "tokenizer",
        "--out",
        str(output),
        "--json",
    ]

    result = CliRunner().invoke(command_module.runtime_behavior_app, args)

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response["format_version"] == (
        command_module.RUNTIME_BEHAVIOR_BUILD_POLICY_CLI_FORMAT
    )
    assert response["ok"] is True
    assert response["output"] == str(output)
    pack = json.loads(output.read_bytes())
    assert output.read_bytes() == json.dumps(
        pack,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert pack["format"] == "policy-pack-v3"
    assert pack["behavioral_claim"]["baseline"]["provider_name"] == "llama_cpp"
    assert pack["behavioral_claim"]["subject"]["provider_name"] == "tensorrt_llm"
    assert pack["behavioral_claim"]["schedule_sha256"] == response["schedule_sha256"]
    assert verify_policy_pack(pack) == []

    repeated = CliRunner().invoke(command_module.runtime_behavior_app, args)
    assert repeated.exit_code == 2
    failure = json.loads(repeated.stdout)
    assert failure["ok"] is False
    assert "output already exists" in failure["errors"][0]
    assert json.loads(output.read_bytes()) == pack


def test_build_policy_rejects_incomplete_role_binding(tmp_path: Path) -> None:
    schedule, baseline, subject = _write_behavioral_policy_inputs(tmp_path)
    binding = json.loads(subject.read_text(encoding="utf-8"))
    del binding["execution_settings_sha256"]
    subject.write_text(json.dumps(binding), encoding="utf-8")

    result = CliRunner().invoke(
        command_module.runtime_behavior_app,
        [
            "build-policy",
            "--schedule",
            str(schedule),
            "--baseline-binding",
            str(baseline),
            "--subject-binding",
            str(subject),
            "--minimum-subject-score",
            "0.95",
            "--maximum-regression",
            "0.01",
            "--out",
            str(tmp_path / "policy.json"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    failure = json.loads(result.stdout)
    assert failure["ok"] is False
    assert "missing execution_settings_sha256" in failure["errors"][0]


def test_run_side_wires_native_provider_to_public_api(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _LlamaProvider()
    captured: dict[str, object] = {}

    def _run(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            directory=tmp_path / "baseline-side",
            manifest_path=tmp_path / "baseline-side" / "runtime.manifest.json",
        )

    monkeypatch.setattr(command_module, "_provider", lambda name: provider)
    monkeypatch.setattr(command_module, "_run_side_api", _run)

    result = CliRunner().invoke(
        command_module.runtime_behavior_app, _run_side_args(tmp_path)
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == {
        "format_version": command_module.RUNTIME_BEHAVIOR_RUN_SIDE_CLI_FORMAT,
        "manifest": str(tmp_path / "baseline-side" / "runtime.manifest.json"),
        "ok": True,
        "provider_name": "llama_cpp",
        "role": "baseline",
        "side_directory": str(tmp_path / "baseline-side"),
    }
    assert provider.validated is True
    assert captured["role"] == "baseline"
    assert captured["provider"] is provider
    assert captured["schedule_path"] == tmp_path / "schedule.json"
    assert captured["policy_pack_path"] == tmp_path / "policy-pack.json"
    context = captured["context"]
    assert context.strict is True
    assert context.allow_network is False
    assert context.container_image_digest == _IMAGE_DIGEST
    assert context.device_kind == "cpu"
    assert context.native_model.gguf_path == tmp_path / "model.gguf"


def test_tensorrt_binding_adapter_requires_tokenizer_and_infers_cuda(
    tmp_path: Path,
) -> None:
    bindings = command_module._native_bindings(
        provider_name="tensorrt_llm",
        artifact=str(tmp_path / "engine"),
        backend_executable=str(tmp_path / "tensorrt-llm-runner"),
        backend_source=None,
        tokenizer_contract=str(tmp_path / "tokenizer.json"),
    )

    assert bindings.engine_bundle_path == tmp_path / "engine"
    assert bindings.runner_executable_path == tmp_path / "tensorrt-llm-runner"
    assert bindings.tokenizer_contract_path == tmp_path / "tokenizer.json"
    assert command_module._provider_device_kind("tensorrt_llm") == "cuda"

    with pytest.raises(ValueError, match="--tokenizer-contract is required"):
        command_module._native_bindings(
            provider_name="tensorrt_llm",
            artifact=str(tmp_path / "engine"),
            backend_executable=str(tmp_path / "tensorrt-llm-runner"),
            backend_source=None,
            tokenizer_contract=None,
        )


def test_verify_pair_wires_independent_replay_to_public_api(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def _verify(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            verification=SimpleNamespace(
                baseline_score=0.9,
                subject_score=0.9,
                regression=0.0,
            ),
            receipt_path=tmp_path / "paired-receipt.json",
        )

    monkeypatch.setattr(command_module, "_verify_pair_api", _verify)
    result = CliRunner().invoke(
        command_module.runtime_behavior_app,
        [
            "verify-pair",
            "--baseline",
            str(tmp_path / "baseline"),
            "--subject",
            str(tmp_path / "subject"),
            "--schedule",
            str(tmp_path / "schedule.json"),
            "--policy-pack",
            str(tmp_path / "policy-pack.json"),
            "--receipt",
            str(tmp_path / "paired-receipt.json"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "baseline_score": 0.9,
        "format_version": command_module.RUNTIME_BEHAVIOR_VERIFY_PAIR_CLI_FORMAT,
        "ok": True,
        "receipt": str(tmp_path / "paired-receipt.json"),
        "regression": 0.0,
        "subject_score": 0.9,
    }
    assert captured == {
        "baseline_directory": tmp_path / "baseline",
        "subject_directory": tmp_path / "subject",
        "schedule_path": tmp_path / "schedule.json",
        "policy_pack_path": tmp_path / "policy-pack.json",
        "receipt_path": tmp_path / "paired-receipt.json",
    }


def test_run_side_rejects_duplicate_settings_before_execution(
    tmp_path: Path, monkeypatch
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{"seed":1,"seed":2}', encoding="utf-8")
    called = False

    def _run(**kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("execution must not start")

    monkeypatch.setattr(command_module, "_run_side_api", _run)
    args = _run_side_args(tmp_path)
    Path(args[args.index("--settings") + 1]).write_text(
        '{"seed":1,"seed":2}', encoding="utf-8"
    )
    result = CliRunner().invoke(command_module.runtime_behavior_app, args)

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "duplicate key" in payload["errors"][0]
    assert called is False


def test_hf_standalone_side_production_fails_with_actionable_message(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _LlamaProvider()
    monkeypatch.setattr(command_module, "_provider", lambda name: provider)
    args = _run_side_args(tmp_path)
    args[args.index("llama_cpp")] = "hf_transformers"

    result = CliRunner().invoke(command_module.runtime_behavior_app, args)

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "Python run_side API with prebound objects" in payload["errors"][0]


def test_verify_pair_failure_is_machine_readable(tmp_path: Path, monkeypatch) -> None:
    def _verify(**kwargs: object) -> object:
        raise ValueError("paired runtime behavioral claim failed")

    monkeypatch.setattr(command_module, "_verify_pair_api", _verify)
    result = CliRunner().invoke(
        command_module.runtime_behavior_app,
        [
            "verify-pair",
            "--baseline",
            str(tmp_path / "baseline"),
            "--subject",
            str(tmp_path / "subject"),
            "--schedule",
            str(tmp_path / "schedule.json"),
            "--policy-pack",
            str(tmp_path / "policy-pack.json"),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "errors": ["paired runtime behavioral claim failed"],
        "format_version": command_module.RUNTIME_BEHAVIOR_VERIFY_PAIR_CLI_FORMAT,
        "ok": False,
    }
