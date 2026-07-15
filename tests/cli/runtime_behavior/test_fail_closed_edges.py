from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from invarlock.cli.commands import runtime_behavior as command_module
from invarlock.core.runtime_provider import (
    GGUFArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
)

_SHA256 = "a" * 64
_IMAGE_DIGEST = "sha256:" + "b" * 64


def test_lazy_runtime_behavior_api_wrappers_forward_exact_arguments(
    monkeypatch,
) -> None:
    import invarlock.runtime_behavior as behavior_api

    side_marker = object()
    pair_marker = object()
    monkeypatch.setattr(
        behavior_api,
        "run_side",
        lambda **kwargs: (side_marker, kwargs),
    )
    monkeypatch.setattr(
        behavior_api,
        "verify_pair",
        lambda **kwargs: (pair_marker, kwargs),
    )

    assert command_module._run_side_api(value=7) == (side_marker, {"value": 7})
    assert command_module._verify_pair_api(value=9) == (pair_marker, {"value": 9})
    assert command_module._provider("llama_cpp").name == "llama_cpp"


def test_scalar_json_loader_rejects_invalid_root_nested_values_and_syntax(
    tmp_path: Path,
) -> None:
    nonobject = tmp_path / "nonobject.json"
    nonobject.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        command_module._load_settings(nonobject)

    nested = tmp_path / "nested.json"
    nested.write_text('{"seed":{"nested":true}}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON scalar values"):
        command_module._load_settings(nested)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        command_module._load_json_value(invalid, label="input", max_bytes=100)


def test_behavioral_binding_reports_missing_and_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"
    path.write_text(
        json.dumps(
            {
                "provider_name": "llama_cpp",
                "artifact_format": "gguf",
                "artifact_identity_sha256": _SHA256,
                "outer_image_digest": _IMAGE_DIGEST,
                "unexpected": "value",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing execution_settings_sha256; unknown unexpected",
    ):
        command_module._load_behavioral_binding(path, role="baseline")


def test_native_binding_helpers_reject_unsupported_or_incomplete_providers(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="--backend-source is required"):
        command_module._native_bindings(
            provider_name="llama_cpp",
            artifact=str(tmp_path / "model.gguf"),
            backend_executable=str(tmp_path / "llama-completion"),
            backend_source=" ",
            tokenizer_contract=None,
        )
    with pytest.raises(ValueError, match="no ephemeral binding adapter"):
        command_module._native_bindings(
            provider_name="unknown_provider",
            artifact="artifact",
            backend_executable="runner",
            backend_source=None,
            tokenizer_contract=None,
        )
    with pytest.raises(ValueError, match="cannot infer a device kind"):
        command_module._provider_device_kind("unknown_provider")


def test_emit_and_fail_plain_text_paths(capsys) -> None:
    command_module._emit({"ok": True}, json_out=False, success="completed")
    assert capsys.readouterr().out == "completed\n"

    with pytest.raises(typer.Exit) as raised:
        command_module._fail(
            format_version="test-v1",
            json_out=False,
            error=ValueError(" \n "),
        )
    captured = capsys.readouterr()
    assert raised.value.exit_code == 2
    assert (
        captured.err
        == "Runtime behavior command failed: runtime behavior command failed\n"
    )


@pytest.mark.parametrize(
    ("records_payload", "dataset_payload", "message"),
    [
        ({"not": "an array"}, {}, "records must be a JSON array"),
        ([], [], "dataset identity must be a JSON object"),
    ],
)
def test_build_schedule_cli_rejects_wrong_json_container_types(
    tmp_path: Path,
    records_payload: object,
    dataset_payload: object,
    message: str,
) -> None:
    records = tmp_path / "records.json"
    dataset = tmp_path / "dataset.json"
    records.write_text(json.dumps(records_payload), encoding="utf-8")
    dataset.write_text(json.dumps(dataset_payload), encoding="utf-8")

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
    assert message in json.loads(result.stdout)["errors"][0]


class _CloseFailingSession:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("native close failed")


def test_prepare_binding_retries_close_and_reports_failure(
    tmp_path: Path, monkeypatch
) -> None:
    session = _CloseFailingSession()
    provider = SimpleNamespace(name="llama_cpp", open=lambda spec, context: session)
    spec = ModelRuntimeSpec(
        provider_name="llama_cpp",
        model_id="model",
        settings={
            "seed": 0,
            "context_length": 32,
            "batch_size": 1,
            "max_output_tokens": 4,
            "timeout_seconds": 5,
        },
    )
    identity = GGUFArtifactIdentity(
        artifact_name="model.gguf",
        sha256=_SHA256,
        byte_length=1,
        gguf_metadata_sha256="b" * 64,
        tensor_inventory_sha256="c" * 64,
        tokenizer_metadata_sha256="d" * 64,
    )
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_SHA256,
    )
    monkeypatch.setattr(
        command_module,
        "_native_provider_inputs",
        lambda **kwargs: (provider, spec, context, identity),
    )

    result = CliRunner().invoke(
        command_module.runtime_behavior_app,
        [
            "prepare-binding",
            "--provider",
            "llama_cpp",
            "--model-id",
            "model",
            "--settings",
            str(tmp_path / "settings.json"),
            "--artifact",
            str(tmp_path / "model.gguf"),
            "--backend-executable",
            str(tmp_path / "llama-completion"),
            "--backend-source",
            str(tmp_path / "source.tar"),
            "--container-image-digest",
            _IMAGE_DIGEST,
            "--out",
            str(tmp_path / "binding.json"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert session.close_calls == 2
    assert "native close failed" in json.loads(result.stdout)["errors"][0]
