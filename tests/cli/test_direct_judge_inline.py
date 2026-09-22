"""Direct native judging uses one declared container execution boundary."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import yaml
from typer.testing import CliRunner

import invarlock.evaluation_oci as evaluation_oci
import invarlock.runtime_security_helpers as runtime_security
from invarlock.cli.app import app
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    HFSnapshotArtifactIdentity,
    artifact_identity_sha256,
)
from invarlock.judge_measurements import runtime_provider as runtime_judge
from tests.cli.test_evaluation_preflight import _materialize_run_request

IMAGE_DIGEST = "sha256:" + "9" * 64


def _direct_request(tmp_path: Path) -> tuple[Path, Path]:
    request_path, signing_key = _materialize_run_request(tmp_path)
    document = yaml.safe_load(request_path.read_text())
    judge_model = tmp_path / "models" / "judge"
    judge_model.mkdir()
    judge_model.joinpath("config.json").write_text(
        json.dumps({"model_type": "test", "role": "judge"})
    )
    checkpoint_digest = checkpoint_tree_sha256(judge_model).removeprefix("sha256:")
    document["comparison"].update(
        metric="judge",
        judge={
            "workspace": "judge-workspace",
            "signer_identity": "release",
            "model": {
                "artifact": {
                    "path": "models/judge",
                    "model_id": "local/judge",
                    "locator": "hf://local/judge@" + "b" * 40,
                },
                "runtime": {
                    "provider": "hf_transformers",
                    "settings": {
                        "batch_size": 1,
                        "checkpoint_tree_sha256": checkpoint_digest,
                        "context_length": 128,
                        "immutable_revision": "b" * 40,
                        "max_output_tokens": 16,
                        "offline": True,
                        "seed": 0,
                        "timeout_seconds": 30,
                        "tokenizer_metadata_sha256": "d" * 64,
                    },
                },
            },
        },
    )
    request_path.write_text(yaml.safe_dump(document, sort_keys=False))

    recipe = json.loads(
        (
            Path(__file__).parents[2]
            / "examples/native-local-judge/recipe-template.json"
        ).read_text()
    )
    recipe["plan"]["sampling"]["case_units"] = [{"case_id": "one", "unit_id": "one"}]
    recipe["plan"]["judge"].update(
        requested_model="local/judge",
        approved_resolved_models=["local/judge"],
    )
    judge_identity = HFSnapshotArtifactIdentity(
        model_id="local/judge",
        immutable_revision="b" * 40,
        checkpoint_tree_sha256=checkpoint_digest,
        tokenizer_metadata_sha256="d" * 64,
    )
    recipe["plan"]["judge"]["model_identity"]["weights_sha256"] = (
        artifact_identity_sha256(judge_identity)
    )
    recipe["plan"]["judge"]["config"]["max_output_tokens"] = 16
    recipe["analysis"]["minimum_units"] = 1
    recipe["collection"].update(max_calls=2, max_output_tokens=32)
    (tmp_path / "inputs" / "policy.json").write_text(json.dumps(recipe))
    return request_path, signing_key


def _environment() -> dict[str, str]:
    return {
        "INVARLOCK_RUNTIME_IMAGE": "example/runtime",
        "INVARLOCK_RUNTIME_IMAGE_DIGEST": IMAGE_DIGEST,
        "INVARLOCK_ALLOW_NETWORK": "0",
        "INVARLOCK_ALLOW_REMOTE_CODE": "0",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS": "0",
    }


def test_direct_judge_cli_selects_inline_and_checks_boundary_before_model_work(
    tmp_path: Path, monkeypatch
) -> None:
    request, signing_key = _direct_request(tmp_path)
    delegated = Mock(side_effect=AssertionError("OCI delegation must not run"))
    authenticated = Mock(
        side_effect=AssertionError("model authentication must not run")
    )
    monkeypatch.setattr(evaluation_oci, "preflight_oci_launch", delegated)
    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.authenticate_artifact",
        authenticated,
    )
    monkeypatch.setattr(
        runtime_security, "strict_container_boundary_present", lambda: False
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(signing_key),
            "--runtime-image-digest",
            IMAGE_DIGEST,
            "--preflight",
            "--json",
        ],
        env=_environment(),
    )

    assert result.exit_code != 0
    assert "strict offline container" in result.stdout
    delegated.assert_not_called()
    authenticated.assert_not_called()
    assert not (tmp_path / "judge-workspace").exists()
    assert not (tmp_path / "artifacts").exists()


def test_direct_judge_cli_preflight_uses_one_inline_container(
    tmp_path: Path, monkeypatch
) -> None:
    request, signing_key = _direct_request(tmp_path)
    delegated = Mock(side_effect=AssertionError("OCI delegation must not run"))
    monkeypatch.setattr(evaluation_oci, "preflight_oci_launch", delegated)
    monkeypatch.setattr(
        runtime_security, "strict_container_boundary_present", lambda: True
    )
    monkeypatch.setattr(
        runtime_judge, "strict_container_boundary_present", lambda: True
    )
    monkeypatch.setattr(runtime_judge, "network_allowed", lambda: False)
    monkeypatch.setattr(runtime_judge, "remote_code_allowed", lambda: False)
    monkeypatch.setattr(runtime_judge, "third_party_plugins_allowed", lambda: False)

    def authenticate(_self, spec, _path):
        return HFSnapshotArtifactIdentity(
            model_id=spec.model_id,
            immutable_revision="b" * 40,
            checkpoint_tree_sha256=spec.settings["checkpoint_tree_sha256"],
            tokenizer_metadata_sha256=spec.settings["tokenizer_metadata_sha256"],
        )

    monkeypatch.setattr(
        "invarlock.runtime_providers.hf_transformers.HFTransformersProvider.authenticate_artifact",
        authenticate,
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(signing_key),
            "--runtime-image-digest",
            IMAGE_DIGEST,
            "--preflight",
            "--json",
        ],
        env=_environment(),
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["execution_mode"] == "run"
    assert payload["runtime_image_digests"] == {
        "baseline": IMAGE_DIGEST,
        "subject": IMAGE_DIGEST,
    }
    assert payload["judge"]["collection"]["outer_image_digest"] == IMAGE_DIGEST
    assert payload["judge"]["collection"]["strict_offline_container"] is True
    delegated.assert_not_called()
    assert not (tmp_path / "judge-workspace").exists()
    assert not (tmp_path / "artifacts").exists()


def test_direct_judge_cli_rejects_digest_split_and_oci_controls_before_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    request, signing_key = _direct_request(tmp_path)
    delegated = Mock(side_effect=AssertionError("OCI delegation must not run"))
    model_preflight = Mock(side_effect=AssertionError("model preflight must not run"))
    monkeypatch.setattr(evaluation_oci, "preflight_oci_launch", delegated)
    monkeypatch.setattr(runtime_judge, "preflight_runtime_provider", model_preflight)
    monkeypatch.setattr(
        runtime_security, "strict_container_boundary_present", lambda: True
    )

    cases = (
        (
            ["--subject-runtime-image-digest", "sha256:" + "8" * 64],
            "requires baseline, subject, and judge",
        ),
        (["--runtime-cpus", "2"], "OCI-only controls are not valid"),
        (["--baseline-runtime-device", "cuda:0"], "devices must be cpu or cuda"),
        (
            ["--runtime-image", "other/runtime"],
            "image references must equal the current container",
        ),
    )
    for arguments, message in cases:
        result = CliRunner().invoke(
            app,
            [
                "evaluate",
                str(request),
                "--signing-key",
                str(signing_key),
                "--runtime-image-digest",
                IMAGE_DIGEST,
                "--preflight",
                "--json",
                *arguments,
            ],
            env=_environment(),
        )
        assert result.exit_code != 0
        assert message in result.stdout

    unsafe_environment = _environment()
    unsafe_environment["INVARLOCK_ALLOW_NETWORK"] = "1"
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(signing_key),
            "--runtime-image-digest",
            IMAGE_DIGEST,
            "--preflight",
            "--json",
        ],
        env=unsafe_environment,
    )
    assert result.exit_code != 0
    assert "INVARLOCK_ALLOW_NETWORK" in result.stdout

    missing_digest_environment = _environment()
    del missing_digest_environment["INVARLOCK_RUNTIME_IMAGE_DIGEST"]
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(signing_key),
            "--runtime-image-digest",
            IMAGE_DIGEST,
            "--preflight",
            "--json",
        ],
        env=missing_digest_environment,
    )
    assert result.exit_code != 0
    assert "INVARLOCK_RUNTIME_IMAGE_DIGEST must bind" in result.stdout

    invalid_digest_environment = _environment()
    invalid_digest_environment["INVARLOCK_RUNTIME_IMAGE_DIGEST"] = "invalid"
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(signing_key),
            "--runtime-image-digest",
            IMAGE_DIGEST,
            "--preflight",
            "--json",
        ],
        env=invalid_digest_environment,
    )
    assert result.exit_code != 0
    assert "must use lowercase sha256" in result.stdout

    delegated.assert_not_called()
    model_preflight.assert_not_called()
