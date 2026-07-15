from __future__ import annotations

import copy
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock import runtime_provider_evidence
from invarlock.core.runtime_provider import GGUFArtifactIdentity
from invarlock.runtime_providers import gguf_identity, llama_cpp
from scripts.release import gguf_runtime_blackbox as blackbox
from tests.scripts._gguf_blackbox_support import (
    exact_schedule,
    valid_cli_journey,
    valid_receipt,
    valid_result,
    write_json,
    write_side_bundle,
)


def test_bounded_fixture_hashing_accepts_only_the_exact_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture.gguf"
    fixture.write_bytes(b"exact fixture")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    monkeypatch.setattr(blackbox, "FIXTURE_BYTE_LENGTH", fixture.stat().st_size)
    monkeypatch.setattr(blackbox, "FIXTURE_SHA256", digest)

    assert blackbox._sha256_file(fixture) == digest
    assert blackbox._validate_fixture(fixture) is None

    fixture.write_bytes(b"wrong length")
    with pytest.raises(blackbox.GGUFBlackBoxError, match="byte length"):
        blackbox._sha256_file(fixture)
    with pytest.raises(blackbox.GGUFBlackBoxError, match="unavailable"):
        blackbox._validate_fixture(tmp_path / "missing.gguf")

    symlink = tmp_path / "fixture-link.gguf"
    symlink.symlink_to(fixture)
    with pytest.raises(blackbox.GGUFBlackBoxError, match="non-symlink"):
        blackbox._validate_fixture(symlink)


def test_bounded_fixture_hashing_normalizes_open_and_read_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture.gguf"
    fixture.write_bytes(b"x")
    monkeypatch.setattr(blackbox, "FIXTURE_BYTE_LENGTH", 1)

    monkeypatch.setattr(
        blackbox.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private path")),
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match="opened safely") as error:
        blackbox._sha256_file(fixture)
    assert str(fixture) not in str(error.value)


def test_captured_command_enforces_launch_timeout_and_output_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, stdout, stderr = blackbox._run_captured(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        timeout_seconds=10,
        stdout_limit=32,
        stderr_limit=32,
    )
    assert (status, stdout, stderr) == (0, b"out\n", b"err\n")

    with pytest.raises(blackbox.GGUFBlackBoxError, match="output limit"):
        blackbox._run_captured(
            [sys.executable, "-c", "print('too much output')"],
            timeout_seconds=10,
            stdout_limit=2,
            stderr_limit=2,
        )

    monkeypatch.setattr(
        blackbox.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("not installed")),
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match="could not be started"):
        blackbox._run_captured(
            ["missing"], timeout_seconds=1, stdout_limit=1, stderr_limit=1
        )


def test_captured_command_kills_a_timed_out_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        waits = 0
        killed = False

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["slow"], 1)
            return -9

        def kill(self) -> None:
            self.killed = True

    process = Process()
    monkeypatch.setattr(blackbox.subprocess, "Popen", lambda *_a, **_k: process)
    with pytest.raises(blackbox.GGUFBlackBoxError, match="time limit"):
        blackbox._run_captured(
            ["slow"], timeout_seconds=1, stdout_limit=1, stderr_limit=1
        )
    assert process.killed is True
    assert process.waits == 2


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (1, b"[]", "could not be inspected"),
        (0, b"not-json", "inspection is invalid"),
        (0, b"[]", "inspection is ambiguous"),
        (0, b'[{"Id":"mutable"}]', "no canonical digest"),
    ],
)
def test_image_inspection_rejects_ambiguous_engine_evidence(
    monkeypatch: pytest.MonkeyPatch, status: int, payload: bytes, expected: str
) -> None:
    monkeypatch.setattr(
        blackbox,
        "_run_captured",
        lambda *_args, **_kwargs: (status, payload, b"private details"),
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._inspect_image("docker", "candidate")


def test_container_command_rejects_mutable_digest_and_control_characters(
    tmp_path: Path,
) -> None:
    with pytest.raises(blackbox.GGUFBlackBoxError, match="digest is invalid"):
        blackbox._container_command(
            engine="docker",
            image_digest="mutable",
            model_path=tmp_path / "model.gguf",
            script_path=tmp_path / "script.py",
        )
    with pytest.raises(blackbox.GGUFBlackBoxError, match="mount source"):
        blackbox._container_command(
            engine="docker",
            image_digest="sha256:" + "a" * 64,
            model_path=tmp_path / "model\n.gguf",
            script_path=tmp_path / "script.py",
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("format_version", "format"),
        ("artifact_identity", "artifact"),
        ("outer_image_digest", "image digest"),
        ("scoring_observation_sha256", "observation binding"),
        ("execution_settings", "settings"),
        ("backend", "backend"),
        ("capabilities", "capabilities"),
        ("device", "device"),
        ("plugin", "plugin"),
    ],
)
def test_provider_receipt_validator_fails_each_authenticated_surface(
    mutation: str, expected: str
) -> None:
    image_digest = "sha256:" + "b" * 64
    receipt = valid_receipt(image_digest=image_digest)
    blackbox._validate_provider_receipt(
        receipt,
        image_digest=image_digest,
        batch_size=32,
        observation_sha256=blackbox.SCORING_OBSERVATION_SHA256,
    )
    receipt[mutation] = {} if mutation in receipt else "invalid"
    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._validate_provider_receipt(
            receipt,
            image_digest=image_digest,
            batch_size=32,
            observation_sha256=blackbox.SCORING_OBSERVATION_SHA256,
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("field_set", "incomplete"),
        ("format_version", "format"),
        ("artifact_identity_sha256", "artifact binding"),
        ("execution_settings_sha256", "settings binding"),
        ("schedule_sha256", "schedule binding"),
        ("observation_sha256", "observation binding"),
        ("observation", "known-answer observation"),
        ("binding_sha256", "binding_sha256"),
        ("policy_digest", "policy digest"),
        ("provider_receipt", "provider receipt is incomplete"),
        ("provider_receipt_sha256", "receipt digest"),
        ("portable_artifact_count", "artifact inventory"),
        ("verification", "paired verification"),
    ],
)
def test_cli_summary_validator_rejects_every_binding(
    mutation: str, expected: str
) -> None:
    digest = "sha256:" + "c" * 64
    journey = valid_cli_journey(image_digest=digest)
    if mutation == "field_set":
        journey.pop("binding_sha256")
    elif mutation == "portable_artifact_count":
        journey[mutation] = 16
    elif mutation == "verification":
        journey[mutation] = {"verdict": "fail"}
    elif mutation == "provider_receipt":
        journey[mutation] = []
    elif mutation == "provider_receipt_sha256":
        journey[mutation] = "0" * 64
    else:
        journey[mutation] = {} if mutation == "observation" else "bad"
    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._validate_cli_journey_summary(journey, image_digest=digest)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("field_set", "field set"),
        ("format_version", "result format"),
        ("image_digest", "image digest"),
        ("fixture", "fixture identity"),
        ("observation_type", "evidence is incomplete"),
        ("observation", "output does not match"),
    ],
)
def test_result_validator_rejects_tampered_top_level_evidence(
    mutation: str, expected: str
) -> None:
    digest = "sha256:" + "d" * 64
    result = valid_result(image_digest=digest)
    if mutation == "field_set":
        result["unexpected"] = True
    elif mutation == "observation_type":
        result["observation"] = []
    elif mutation == "observation":
        result["observation"] = copy.deepcopy(result["observation"])
        assert isinstance(result["observation"], dict)
        result["observation"]["provider_name"] = "other"
    else:
        result[mutation] = "bad"
    payload = blackbox._canonical_json(result) + b"\n"
    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._validate_result_payload(payload, image_digest=digest)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"{}", "framing"),
        (b"{}\n\n", "framing"),
        (b"not-json\n", "result is invalid"),
        (b'{"b":1, "a":2}\n', "not canonical"),
    ],
)
def test_result_validator_requires_canonical_single_line_framing(
    payload: bytes, expected: str
) -> None:
    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._validate_result_payload(payload, image_digest="sha256:" + "e" * 64)


def test_run_container_once_validates_exit_status_and_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "sha256:" + "f" * 64
    result = blackbox._canonical_json(valid_result(image_digest=digest)) + b"\n"
    monkeypatch.setattr(blackbox, "_run_captured", lambda *_a, **_k: (0, result, b""))
    canonical = blackbox._run_container_once(
        engine="docker",
        image_digest=digest,
        model_path=tmp_path / "model.gguf",
        script_path=tmp_path / "script.py",
    )
    assert canonical == result[:-1]

    monkeypatch.setattr(
        blackbox, "_run_captured", lambda *_a, **_k: (7, b"", b"private")
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match="container run failed"):
        blackbox._run_container_once(
            engine="docker",
            image_digest=digest,
            model_path=tmp_path / "model.gguf",
            script_path=tmp_path / "script.py",
        )


@pytest.mark.parametrize(
    ("status", "stdout", "stderr", "expected"),
    [
        (1, b"", b"", "version probe failed"),
        (0, b"\xff", b"", "version is invalid"),
        (0, b"version: duplicate\n", b"", "version is ambiguous"),
        (0, b"version: 1\nbuilt with test\n", b"", "does not match"),
    ],
)
def test_backend_version_probe_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    stdout: bytes,
    stderr: bytes,
    expected: str,
) -> None:
    monkeypatch.setattr(
        blackbox,
        "_run_captured",
        lambda *_args, **_kwargs: (status, stdout, stderr),
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
        blackbox._normalized_backend_version()


def test_backend_version_probe_normalizes_the_pinned_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = f"version: 10015 ({blackbox.LLAMA_CPP_SOURCE_COMMIT})"
    monkeypatch.setattr(
        blackbox,
        "_run_captured",
        lambda *_a, **_k: (
            0,
            f" {version} \n".encode(),
            b"built   with Test for Linux x86_64\n",
        ),
    )
    assert blackbox._normalized_backend_version() == (
        f"{version} built with Test for Linux x86_64"
    )


def test_installed_wheel_guard_rejects_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/checkout/src")
    with pytest.raises(blackbox.GGUFBlackBoxError, match="must not set PYTHONPATH"):
        blackbox._require_installed_wheel()


def _identity(*, sha256: str = blackbox.FIXTURE_SHA256) -> GGUFArtifactIdentity:
    return GGUFArtifactIdentity(
        artifact_name=f"gguf-sha256-{blackbox.FIXTURE_SHA256}.gguf",
        sha256=sha256,
        byte_length=blackbox.FIXTURE_BYTE_LENGTH,
        gguf_metadata_sha256=blackbox.FIXTURE_METADATA_SHA256,
        tensor_inventory_sha256=blackbox.FIXTURE_TENSOR_INVENTORY_SHA256,
        tokenizer_metadata_sha256=blackbox.FIXTURE_TOKENIZER_METADATA_SHA256,
    )


def _install_provider_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    image_digest: str,
    output_text: str = blackbox.EXPECTED_OUTPUT,
    evidence_errors: tuple[str, ...] = (),
    identity: GGUFArtifactIdentity | None = None,
) -> dict[str, bool]:
    closed = {"value": False}
    observation = SimpleNamespace(records=(SimpleNamespace(output_text=output_text),))
    receipt = valid_receipt(image_digest=image_digest)

    class Session:
        def score(self, _batch: object) -> object:
            return observation

        def runtime_receipt(self) -> object:
            return receipt

        def close(self) -> None:
            closed["value"] = True

    class Provider:
        def open(self, _spec: object, _context: object) -> Session:
            return Session()

    monkeypatch.setattr(blackbox, "_require_installed_wheel", lambda: None)
    monkeypatch.setattr(blackbox, "_validate_fixture", lambda _path: None)
    monkeypatch.setattr(
        blackbox,
        "_sha256_file_unbounded",
        lambda path: (
            blackbox.LLAMA_CPP_SOURCE_SHA256
            if str(path) == blackbox._CONTAINER_SOURCE
            else "a" * 64
        ),
    )
    monkeypatch.setattr(
        blackbox,
        "_normalized_backend_version",
        lambda: (
            "version: 10015 "
            f"({blackbox.LLAMA_CPP_SOURCE_COMMIT}) "
            "built with Test for Linux x86_64"
        ),
    )
    monkeypatch.setattr(
        gguf_identity,
        "read_gguf_artifact_identity",
        lambda _path: identity or _identity(),
    )
    monkeypatch.setattr(llama_cpp, "LlamaCppProvider", Provider)
    monkeypatch.setattr(
        runtime_provider_evidence,
        "encode_scoring_observation",
        lambda _value: blackbox._canonical_json(
            blackbox._expected_observation(schedule_sha256=blackbox.SCHEDULE_SHA256)
        ),
    )
    monkeypatch.setattr(
        runtime_provider_evidence,
        "encode_runtime_provider_receipt",
        lambda _value: blackbox._canonical_json(receipt),
    )
    monkeypatch.setattr(
        runtime_provider_evidence,
        "runtime_provider_evidence_errors",
        lambda **_kwargs: evidence_errors,
    )
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", image_digest)
    return closed


def test_inside_provider_runs_the_authenticated_provider_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "1" * 64
    closed = _install_provider_doubles(monkeypatch, image_digest=digest)

    result = blackbox._inside_provider_result(image_digest=digest)

    assert result["image_digest"] == digest
    assert result["observation"] == blackbox._expected_observation(
        schedule_sha256=blackbox.SCHEDULE_SHA256
    )
    assert result["receipt"] == valid_receipt(image_digest=digest)
    assert closed["value"] is True


@pytest.mark.parametrize(
    "failure", ["digest", "binding", "identity", "source", "output", "evidence"]
)
def test_inside_provider_rejects_each_broken_trust_binding(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    digest = "sha256:" + "2" * 64
    supplied = "bad" if failure == "digest" else digest
    if failure == "binding":
        monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", "sha256:" + "9" * 64)
    identity = _identity(sha256="0" * 64) if failure == "identity" else None
    _install_provider_doubles(
        monkeypatch,
        image_digest=digest,
        output_text="wrong" if failure == "output" else blackbox.EXPECTED_OUTPUT,
        evidence_errors=("broken",) if failure == "evidence" else (),
        identity=identity,
    )
    if failure == "binding":
        monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", "sha256:" + "9" * 64)
    if failure == "source":
        monkeypatch.setattr(blackbox, "_sha256_file_unbounded", lambda _path: "0" * 64)

    with pytest.raises(blackbox.GGUFBlackBoxError):
        blackbox._inside_provider_result(image_digest=supplied)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (None, None),
        ("schedule_digest", "schedule does not match"),
        ("schedule_clobber", "clobbered its schedule"),
        ("binding_result", "binding does not match"),
        ("directed_binding", "bindings are not exact"),
        ("policy", "policy is not exactly directed"),
        ("pair_result", "paired receipt did not pass"),
        ("pair_clobber", "clobbered its paired receipt"),
        ("inventory", "artifact inventory is incomplete"),
        ("receipt_determinism", "receipts are not deterministic"),
    ],
)
def test_inside_cli_journey_enforces_the_complete_installed_cli_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
    expected: str | None,
) -> None:
    digest = "sha256:" + "3" * 64
    work = tmp_path / "private-work"
    cli = tmp_path / "invarlock"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    cli.chmod(0o700)
    executable = tmp_path / "llama-completion"
    executable.write_bytes(b"binary")
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"source")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    monkeypatch.setattr(blackbox, "_CONTAINER_WORK_ROOT", str(work))
    monkeypatch.setattr(blackbox, "_CONTAINER_CLI", str(cli))
    monkeypatch.setattr(blackbox, "_CONTAINER_EXECUTABLE", str(executable))
    monkeypatch.setattr(blackbox, "_CONTAINER_SOURCE", str(source))
    monkeypatch.setattr(blackbox, "_CONTAINER_MODEL", str(model))
    monkeypatch.setattr(blackbox, "_normalized_backend_version", lambda: "pinned")
    for name in (
        "INVARLOCK_ALLOW_HOST_EXECUTION",
        "INVARLOCK_ALLOW_NETWORK",
        "INVARLOCK_ALLOW_REMOTE_CODE",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
        "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE",
    ):
        monkeypatch.setenv(name, "0")

    expected_binding = {
        "artifact_format": "gguf",
        "artifact_identity_sha256": blackbox.ARTIFACT_IDENTITY_SHA256,
        "execution_settings_sha256": blackbox.CLI_EXECUTION_SETTINGS_SHA256,
        "outer_image_digest": digest,
        "provider_name": "llama_cpp",
    }
    policy_digest = "sha256:" + "4" * 64

    def installed_cli(
        arguments: tuple[str, ...],
        *,
        expected_format: str,
        expect_success: bool = True,
        timeout_seconds: int = 240,
    ) -> dict[str, object]:
        del expected_format, timeout_seconds
        command = arguments[0]
        if not expect_success:
            if failure == "schedule_clobber" and command == "build-schedule":
                output = Path(arguments[arguments.index("--out") + 1])
                write_json(output, {"clobbered": True})
            elif failure == "pair_clobber" and command == "verify-pair":
                receipt = Path(arguments[arguments.index("--receipt") + 1])
                write_json(receipt, {"clobbered": True})
            elif failure == "receipt_determinism" and command == "verify-pair":
                subject = Path(arguments[arguments.index("--subject") + 1])
                write_json(
                    subject / "runtime-provider.receipt.json", {"different": True}
                )
            return {"ok": False}
        output = (
            Path(arguments[arguments.index("--out") + 1])
            if "--out" in arguments
            else None
        )
        if command == "build-schedule":
            assert output is not None
            write_json(output, exact_schedule())
            return {
                "schedule_sha256": (
                    "0" * 64
                    if failure == "schedule_digest"
                    else blackbox.CLI_SCHEDULE_SHA256
                )
            }
        if command == "prepare-binding":
            assert output is not None
            binding = expected_binding
            if failure == "directed_binding" and output.name == "subject-binding.json":
                binding = expected_binding | {"provider_name": "wrong"}
            write_json(output, binding)
            return {
                "artifact_identity_sha256": (
                    "0" * 64
                    if failure == "binding_result"
                    else blackbox.ARTIFACT_IDENTITY_SHA256
                ),
                "execution_settings_sha256": blackbox.CLI_EXECUTION_SETTINGS_SHA256,
            }
        if command == "build-policy":
            assert output is not None
            claim = {
                "schedule_sha256": blackbox.CLI_SCHEDULE_SHA256,
                "baseline": expected_binding,
                "subject": expected_binding,
            }
            if failure == "policy":
                claim["subject"] = {"provider_name": "wrong"}
            write_json(
                output,
                {"behavioral_claim": claim, "policy_digest": policy_digest},
            )
            return {"policy_digest": policy_digest}
        if command == "run-side":
            assert output is not None
            role = arguments[arguments.index("--role") + 1]
            write_side_bundle(output, role=role, image_digest=digest)
            return {"ok": True}
        if command == "verify-pair":
            receipt = Path(arguments[arguments.index("--receipt") + 1])
            baseline = Path(arguments[arguments.index("--baseline") + 1])
            subject = Path(arguments[arguments.index("--subject") + 1])
            write_json(
                receipt,
                {
                    "baseline": blackbox._expected_side_bindings(baseline),
                    "baseline_score": 1.0,
                    "claim_set": "invarlock-runtime-behavioral-regression-v1",
                    "format_version": "invarlock/runtime-behavioral-claim-receipt-v1",
                    "metric": "exact_match",
                    "policy_digest": policy_digest,
                    "regression": 0.0,
                    "schedule_sha256": blackbox.CLI_SCHEDULE_SHA256,
                    "subject": blackbox._expected_side_bindings(subject),
                    "subject_score": 1.0,
                    "verdict": "pass",
                },
            )
            if failure == "inventory":
                write_json(baseline / "unexpected.json", {"unexpected": True})
            return {
                "baseline_score": 1.0,
                "subject_score": 0.0 if failure == "pair_result" else 1.0,
                "regression": 0.0,
            }
        raise AssertionError(command)

    monkeypatch.setattr(blackbox, "_run_installed_cli", installed_cli)

    if expected is not None:
        with pytest.raises(blackbox.GGUFBlackBoxError, match=expected):
            blackbox._inside_cli_journey(image_digest=digest)
        return

    result = blackbox._inside_cli_journey(image_digest=digest)

    assert result["format_version"] == blackbox.CLI_JOURNEY_FORMAT
    assert result["portable_artifact_count"] == 17
    assert result["verification"] == {
        "baseline_score": 1.0,
        "regression": 0.0,
        "subject_score": 1.0,
        "verdict": "pass",
    }


def test_cli_journey_requires_fail_closed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "1")
    with pytest.raises(blackbox.GGUFBlackBoxError, match="fail-closed"):
        blackbox._inside_cli_journey(image_digest="sha256:" + "5" * 64)


def test_canonical_private_writer_and_unbounded_hash_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private.json"
    blackbox._write_canonical_new(path, {"ok": True})
    assert path.read_bytes() == b'{"ok":true}'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(blackbox.GGUFBlackBoxError, match="could not be created"):
        blackbox._write_canonical_new(path, {"ok": True})
    assert (
        blackbox._sha256_file_unbounded(path)
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(blackbox.GGUFBlackBoxError, match="cannot be read"):
        blackbox._sha256_file_unbounded(tmp_path / "missing")


def test_inside_result_combines_provider_and_cli_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = {"format_version": blackbox.RESULT_FORMAT}
    journey = {"format_version": blackbox.CLI_JOURNEY_FORMAT}
    monkeypatch.setattr(blackbox, "_inside_provider_result", lambda **_kwargs: provider)
    monkeypatch.setattr(blackbox, "_inside_cli_journey", lambda **_kwargs: journey)
    assert blackbox._inside_result(image_digest="sha256:" + "6" * 64) == {
        "format_version": blackbox.RESULT_FORMAT,
        "cli_journey": journey,
    }


def test_main_emits_canonical_success_for_host_and_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(blackbox, "_run_host", lambda **_kwargs: {"status": "host"})
    assert blackbox.main(["--model", str(tmp_path / "model.gguf")]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "host"}

    monkeypatch.setattr(
        blackbox, "_inside_result", lambda **_kwargs: {"status": "container"}
    )
    assert (
        blackbox.main(["--inside-container", "--image-digest", "sha256:" + "7" * 64])
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"status": "container"}


def test_main_rejects_mixed_host_and_container_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        blackbox.main(
            [
                "--inside-container",
                "--image-digest",
                "sha256:" + "8" * 64,
                "--model",
                str(tmp_path / "model.gguf"),
            ]
        )
        == 2
    )
    assert "inside-container invocation is invalid" in capsys.readouterr().err
    assert blackbox.main(["--image-digest", "sha256:" + "8" * 64]) == 2
    assert "--model is required" in capsys.readouterr().err
