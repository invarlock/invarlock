from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tests.scripts._tensorrt_llm_fixture_support import (
    canary_payload as _canary_payload,
)
from tests.scripts._tensorrt_llm_fixture_support import (
    fixture,
)
from tests.scripts._tensorrt_llm_fixture_support import (
    identity as _identity,
)
from tests.scripts._tensorrt_llm_fixture_support import (
    inspection as _inspection,
)
from tests.scripts._tensorrt_llm_fixture_support import (
    valid_manifest as _valid_manifest,
)

_EXPECTED_ARTIFACT_SHA256 = fixture.artifact_identity_sha256(_identity())


@pytest.fixture(autouse=True)
def _stub_exact_base_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fixture,
        "_validate_hardware",
        lambda **_kwargs: (
            ("GPU-01234567-89ab-cdef-0123-456789abcdef", "9.0"),
            ("GPU-fedcba98-7654-3210-fedc-ba9876543210", "9.0"),
        ),
    )


def test_image_inspection_requires_pinned_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fixture, "_run_captured", lambda *_args, **_kwargs: (0, _inspection(), b"")
    )
    assert fixture._inspect_image("docker", "candidate:tag") == "sha256:" + "a" * 64

    payload = json.loads(_inspection())
    payload[0]["Config"]["Labels"]["dev.invarlock.tensorrt-llm.version"] = "latest"
    monkeypatch.setattr(
        fixture,
        "_run_captured",
        lambda *_args, **_kwargs: (0, json.dumps(payload).encode(), b""),
    )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="labels"):
        fixture._inspect_image("docker", "candidate:tag")


def test_candidate_image_build_uses_fixed_argv_and_hard_pinned_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **_kwargs: object):
        commands.append(command)
        return (0, b"", b"")

    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(fixture, "_run_captured", run)
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: digest)
    result = fixture.build_candidate_image(
        engine="docker",
        image="invarlock-runtime:tensorrt-llm-local-candidate",
        source_date_epoch="1784073600",
    )
    repository = Path(fixture.__file__).resolve().parents[2]
    assert commands == [
        (
            "docker",
            "buildx",
            "build",
            "--load",
            "--provenance=false",
            "--build-arg",
            "SOURCE_DATE_EPOCH=1784073600",
            "-f",
            str(repository / "runtime" / "Dockerfile.tensorrt-llm"),
            "-t",
            "invarlock-runtime:tensorrt-llm-local-candidate",
            str(repository),
        )
    ]
    assert result == {
        "candidate_image_digest": digest,
        "format_version": fixture._boundary.BUILD_FORMAT,
        "ok": True,
    }
    assert fixture._boundary.BASE_IMAGE not in commands[0]


@pytest.mark.parametrize(
    ("engine", "image", "epoch"),
    [
        ('docker"; command; #', "candidate:tag", "1784073600"),
        ("docker", 'candidate:tag"; command; #', "1784073600"),
        ("docker", "candidate:tag", '1"; command; #'),
    ],
)
def test_candidate_image_build_rejects_boundary_injection_before_process_start(
    engine: str, image: str, epoch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fixture,
        "_run_captured",
        lambda *_a, **_k: pytest.fail("process boundary was reached"),
    )
    with pytest.raises(fixture.TensorRTLLMFixtureError):
        fixture.build_candidate_image(
            engine=engine,
            image=image,
            source_date_epoch=epoch,
        )


def test_candidate_smoke_uses_the_inspected_digest_and_validated_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: digest)
    monkeypatch.setattr(
        fixture,
        "_run_captured",
        lambda command, **_k: commands.append(command) or (0, b"", b""),
    )
    result = fixture.smoke_candidate_image(
        engine="docker", image="candidate:tag", selector="device=0"
    )
    assert commands[0][:7] == (
        "docker",
        "run",
        "--rm",
        "--gpus",
        "device=0",
        "--network",
        "none",
    )
    assert digest in commands[0]
    assert "candidate:tag" not in commands[0]
    entrypoint = commands[0].index("--entrypoint")
    assert commands[0][entrypoint : entrypoint + 2] == (
        "--entrypoint",
        "/bin/bash",
    )
    assert all(
        value in commands[0] for value in fixture._boundary.VENDOR_CACHE_ENV_ARGS
    )
    assert result["candidate_image_digest"] == digest
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="selector"):
        fixture.smoke_candidate_image(
            engine="docker", image="candidate:tag", selector='all"; command; #'
        )


def test_candidate_build_and_smoke_fail_closed_on_container_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fixture, "_run_captured", lambda *_args, **_kwargs: (1, b"", b"failed")
    )
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_args: "sha256:" + "a" * 64)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="image build failed"):
        fixture.build_candidate_image(
            engine="docker",
            image="candidate:tag",
            source_date_epoch="1784073600",
        )
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="smoke failed"):
        fixture.smoke_candidate_image(
            engine="docker", image="candidate:tag", selector="device=0"
        )


def test_build_fixture_runs_two_builds_and_two_cross_gpu_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    output = Path("result")
    absolute_output = tmp_path / output
    calls: list[tuple[str, str]] = []
    owned_inputs: list[tuple[Path, Path]] = []

    monkeypatch.setattr(fixture, "_inspect_image", lambda *_args: "sha256:" + "a" * 64)

    def fake_build_one(
        *,
        selector: str,
        output: Path,
        image: str,
        model: Path,
        worker: Path,
        **_kwargs: object,
    ) -> None:
        assert image == "sha256:" + "a" * 64
        assert output.is_absolute()
        assert model.is_absolute()
        assert worker.is_absolute()
        owned_inputs.append((model, worker))
        calls.append(("build", selector))
        output.mkdir()
        engine = output / "engine"
        engine.mkdir()
        (engine / "config.json").write_text("{}", encoding="utf-8")
        (engine / "rank0.engine").write_bytes(selector.encode())
        (output / "tokenizer.json").write_bytes(b"tokenizer")

    monkeypatch.setattr(fixture, "_build_one", fake_build_one)
    identities = iter(
        (_identity(), replace(_identity(), engine_bundle_tree_sha256="6" * 64))
    )
    monkeypatch.setattr(
        fixture,
        "read_tensorrt_llm_artifact_identity",
        lambda *_args, **_kwargs: next(identities),
    )

    def fake_probe_one(*, selector: str, image: str, **_kwargs: object) -> str:
        assert image == "sha256:" + "a" * 64
        calls.append(("probe", selector))
        return "token"

    monkeypatch.setattr(fixture, "_probe_one", fake_probe_one)
    manifest = fixture.build_fixture(
        engine="docker",
        image="candidate",
        model=model,
        output=output,
        selectors=("device=0", "device=1"),
        expected_model_inventory_sha256=fixture._model_inventory_sha256(model),
    )
    assert manifest["engine_byte_reproduction"] == "different"
    assert (
        manifest["expected_output_sha256"]
        == fixture.hashlib.sha256(b"token").hexdigest()
    )
    assert sorted(calls) == sorted(
        [
            ("build", "device=0"),
            ("build", "device=1"),
            ("probe", "device=0"),
            ("probe", "device=1"),
        ]
    )
    assert all(
        model_path == absolute_output / ".inputs" / "model"
        for model_path, _ in owned_inputs
    )
    assert all(
        worker_path == absolute_output / ".inputs" / "worker.py"
        for _, worker_path in owned_inputs
    )
    persisted = json.loads((absolute_output / "fixture-manifest.json").read_text())
    assert persisted == manifest
    serialized = json.dumps(manifest)
    assert str(tmp_path) not in serialized
    assert "device=0" not in serialized


def test_build_fixture_rejects_mismatched_tokenizers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_args: "sha256:" + "a" * 64)

    def fake_build_one(*, selector: str, output: Path, **_kwargs: object) -> None:
        output.mkdir()
        (output / "tokenizer.json").write_bytes(selector.encode())

    monkeypatch.setattr(fixture, "_build_one", fake_build_one)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="tokenizer"):
        fixture.build_fixture(
            engine="docker",
            image="candidate",
            model=model,
            output=tmp_path / "result",
            selectors=("device=0", "device=1"),
            expected_model_inventory_sha256=fixture._model_inventory_sha256(model),
        )


def test_low_level_build_probe_and_canary_commands_are_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_path = tmp_path / "worker.py"
    worker_path.write_text("pass", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    build_parent = tmp_path / "builds"
    build_parent.mkdir()
    commands: list[list[str]] = []
    image_digest = "sha256:" + "a" * 64

    def fake_run(command: list[str], **_kwargs: object):
        commands.append(list(command))
        if "build" in command:
            return (
                0,
                fixture._canonical_json(
                    {
                        "backend_version": fixture.BACKEND_VERSION,
                        "format_version": fixture.BUILD_RESULT_FORMAT,
                        "ok": True,
                    }
                ),
                b"",
            )
        if "probe" in command:
            return (
                0,
                fixture._canonical_json(
                    {
                        "format_version": fixture.PROBE_RESULT_FORMAT,
                        "ok": True,
                        "output_text": "token",
                    }
                ),
                b"",
            )
        return (0, fixture._canonical_json(_canary_payload()), b"")

    monkeypatch.setattr(fixture, "_run_captured", fake_run)
    fixture._build_one(
        engine="docker",
        image=image_digest,
        selector="device=0",
        worker=worker_path,
        model=model,
        output=build_parent / "gpu-0",
    )
    frozen = tmp_path / "frozen"
    (frozen / "engine").mkdir(parents=True)
    (frozen / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert (
        fixture._probe_one(
            engine="docker",
            image=image_digest,
            selector="device=1",
            worker=worker_path,
            fixture=frozen,
        )
        == "token"
    )
    manifest = {
        "expected_output_sha256": "7" * 64,
        "selected_engine_identity": {"engine_bundle_tree_sha256": "1" * 64},
        "tokenizer_sha256": "4" * 64,
    }
    assert (
        fixture._canary_one(
            engine="docker",
            image=image_digest,
            image_digest=image_digest,
            selector="device=1",
            fixture=frozen,
            manifest=manifest,
            expected_artifact_identity_sha256=_EXPECTED_ARTIFACT_SHA256,
        )["ok"]
        is True
    )
    assert all("--privileged" not in command for command in commands)
    assert all(
        image_digest in command and "candidate" not in command for command in commands
    )
    assert all(
        command[command.index("--gpus") + 1] in {"device=0", "device=1"}
        for command in commands
    )
    assert all(
        command[command.index("--entrypoint") + 1] == "/bin/bash"
        for command in commands
    )
    assert all(
        command[command.index("-c") : command.index("-c") + 3]
        == ["-c", 'exec "$@"', "--"]
        for command in commands
    )
    assert all(
        all(value in command for value in fixture._boundary.VENDOR_CACHE_ENV_ARGS)
        for command in commands
    )


def test_qualify_two_gpu_reauthenticates_and_writes_path_free_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    frozen = root / "fixture"
    (frozen / "engine").mkdir(parents=True)
    (frozen / "engine" / "config.json").write_text("{}", encoding="utf-8")
    (frozen / "engine" / "rank0.engine").write_bytes(b"engine")
    (frozen / "tokenizer.json").write_bytes(b"tokenizer")
    tokenizer_sha = fixture._sha256_file(frozen / "tokenizer.json")
    identity = replace(_identity(), tokenizer_metadata_sha256=tokenizer_sha)
    manifest = _valid_manifest(identity)
    (root / "fixture-manifest.json").write_bytes(fixture._canonical_json(manifest))
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_args: "sha256:" + "a" * 64)
    monkeypatch.setattr(
        fixture, "read_tensorrt_llm_artifact_identity", lambda *_a, **_k: identity
    )
    canary = {"format_version": fixture.CANARY_FORMAT, "ok": True}

    def fake_canary(**kwargs: object):
        assert kwargs["image"] == "sha256:" + "a" * 64
        return canary

    monkeypatch.setattr(fixture, "_canary_one", fake_canary)
    output = tmp_path / "qualification.json"
    summary = fixture.qualify_two_gpu(
        engine="docker",
        image="candidate",
        fixture_root=root,
        output=output,
        selectors=("device=0", "device=1"),
    )
    assert summary["gpu_count"] == 2
    assert summary["ok"] is True
    assert json.loads(output.read_text()) == summary
    assert str(tmp_path) not in output.read_text()


def test_run_captured_bounds_and_translates_process_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, stdout, stderr = fixture._run_captured(
        ("/bin/echo", "ok"), timeout_seconds=2
    )
    assert (status, stdout, stderr) == (0, b"ok\n", b"")
    monkeypatch.setattr(fixture, "_MAX_CAPTURE", 1)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="output limit"):
        fixture._run_captured(("/bin/echo", "long"), timeout_seconds=2)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="could not be started"):
        fixture._run_captured(("/not/a/real/executable",), timeout_seconds=2)

    class TimeoutProcess:
        def wait(self, **_kwargs: object) -> int:
            raise subprocess.TimeoutExpired("tool", 1)

        def kill(self) -> None:
            return None

    process = TimeoutProcess()
    calls = 0

    def wait(**_kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired("tool", 1)
        return -9

    process.wait = wait
    monkeypatch.setattr(fixture.subprocess, "Popen", lambda *_a, **_k: process)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="timed out"):
        fixture._run_captured(("tool",), timeout_seconds=1)


@pytest.mark.parametrize(
    "response",
    [
        (1, b"", b"bad"),
        (0, b"not-json", b""),
        (0, b"[]", b""),
        (0, b'[{"Id":"latest","Config":{"Labels":{}}}]', b""),
    ],
)
def test_image_inspection_rejects_invalid_results(
    response: tuple[int, bytes, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fixture, "_run_captured", lambda *_a, **_k: response)
    with pytest.raises(fixture.TensorRTLLMFixtureError):
        fixture._inspect_image("docker", "candidate")


def test_low_level_helpers_reject_failed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_path = tmp_path / "worker.py"
    worker_path.write_text("pass", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    parent = tmp_path / "builds"
    parent.mkdir()
    monkeypatch.setattr(fixture, "_run_captured", lambda *_a, **_k: (2, b"", b"bad"))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="build failed"):
        fixture._build_one(
            engine="docker",
            image="candidate",
            selector="device=0",
            worker=worker_path,
            model=model,
            output=parent / "gpu-0",
        )
    monkeypatch.setattr(fixture, "_run_captured", lambda *_a, **_k: (0, b"{}", b""))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="build result"):
        fixture._build_one(
            engine="docker",
            image="candidate",
            selector="device=0",
            worker=worker_path,
            model=model,
            output=parent / "gpu-0",
        )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    monkeypatch.setattr(fixture, "_run_captured", lambda *_a, **_k: (0, b"{}", b"bad"))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="probe failed"):
        fixture._probe_one(
            engine="docker",
            image="candidate",
            selector="device=0",
            worker=worker_path,
            fixture=frozen,
        )
    monkeypatch.setattr(fixture, "_run_captured", lambda *_a, **_k: (0, b"{}", b""))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="probe result"):
        fixture._probe_one(
            engine="docker",
            image="candidate",
            selector="device=0",
            worker=worker_path,
            fixture=frozen,
        )
    source = tmp_path / "source"
    source.write_bytes(b"x")
    destination = tmp_path / "destination"
    destination.write_bytes(b"exists")
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="copied safely"):
        fixture._copy_new(source, destination)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="created safely"):
        fixture._write_new_json(destination, {})


def test_build_fixture_rejects_existing_output_and_cross_gpu_output_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(fixture, "_inspect_image", lambda *_a: "sha256:" + "a" * 64)
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="must be new"):
        fixture.build_fixture(
            engine="docker",
            image="candidate",
            model=model,
            output=existing,
            selectors=("device=0", "device=1"),
            expected_model_inventory_sha256=fixture._model_inventory_sha256(model),
        )

    def fake_build(*, output: Path, **_kwargs: object) -> None:
        output.mkdir()
        engine = output / "engine"
        engine.mkdir()
        (engine / "config.json").write_text("{}", encoding="utf-8")
        (engine / "rank0.engine").write_bytes(b"engine")
        (output / "tokenizer.json").write_bytes(b"tokenizer")

    monkeypatch.setattr(fixture, "_build_one", fake_build)
    monkeypatch.setattr(
        fixture, "read_tensorrt_llm_artifact_identity", lambda *_a, **_k: _identity()
    )
    results = iter(("first", "second"))
    monkeypatch.setattr(fixture, "_probe_one", lambda **_kwargs: next(results))
    with pytest.raises(fixture.TensorRTLLMFixtureError, match="differs across GPUs"):
        fixture.build_fixture(
            engine="docker",
            image="candidate",
            model=model,
            output=tmp_path / "new",
            selectors=("device=0", "device=1"),
            expected_model_inventory_sha256=fixture._model_inventory_sha256(model),
        )


def test_main_dispatches_both_commands_and_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(fixture, "build_fixture", lambda **_k: {"ok": True})
    common = ["--image", "candidate", "--gpu-0", "device=0", "--gpu-1", "device=1"]
    assert (
        fixture.main(
            [
                *common,
                "build-fixture",
                "--model",
                "/model",
                "--output",
                "/out",
                "--expected-model-inventory-sha256",
                "1" * 64,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    monkeypatch.setattr(fixture, "promote_candidate", lambda **_k: {"ok": True})
    assert (
        fixture.main(
            [
                "--image",
                "candidate:qualified",
                "promote",
                "--qualification-summary",
                "/qualification.json",
                "--stable-tag",
                "invarlock-runtime:tensorrt-llm-local",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    monkeypatch.setattr(fixture, "qualify_two_gpu", lambda **_k: {"ok": True})
    assert (
        fixture.main(
            [
                *common,
                "qualify-two-gpu",
                "--fixture-root",
                "/fixture",
                "--output",
                "/out",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"ok": True}

    def fail(**_kwargs: object):
        raise fixture.TensorRTLLMFixtureError("closed")

    monkeypatch.setattr(fixture, "qualify_two_gpu", fail)
    assert (
        fixture.main(
            [
                *common,
                "qualify-two-gpu",
                "--fixture-root",
                "/fixture",
                "--output",
                "/out",
            ]
        )
        == 2
    )
    assert "failed" in capsys.readouterr().err


def test_main_consumes_make_exported_defaults_without_shell_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    values = {
        fixture._boundary.ENV_CONTAINER_ENGINE: "docker",
        fixture._boundary.ENV_IMAGE: "candidate:qualified",
        fixture._boundary.ENV_STABLE_TAG: "invarlock-runtime:tensorrt-llm-local",
        fixture._boundary.ENV_GPU_0: "device=0",
        fixture._boundary.ENV_GPU_1: "device=1",
        fixture._boundary.ENV_MODEL: str(tmp_path / "model"),
        fixture._boundary.ENV_FIXTURE_ROOT: str(tmp_path / "fixture"),
        fixture._boundary.ENV_MODEL_INVENTORY: "1" * 64,
        fixture._boundary.ENV_SOURCE_DATE_EPOCH: "1784073600",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        fixture,
        "preflight_flow",
        lambda **kwargs: calls.append(("preflight", kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        fixture,
        "build_candidate_image",
        lambda **kwargs: calls.append(("image", kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        fixture,
        "build_fixture",
        lambda **kwargs: calls.append(("fixture", kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        fixture,
        "smoke_candidate_image",
        lambda **kwargs: calls.append(("smoke", kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        fixture,
        "qualify_two_gpu",
        lambda **kwargs: calls.append(("qualify", kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        fixture,
        "promote_candidate",
        lambda **kwargs: calls.append(("promote", kwargs)) or {"ok": True},
    )
    for command in (
        "preflight",
        "build-image",
        "smoke-image",
        "build-fixture",
        "qualify-two-gpu",
        "promote",
    ):
        assert fixture.main([command]) == 0
        assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert calls == [
        (
            "preflight",
            {
                "engine": "docker",
                "image": "candidate:qualified",
                "stable_tag": "invarlock-runtime:tensorrt-llm-local",
                "source_date_epoch": "1784073600",
                "smoke_selector": "all",
                "model": tmp_path / "model",
                "output": tmp_path / "fixture",
                "selectors": ("device=0", "device=1"),
                "expected_model_inventory_sha256": "1" * 64,
            },
        ),
        (
            "image",
            {
                "engine": "docker",
                "image": "candidate:qualified",
                "source_date_epoch": "1784073600",
            },
        ),
        (
            "smoke",
            {
                "engine": "docker",
                "image": "candidate:qualified",
                "selector": "all",
            },
        ),
        (
            "fixture",
            {
                "engine": "docker",
                "image": "candidate:qualified",
                "model": tmp_path / "model",
                "output": tmp_path / "fixture",
                "selectors": ("device=0", "device=1"),
                "expected_model_inventory_sha256": "1" * 64,
            },
        ),
        (
            "qualify",
            {
                "engine": "docker",
                "image": "candidate:qualified",
                "fixture_root": tmp_path / "fixture",
                "output": tmp_path / "fixture" / "qualification-summary.json",
                "selectors": ("device=0", "device=1"),
            },
        ),
        (
            "promote",
            {
                "engine": "docker",
                "image": "candidate:qualified",
                "qualification_summary": tmp_path
                / "fixture"
                / "qualification-summary.json",
                "stable_tag": "invarlock-runtime:tensorrt-llm-local",
            },
        ),
    ]
