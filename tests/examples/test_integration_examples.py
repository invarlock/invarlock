from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from examples.integrations import launch
from examples.integrations import run as integration

ZERO_DIGEST = "sha256:" + ("0" * 64)


def test_hf_preparation_creates_closed_distinct_transaction(tmp_path: Path) -> None:
    paths, anchors = integration._prepare_workspace(
        tmp_path / "hf",
        integration="hf-transformers",
        runtime_image_digest=ZERO_DIGEST,
    )

    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    records = (
        (paths.evaluation / "inputs/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(records) == 50
    assert request["comparison"]["dataset"]["name"] == "hf-transformers-smoke"
    assert request["comparison"]["baseline"]["runtime"]["provider"] == "hf_transformers"
    assert request["comparison"]["subject"]["runtime"]["provider"] == "hf_transformers"
    assert anchors["baseline_artifact_digest"] != anchors["subject_artifact_digest"]
    assert anchors["baseline_runtime_digest"] == ZERO_DIGEST
    assert paths.evidence_key.stat().st_mode & 0o777 == 0o600
    assert paths.verifier_key.stat().st_mode & 0o777 == 0o600


def test_peft_preparation_trains_serializes_reloads_and_merges(tmp_path: Path) -> None:
    paths, anchors = integration._prepare_workspace(
        tmp_path / "peft",
        integration="peft-lora",
        runtime_image_digest=ZERO_DIGEST,
    )

    summary = json.loads(
        (paths.root / "upstream/peft-summary.json").read_text(encoding="utf-8")
    )
    assert summary["library"] == "peft"
    assert summary["library_version"] == "0.19.1"
    assert summary["final_loss"] < summary["initial_loss"]
    assert (paths.root / "upstream/peft-adapter/adapter_model.safetensors").is_file()
    assert anchors["baseline_artifact_digest"] != anchors["subject_artifact_digest"]


def test_preparation_rejects_existing_and_unknown_workspace(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="workspace already exists"):
        integration._prepare_workspace(
            existing,
            integration="hf-transformers",
            runtime_image_digest=ZERO_DIGEST,
        )
    paths = integration._paths(tmp_path / "unused")
    with pytest.raises(RuntimeError, match="unsupported integration"):
        integration._create_checkpoints(paths, "unknown")


def test_execute_invokes_public_commands_and_checks_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = integration._paths(tmp_path)
    report = paths.evidence / "reports/evaluation.report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({"comparison": {"value": 0.75}, "verdict": "pass"}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(integration, "_run", lambda command: commands.append(command))

    integration._execute(
        paths,
        container_engine="docker",
        runtime_image="example:current",
        runtime_image_digest=ZERO_DIGEST,
        runtime_device="cpu",
    )

    assert [command[3] for command in commands] == ["evaluate", "verify", "report"]
    assert "--trust-profile" in commands[1]
    assert "--html" in commands[2]

    report.write_text(
        json.dumps({"comparison": {"value": "bad"}, "verdict": "pass"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not produce a passing ratio"):
        integration._execute(
            paths,
            container_engine="docker",
            runtime_image="example:current",
            runtime_image_digest=ZERO_DIGEST,
            runtime_device="cpu",
        )


def test_command_runner_surfaces_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 3, stdout="out\n", stderr="bad\n"
        ),
    )
    with pytest.raises(RuntimeError, match="status 3"):
        integration._run(["false"])


def test_run_main_prepare_only_and_input_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = integration.main(
        [
            "hf-transformers",
            "--workspace",
            str(tmp_path / "prepared"),
            "--runtime-image-digest",
            ZERO_DIGEST,
            "--prepare-only",
        ]
    )
    assert prepared == 0

    existing = tmp_path / "existing"
    existing.mkdir()
    assert (
        integration.main(
            [
                "hf-transformers",
                "--workspace",
                str(existing),
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--prepare-only",
            ]
        )
        == 2
    )
    with pytest.raises(SystemExit, match="full execution requires"):
        integration.main(
            [
                "hf-transformers",
                "--workspace",
                str(tmp_path / "missing-image"),
                "--runtime-image-digest",
                ZERO_DIGEST,
            ]
        )


def test_launch_helpers_require_committed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = iter(("a" * 40, ""))
    monkeypatch.setattr(launch, "_git", lambda *args: next(replies))
    assert launch._require_committed_checkout(tmp_path) == "a" * 40

    replies = iter(("b" * 40, " M tracked.py"))
    monkeypatch.setattr(launch, "_git", lambda *args: next(replies))
    with pytest.raises(RuntimeError, match="commit or stash"):
        launch._require_committed_checkout(tmp_path)


def test_runtime_image_builds_from_authenticated_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    commands: list[list[str]] = []
    monkeypatch.setattr(launch, "_require_committed_checkout", lambda _repo: "c" * 40)
    monkeypatch.setattr(launch, "_git", lambda *args: "1234567890")

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "qualification_source.py" in " ".join(command):
            output = json.dumps({"source_bundle_sha256": ZERO_DIGEST})
        elif command[1:4] == ["image", "inspect", "--format"]:
            output = "sha256:" + ("d" * 64)
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(launch, "_run", fake_run)
    image, digest = launch._runtime_image(
        repository=repository,
        build_root=build,
        container_engine="docker",
    )
    assert image == "invarlock-example-runtime:" + ("c" * 12)
    assert digest == "sha256:" + ("d" * 64)
    assert any("authenticated_runtime_build.py" in " ".join(item) for item in commands)

    def invalid_inspect(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        if "qualification_source.py" in " ".join(command):
            output = json.dumps({"source_bundle_sha256": ZERO_DIGEST})
        elif command[1:4] == ["image", "inspect", "--format"]:
            output = "not-a-digest"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(launch, "_run", invalid_inspect)
    with pytest.raises(RuntimeError, match="sha256 image ID"):
        launch._runtime_image(
            repository=repository,
            build_root=build,
            container_engine="docker",
        )


def test_launch_main_dispatches_prepare_and_full_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, cwd: Path, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(launch, "_run", fake_run)
    assert (
        launch.main(
            [
                "hf-transformers",
                "--prepare-only",
                "--workspace",
                str(tmp_path / "prepare"),
            ]
        )
        == 0
    )
    assert "--prepare-only" in commands[-1]

    monkeypatch.setattr(
        launch,
        "_runtime_image",
        lambda **kwargs: ("example:current", "sha256:" + ("e" * 64)),
    )
    assert launch.main(["peft-lora", "--workspace", str(tmp_path / "full")]) == 0
    assert "example:current" in commands[-1]

    existing = tmp_path / "existing"
    existing.mkdir()
    assert launch.main(["hf-transformers", "--workspace", str(existing)]) == 2


def test_launch_runner_reports_subprocess_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 7, stdout="", stderr="failed"
        ),
    )
    with pytest.raises(RuntimeError, match="status 7"):
        launch._run(["bad"], cwd=tmp_path)
