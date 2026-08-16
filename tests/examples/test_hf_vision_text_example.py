from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from PIL import Image

from examples.integrations import hf_vision_text, launch
from invarlock.core.runtime_provider.behavioral_schedule import (
    load_runtime_behavioral_schedule,
)

ZERO_DIGEST = "sha256:" + ("0" * 64)


def test_fixture_is_deterministic_single_frame_png(tmp_path: Path) -> None:
    first = hf_vision_text.tutorial_image_png()
    second = hf_vision_text.tutorial_image_png()

    assert first == second
    assert hashlib.sha256(first).hexdigest() == hf_vision_text.IMAGE_SHA256
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(first)
    with Image.open(image_path) as image:
        assert image.format == "PNG"
        assert image.size == (96, 96)
        assert image.n_frames == 1
        assert image.getpixel((12, 12)) == (220, 38, 38)
        assert image.getpixel((84, 12)) == (38, 92, 220)
        assert image.getpixel((12, 84)) == (34, 160, 78)
        assert image.getpixel((84, 84)) == (244, 196, 38)


def test_prepare_only_writes_closed_authenticated_tutorial(tmp_path: Path) -> None:
    paths, anchors = hf_vision_text.prepare_workspace(
        tmp_path / "vision",
        runtime_image_digest=ZERO_DIGEST,
        materialize_models=False,
    )

    request = yaml.safe_load(paths.request.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in paths.records.read_text(encoding="utf-8").splitlines()
    ]
    policy = json.loads(paths.policy.read_text(encoding="utf-8"))
    schedule = load_runtime_behavioral_schedule(paths.schedule)
    trust = json.loads(paths.trusted_inputs.read_text(encoding="utf-8"))

    assert request["format_version"] == "invarlock/evaluation-request-v1"
    assert request["comparison"]["task"] == "vision_text_generation"
    assert request["comparison"]["metric"] == "exact_match"
    assert request["comparison"]["dataset"]["content_role"] == "image"
    assert request["comparison"]["baseline"]["runtime"]["provider"] == (
        "hf_vision_text"
    )
    assert request["comparison"]["subject"]["runtime"]["provider"] == ("hf_vision_text")
    for role, profile in hf_vision_text.MODEL_PROFILES.items():
        side = request["comparison"][role]
        assert side["artifact"]["model_id"] == profile.model_id
        assert side["artifact"]["locator"] == (
            f"hf://{profile.model_id}@{profile.revision}"
        )
        assert side["runtime"]["settings"]["checkpoint_tree_sha256"] == (
            profile.checkpoint_tree_sha256
        )
        model_root = paths.evaluation / "models" / role
        assert model_root.is_dir()
        assert list(model_root.iterdir()) == [model_root / ".model_id"]
        marker = json.loads(model_root.joinpath(".model_id").read_text())
        assert marker["weights_materialized"] is False
        assert marker["revision"] == profile.revision

    assert len(records) == 4
    assert [record["id"] for record in records] == [
        "color-grid-bottom-left",
        "color-grid-bottom-right",
        "color-grid-top-left",
        "color-grid-top-right",
    ]
    assert {record["content_id"] for record in records} == {
        hf_vision_text.IMAGE_CONTENT_ID
    }
    assert {record["content_sha256"] for record in records} == {
        hf_vision_text.IMAGE_SHA256
    }
    assert paths.content.joinpath(hf_vision_text.IMAGE_CONTENT_ID).read_bytes() == (
        hf_vision_text.tutorial_image_png()
    )
    assert len(schedule.records) == 4
    assert anchors["schedule_digest"] == f"sha256:{schedule.schedule_sha256}"
    assert trust["anchors"] == anchors
    assert trust["verifier"]["signing_key_path"] == "keys/verifier.pem"
    assert paths.evidence_key.stat().st_mode & 0o777 == 0o600
    assert paths.verifier_key.stat().st_mode & 0o777 == 0o600
    assert policy == {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": -100.0,
                    "maximum_interval_width_pp": 200.0,
                    "minimum_record_count": 4,
                }
            }
        }
    }


def test_download_models_authenticates_exact_revision_and_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        destination = Path(str(kwargs["local_dir"]))
        destination.mkdir(parents=True)
        destination.joinpath("config.json").write_text("{}\n", encoding="utf-8")
        return str(destination)

    monkeypatch.setattr(hf_vision_text, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        hf_vision_text,
        "checkpoint_tree_sha256",
        lambda path: hf_vision_text.MODEL_PROFILES[path.name].checkpoint_tree_sha256,
    )

    hf_vision_text.download_models(tmp_path)

    assert [call["repo_id"] for call in calls] == [
        hf_vision_text.MODEL_PROFILES["baseline"].model_id,
        hf_vision_text.MODEL_PROFILES["subject"].model_id,
    ]
    assert [call["revision"] for call in calls] == [
        hf_vision_text.MODEL_PROFILES["baseline"].revision,
        hf_vision_text.MODEL_PROFILES["subject"].revision,
    ]
    assert all(
        call["ignore_patterns"] == (".gitattributes", "LICENSE", "README.md")
        for call in calls
    )
    assert all(path.stat().st_mode & 0o004 for path in tmp_path.rglob("config.json"))

    monkeypatch.setattr(
        hf_vision_text,
        "checkpoint_tree_sha256",
        lambda _path: "sha256:" + ("f" * 64),
    )
    with pytest.raises(RuntimeError, match="checkpoint tree digest mismatch"):
        hf_vision_text.download_models(tmp_path / "mismatch")

    existing = tmp_path / "existing"
    existing.joinpath("baseline").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="model destination already exists"):
        hf_vision_text.download_models(existing)


def test_preparation_materializes_models_and_removes_partial_output_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        hf_vision_text,
        "download_models",
        lambda path: calls.append(path),
    )
    complete = tmp_path / "materialized"
    hf_vision_text.prepare_workspace(
        complete,
        runtime_image_digest=ZERO_DIGEST,
        materialize_models=True,
    )
    assert calls == [complete / "evaluation/models"]

    failed = tmp_path / "failed"
    monkeypatch.setattr(
        hf_vision_text,
        "prepare_local_evaluation_schedule_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("schedule")),
    )
    with pytest.raises(RuntimeError, match="schedule"):
        hf_vision_text.prepare_workspace(
            failed,
            runtime_image_digest=ZERO_DIGEST,
            materialize_models=False,
        )
    assert not failed.exists()

    missing = tmp_path / "missing-workspace"
    linked = tmp_path / "linked-workspace"
    linked.symlink_to(missing, target_is_directory=True)
    with pytest.raises(FileExistsError, match="workspace already exists"):
        hf_vision_text.prepare_workspace(
            linked,
            runtime_image_digest=ZERO_DIGEST,
            materialize_models=False,
        )
    assert not missing.exists()


def test_execute_uses_public_commands_and_vision_resource_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _anchors = hf_vision_text.prepare_workspace(
        tmp_path / "execute",
        runtime_image_digest=ZERO_DIGEST,
        materialize_models=False,
    )
    report = paths.evidence / "reports/evaluation.report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({"comparison": {"value": 0.0}, "verdict": "pass"}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(command: list[str], *, environment: dict[str, str]) -> None:
        commands.append(command)
        environments.append(environment)

    monkeypatch.setattr(hf_vision_text, "_run", fake_run)
    hf_vision_text.execute(
        paths,
        container_engine="docker",
        runtime_image="sha256:" + ("a" * 64),
        runtime_image_digest="sha256:" + ("a" * 64),
        runtime_device="cuda:1",
    )

    assert [command[3] for command in commands] == ["evaluate", "verify", "report"]
    assert commands[0][commands[0].index("--runtime-device") + 1] == "cuda:1"
    assert all(
        environment["INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT"] == str(paths.evaluation)
        for environment in environments
    )
    assert all(
        environment["INVARLOCK_HF_VISION_TEXT_CONTENT_STORE"] == "inputs/content"
        for environment in environments
    )

    report.write_text(
        json.dumps({"comparison": {"value": "invalid"}, "verdict": "pass"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not produce a passing result"):
        hf_vision_text.execute(
            paths,
            container_engine="docker",
            runtime_image="sha256:" + ("a" * 64),
            runtime_image_digest="sha256:" + ("a" * 64),
            runtime_device="cuda:1",
        )


def test_vision_command_runner_reports_output_and_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        hf_vision_text,
        "run_bounded_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="done\n", stderr=""
        ),
    )
    hf_vision_text._run(["true"], environment={})
    assert "done" in capsys.readouterr().out

    monkeypatch.setattr(
        hf_vision_text,
        "run_bounded_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 4, stdout="", stderr="failed\n"
        ),
    )
    with pytest.raises(RuntimeError, match="status 4"):
        hf_vision_text._run(["false"], environment={})
    assert "failed" in capsys.readouterr().err


def test_vision_main_prepares_executes_and_handles_input_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = tmp_path / "prepared"
    assert (
        hf_vision_text.main(
            [
                "--workspace",
                str(prepared),
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--prepare-only",
                "--ephemeral-trust-root",
            ]
        )
        == 0
    )

    existing = tmp_path / "existing"
    existing.mkdir()
    assert (
        hf_vision_text.main(
            [
                "--workspace",
                str(existing),
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--prepare-only",
                "--ephemeral-trust-root",
            ]
        )
        == 2
    )
    with pytest.raises(SystemExit, match="full execution requires"):
        hf_vision_text.main(
            [
                "--workspace",
                str(tmp_path / "missing-image"),
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--ephemeral-trust-root",
            ]
        )

    observed: list[tuple[Path, str]] = []

    def fake_prepare(
        root: Path,
        *,
        runtime_image_digest: str,
        materialize_models: bool,
        **_kwargs: object,
    ) -> tuple[hf_vision_text.VisionExamplePaths, dict[str, str]]:
        assert materialize_models is True
        paths = hf_vision_text._paths(root)
        observed.append((root, runtime_image_digest))
        return paths, {}

    monkeypatch.setattr(hf_vision_text, "prepare_workspace", fake_prepare)
    monkeypatch.setattr(
        hf_vision_text,
        "execute",
        lambda paths, **_kwargs: observed.append((paths.root, "executed")),
    )
    full = tmp_path / "full"
    assert (
        hf_vision_text.main(
            [
                "--workspace",
                str(full),
                "--runtime-image-digest",
                ZERO_DIGEST,
                "--runtime-image",
                "sha256:" + ("a" * 64),
                "--runtime-device",
                "cuda:1",
                "--ephemeral-trust-root",
            ]
        )
        == 0
    )
    assert observed == [(full, ZERO_DIGEST), (full, "executed")]


def test_launch_builds_layered_vision_runtime_and_dispatches_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    builds: list[dict[str, object]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool = False,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_runtime(**kwargs: object) -> tuple[str, str]:
        builds.append(kwargs)
        digest = "sha256:" + str(len(builds)) * 64
        return digest, digest

    @contextmanager
    def fake_publish(**kwargs: object):
        assert kwargs["image"] == "sha256:" + ("1" * 64)
        assert kwargs["image_digest"] == "sha256:" + ("1" * 64)
        yield "127.0.0.1:49152/invarlock-example-runtime-cuda@sha256:" + ("1" * 64)

    monkeypatch.setattr(launch, "_run", fake_run)
    monkeypatch.setattr(launch, "_runtime_image", fake_runtime)
    monkeypatch.setattr(launch, "published_local_image", fake_publish)
    monkeypatch.setattr(launch, "_require_committed_checkout", lambda _repo: "c" * 40)

    assert (
        launch.main(
            [
                "hf-vision-text",
                "--workspace",
                str(tmp_path / "journey"),
                "--runtime-device",
                "cuda:1",
                "--ephemeral-trust-root",
            ]
        )
        == 0
    )
    assert [build["dockerfile"] for build in builds] == [
        "runtime/Dockerfile.cuda",
        "addins/multimodal/runtime/Dockerfile",
    ]
    assert builds[1]["authenticated_base_image"] == (
        "127.0.0.1:49152/invarlock-example-runtime-cuda@sha256:" + ("1" * 64)
    )
    dockerfile = Path(__file__).resolve().parents[2] / (
        "addins/multimodal/runtime/Dockerfile"
    )
    assert "ARG RUNTIME_BASE_IMAGE" in dockerfile.read_text(encoding="utf-8")
    assert commands[-1][1].endswith("examples/integrations/hf_vision_text.py")
    assert commands[-1][commands[-1].index("--runtime-device") + 1] == "cuda:1"


def test_vision_example_rejects_unqualified_container_engine(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        launch.main(
            [
                "hf-vision-text",
                "--container-engine",
                "podman",
                "--ephemeral-trust-root",
            ]
        )
        == 2
    )
    assert "requires Docker" in capsys.readouterr().err


def test_makefile_and_docs_expose_one_command_vision_example() -> None:
    root = Path(__file__).resolve().parents[2]
    makefile = root.joinpath("Makefile").read_text(encoding="utf-8")
    overview = root.joinpath("examples/README.md").read_text(encoding="utf-8")
    integrations = root.joinpath("examples/integrations/README.md").read_text(
        encoding="utf-8"
    )
    guide = root.joinpath("examples/integrations/hf-vision-text/README.md")

    assert "example-hf-vision-text:" in makefile
    assert "-m examples.integrations.launch hf-vision-text" in makefile
    assert "make example-hf-vision-text" in overview
    assert "make example-hf-vision-text" in integrations
    assert guide.is_file()
    assert "--prepare-only" in guide.read_text(encoding="utf-8")


def test_vision_example_sources_are_tracked_for_clean_export() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "examples/integrations/hf_vision_text.py",
        "examples/integrations/hf-vision-text/README.md",
        "examples/integrations/local_registry.py",
    }
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    assert expected <= tracked
