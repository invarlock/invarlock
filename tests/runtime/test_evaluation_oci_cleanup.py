"""Failed worker outputs remain bounded and removable across numeric users."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from invarlock import evaluation_oci as oci


def _launch():
    side = oci.OciSideLaunch("sha256:" + "a" * 64, "sha256:" + "a" * 64, "cpu")
    return oci.OciEvaluationLaunch("docker", side, side)


def test_private_worker_directory_recovers_foreign_uid_output(monkeypatch, tmp_path):
    original = shutil.rmtree
    collectible = oci._remove_collectible_output
    calls = []
    blocked = set()

    def remove(path, *args, **kwargs):
        path = Path(path)
        if path.name.endswith("-output") and path not in blocked:
            blocked.add(path)
            raise PermissionError("worker owns side/report.json")
        return collectible(path)

    def recover(command, *, timeout_seconds):
        calls.append(command)
        assert timeout_seconds == 30
        source = next(value for value in command if value.startswith("type=bind,"))
        output = Path(source.split("source=", 1)[1].split(",", 1)[0])
        original(output / "side")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(oci, "_remove_collectible_output", remove)
    monkeypatch.setattr(oci, "run_side_worker", recover)
    original_error = ValueError("runtime side output is not the closed six-file bundle")
    with pytest.raises(ValueError) as observed:
        with oci._worker_directory(_launch(), {"baseline": "hf_transformers"}) as root:
            output = root / "baseline-output"
            output.mkdir(mode=0o733)
            (output / "side").mkdir()
            (output / "side/report.json").write_text("{}")
            assert root.stat().st_mode & 0o777 == 0o700
            raise original_error
    assert observed.value is original_error
    assert not root.exists()
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--user") + 1] == "65532:65532"
    assert command[command.index("--memory") + 1] == "128m"
    assert command[command.index("--cpus") + 1] == "1"
    assert "--read-only" in command and "--cap-drop=ALL" in command
    assert "no-new-privileges" in command and "--pull=never" in command
    assert command.count("--mount") == 1
    assert "--gpus" not in command and "--device" not in command
    assert _launch().baseline.image_ref in command
    assert command[-5:-2] == ["-I", "-S", "-c"]


@pytest.mark.parametrize(
    "failure", [ValueError("original failure"), KeyboardInterrupt()]
)
def test_cleanup_failure_preserves_original_and_reports_retained_path(
    monkeypatch, caplog, failure
):
    original = shutil.rmtree

    def remove(_path):
        raise PermissionError("foreign owner")

    monkeypatch.setattr(oci, "_remove_collectible_output", remove)
    monkeypatch.setattr(
        oci,
        "run_side_worker",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 124, "", "deadline"
        ),
    )
    with pytest.raises(type(failure)) as observed:
        with oci._worker_directory(_launch(), {"baseline": "hf_transformers"}) as root:
            (root / "baseline-output").mkdir()
            raise failure
    assert observed.value is failure
    assert root.exists()
    assert str(root) in caplog.text and "cleanup" in caplog.text
    assert any(str(root) in note for note in failure.__notes__)
    original(root)


def test_cleanup_failure_cannot_return_success(monkeypatch):
    original = shutil.rmtree
    monkeypatch.setattr(
        oci,
        "_remove_collectible_output",
        lambda path: (_ for _ in ()).throw(PermissionError()),
    )
    monkeypatch.setattr(
        oci,
        "run_side_worker",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 9, "", "failed"),
    )
    with pytest.raises(oci.OciEvaluationError, match="cleanup.*retained"):
        with oci._worker_directory(_launch(), {"baseline": "hf_transformers"}) as root:
            (root / "baseline-output").mkdir()
    original(root)


def test_cleanup_refuses_replaced_output_root_and_preserves_external_files(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("unchanged")
    (root / "baseline-output").symlink_to(external, target_is_directory=True)
    with pytest.raises(oci.OciEvaluationError, match="real directory"):
        oci._cleanup_worker_directory(root, _launch(), {"baseline": "hf_transformers"})
    assert sentinel.read_text() == "unchanged"


def test_cleanup_missing_outputs_and_empty_outputs_need_no_container(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("empty or absent output must not launch a cleanup container")

    monkeypatch.setattr(oci, "run_side_worker", unexpected)
    with oci._worker_directory(_launch(), {}) as missing:
        (missing / "baseline").mkdir()
        (missing / "baseline/job.json").write_text("{}")
    assert not missing.exists()
    with oci._worker_directory(_launch(), {}) as empty:
        (empty / "baseline-output").mkdir()
    assert not empty.exists()


@pytest.mark.parametrize("kind", ["extra", "nested", "too-many", "side-link"])
def test_host_cleanup_defers_unexpected_shapes_without_following_links(tmp_path, kind):
    output = tmp_path / "output"
    output.mkdir()
    side = output / "side"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("unchanged")
    if kind == "side-link":
        side.symlink_to(external, target_is_directory=True)
    else:
        side.mkdir()
        (side / "report.json").write_text("{}")
        if kind == "extra":
            (output / "unexpected").write_text("extra")
        elif kind == "nested":
            (side / "nested").mkdir()
        else:
            for index in range(6):
                (side / str(index)).write_text("extra")
    with pytest.raises(OSError):
        oci._remove_collectible_output(output)
    assert sentinel.read_text() == "unchanged"
    assert side.exists()


def test_host_cleanup_unlinks_symlinks_and_hardlinks_without_mutating_targets(tmp_path):
    output = tmp_path / "output"
    side = output / "side"
    side.mkdir(parents=True)
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("unchanged")
    sentinel.chmod(0o400)
    (side / "symbolic").symlink_to(sentinel)
    (side / "hard").hardlink_to(sentinel)
    os.mkfifo(side / "fifo")
    oci._remove_collectible_output(output)
    assert list(output.iterdir()) == []
    assert sentinel.read_text() == "unchanged"
    assert sentinel.stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize(
    ("profile", "provider", "expected"),
    [
        ("auto", "unknown", "python"),
        ("auto", "hf_transformers", "python"),
        ("auto", "tensorrt_llm", "/opt/invarlock/cli-venv/bin/python"),
        ("nvidia", "unknown", "/opt/invarlock/cli-venv/bin/python"),
        ("python", "tensorrt_llm", "python"),
    ],
)
def test_cleanup_uses_selected_profile_and_engine(
    tmp_path, profile, provider, expected
):
    side = oci.OciSideLaunch("sha256:" + "b" * 64, "sha256:" + "b" * 64, "cpu", profile)
    launch = oci.OciEvaluationLaunch("podman", side, side, engine_path="/bin/podman")
    command = oci._output_cleanup_command(
        tmp_path, tmp_path / "subject-output", launch, side, provider
    )
    assert command[0] == "/bin/podman"
    assert command[command.index("--entrypoint") + 1] == expected
    assert command[command.index("--entrypoint") + 2] == side.image_ref
    assert command[command.index("--cidfile") + 1] == str(
        tmp_path / "subject-output-cleanup.cid"
    )


@pytest.mark.parametrize("failure", ["launch", "false-success", "empty-diagnostic"])
def test_helper_failure_preserves_restricted_output_and_reports_it(
    monkeypatch, tmp_path, failure
):
    root = tmp_path / "owned"
    output = root / "baseline-output"
    output.mkdir(parents=True)
    (output / "unexpected").mkdir()

    def run(command, **_kwargs):
        assert output.stat().st_mode & 0o777 == 0o777
        if failure == "launch":
            raise oci.OciEvaluationError("engine unavailable")
        return subprocess.CompletedProcess(
            command, 0 if failure == "false-success" else 9, "", ""
        )

    monkeypatch.setattr(oci, "run_side_worker", run)
    with pytest.raises(oci.OciEvaluationError, match="cleanup failed"):
        oci._cleanup_worker_directory(root, _launch(), {"baseline": "hf_transformers"})
    assert output.stat().st_mode & 0o777 == 0o733
    assert (output / "unexpected").exists()


def test_cleanup_changed_inode_is_rejected_before_launch(monkeypatch, tmp_path):
    root = tmp_path / "owned"
    output = root / "baseline-output"
    output.mkdir(parents=True)
    replacement = root / "replacement"
    replacement.mkdir()

    def replace(_directory):
        output.rmdir()
        replacement.rename(output)
        raise OSError("changed while collecting")

    monkeypatch.setattr(oci, "_remove_collectible_output", replace)
    with pytest.raises(oci.OciEvaluationError, match="changed before cleanup"):
        oci._cleanup_worker_directory(root, _launch(), {"baseline": "hf_transformers"})


def test_restore_permission_failure_still_closes_pinned_descriptor(
    monkeypatch, tmp_path
):
    root = tmp_path / "owned"
    output = root / "baseline-output"
    output.mkdir(parents=True)
    (output / "unexpected").mkdir()
    original_fchmod = os.fchmod
    pinned = []

    def chmod(descriptor, mode):
        if mode == 0o733:
            pinned.append(descriptor)
            raise PermissionError("restore refused")
        return original_fchmod(descriptor, mode)

    monkeypatch.setattr(oci.os, "fchmod", chmod)
    monkeypatch.setattr(
        oci,
        "run_side_worker",
        lambda command, **_: subprocess.CompletedProcess(command, 9, "", ""),
    )
    with pytest.raises(oci.OciEvaluationError, match="restore refused"):
        oci._cleanup_worker_directory(root, _launch(), {"baseline": "hf_transformers"})
    assert len(pinned) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(pinned[0])
