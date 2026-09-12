"""Directory publishers retain ownership across staging and cleanup races."""

import importlib
import json
import os
import shutil
from dataclasses import replace

import pytest

from invarlock import evidence_pack_publication, filesystem, runtime_import_authoring
from invarlock.filesystem import AtomicDirectoryPublicationError
from invarlock.filesystem.staged_directory import staged_directory
from tests.evidence_packs.test_evidence_pack import _publish
from tests.runtime.test_runtime_import_authoring import _artifact, _write_side

stages = importlib.import_module("invarlock.filesystem.staged_directory")


@pytest.fixture(params=["setup", "runtime", "evidence"])
def publisher(request, tmp_path):
    if request.param == "setup":
        app = importlib.import_module("invarlock.cli.app")
        return filesystem, lambda path: app._publish_setup_directory(
            path, {"nested/config.json": b"complete setup"}
        )
    if request.param == "runtime":
        return runtime_import_authoring, lambda path: _write_side(
            path,
            role="baseline",
            artifact=_artifact("baseline.gguf", "1"),
            outputs=("A", "B"),
            image_marker="a",
        )
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    *_rest, arguments = _publish(fixture)
    return (
        evidence_pack_publication,
        lambda path: evidence_pack_publication.publish_comparison_evidence(
            path, **arguments
        ),
    )


def _files(directory):
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_each_publisher_rejects_complete_staging_substitution(
    tmp_path, monkeypatch, publisher
):
    module, publish = publisher
    original = module.publish_directory_no_replace
    retained = tmp_path / "retained"
    replacement = None
    expected = {}

    def substitute(stage, destination, **ownership):
        nonlocal replacement, expected
        assert os.fstat(ownership["expected_source_fd"]).st_ino == stage.stat().st_ino
        stage.rename(retained)
        shutil.copytree(retained, stage)
        replacement = stage
        expected = _files(stage)
        return original(stage, destination, **ownership)

    monkeypatch.setattr(module, "publish_directory_no_replace", substitute)
    with pytest.raises((OSError, ValueError), match="staging identity changed"):
        publish(tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert replacement is not None
    assert _files(replacement) == expected
    assert _files(retained) == expected


def test_each_publisher_preserves_recreated_stage_after_success(
    tmp_path, monkeypatch, publisher
):
    module, publish = publisher
    original = module.publish_directory_no_replace
    markers = []

    def recreate(stage, destination, **ownership):
        original(stage, destination, **ownership)
        stage.mkdir()
        marker = stage / "foreign"
        marker.write_bytes(b"foreign data")
        markers.append(marker)

    monkeypatch.setattr(module, "publish_directory_no_replace", recreate)
    publish(tmp_path / "output")
    assert markers[0].read_bytes() == b"foreign data"
    assert (tmp_path / "output").is_dir()


def test_each_publisher_preserves_foreign_stage_after_failure(
    tmp_path, monkeypatch, publisher
):
    module, publish = publisher
    markers = []

    def substitute_then_fail(stage, destination, **ownership):
        stage.rename(tmp_path / "retained")
        stage.mkdir()
        marker = stage / "foreign"
        marker.write_bytes(b"foreign data")
        markers.append(marker)
        raise AtomicDirectoryPublicationError("primary publication failure")

    monkeypatch.setattr(module, "publish_directory_no_replace", substitute_then_fail)
    with pytest.raises((OSError, ValueError), match="primary publication failure"):
        publish(tmp_path / "output")
    assert markers[0].read_bytes() == b"foreign data"
    assert not (tmp_path / "output").exists()


def test_cleanup_failure_preserves_primary_error_and_releases_descriptors(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="primary operation failure"):
        with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
            (stage.path / "file").write_bytes(b"payload")
            monkeypatch.setattr(
                os,
                "unlink",
                lambda *a, **kw: (_ for _ in ()).throw(
                    PermissionError("secondary cleanup failure")
                ),
            )
            raise ValueError("primary operation failure")
    for descriptor in (stage.descriptor, stage.parent_descriptor):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_cleanup_removes_private_contents_without_following_symlinks(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_bytes(b"foreign")
    with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
        (stage.path / "nested").mkdir()
        (stage.path / "nested" / "file").write_bytes(b"payload")
        (stage.path / "linked").symlink_to(outside, target_is_directory=True)
        stage.make_read_only()
    assert not stage.path.exists()
    assert (outside / "keep").read_bytes() == b"foreign"


@pytest.mark.parametrize("failure", ["open", "fstat"])
def test_stage_open_failure_closes_descriptor_and_removes_empty_root(
    tmp_path, monkeypatch, failure
):
    original_open, original_fstat = os.open, os.fstat
    opened = []

    def open_stage(path, *args, **kwargs):
        if str(path).startswith(".stage-") and failure == "open":
            raise OSError("stage open failure")
        descriptor = original_open(path, *args, **kwargs)
        if str(path).startswith(".stage-"):
            opened.append(descriptor)
        return descriptor

    def fstat_stage(descriptor):
        if descriptor in opened:
            raise OSError("stage fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(os, "open", open_stage)
    monkeypatch.setattr(os, "fstat", fstat_stage)
    with pytest.raises(OSError, match=f"stage {failure} failure"):
        with staged_directory(tmp_path / "output", prefix=".stage-"):
            pytest.fail("stage must not be exposed")
    assert list(tmp_path.iterdir()) == []
    for descriptor in opened:
        with pytest.raises(OSError):
            original_fstat(descriptor)


def test_post_rename_failure_preserves_completed_directory(tmp_path):
    with pytest.raises(OSError, match="post-rename failure"):
        with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
            (stage.path / "file").write_bytes(b"published")

            def fail_after_publish(source, destination, **ownership):
                filesystem.publish_directory_no_replace(
                    source, destination, **ownership
                )
                raise OSError("post-rename failure")

            stage.publish(fail_after_publish)
    assert (tmp_path / "output" / "file").read_bytes() == b"published"


def test_directory_modes_use_retained_tree_after_output_substitution(tmp_path):
    with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
        (stage.path / "nested").mkdir()
        stage.publish()
        stage.destination.rename(tmp_path / "retained")
        stage.destination.mkdir()
        stage.make_read_only()
        with pytest.raises(OSError, match="published directory identity changed"):
            stage.require_published_binding()
    assert (tmp_path / "retained").stat().st_mode & 0o777 == 0o555
    assert (tmp_path / "retained" / "nested").stat().st_mode & 0o777 == 0o555
    assert stage.destination.stat().st_mode & 0o777 != 0o555


def test_native_publication_rejects_destination_replacement_during_finalization(
    tmp_path, monkeypatch
):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    *_rest, arguments = _publish(fixture)
    original = stages.StagedDirectory.make_read_only
    destination = tmp_path / "output"

    def replace_then_finalize(stage):
        stage.destination.rename(tmp_path / "retained")
        stage.destination.mkdir()
        (stage.destination / "foreign").write_bytes(b"foreign")
        original(stage)

    monkeypatch.setattr(stages.StagedDirectory, "make_read_only", replace_then_finalize)
    with pytest.raises(ValueError, match="published directory identity changed"):
        evidence_pack_publication.publish_comparison_evidence(destination, **arguments)
    assert (destination / "foreign").read_bytes() == b"foreign"
    assert destination.stat().st_mode & 0o777 != 0o555


def test_runtime_publication_rejects_destination_replacement_after_reload(
    tmp_path, monkeypatch
):
    original = runtime_import_authoring.load_runtime_import_side
    destination = tmp_path / "output"

    def reload_then_replace(path, **kwargs):
        result = original(path, **kwargs)
        path.rename(tmp_path / "retained")
        path.mkdir()
        (path / "foreign").write_bytes(b"foreign")
        return result

    monkeypatch.setattr(
        runtime_import_authoring, "load_runtime_import_side", reload_then_replace
    )
    with pytest.raises(ValueError, match="published directory identity changed"):
        _write_side(
            destination,
            role="baseline",
            artifact=_artifact("baseline.gguf", "1"),
            outputs=("A", "B"),
            image_marker="a",
        )
    assert (destination / "foreign").read_bytes() == b"foreign"


def test_stage_rejects_directory_not_owned_by_caller(tmp_path, monkeypatch):
    actual_uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: actual_uid + 1)
    with pytest.raises(OSError, match="private staging directory identity changed"):
        with staged_directory(tmp_path / "output", prefix=".stage-"):
            pytest.fail("foreign directory must not be exposed")
    assert list(tmp_path.iterdir()) == []


def test_private_tree_nesting_limit_fails_without_publishing(tmp_path):
    with pytest.raises(OSError, match="nesting exceeds cleanup limit"):
        with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
            nested = stage.path
            for _ in range(17):
                nested /= "nested"
                nested.mkdir()
            (nested / "file").write_bytes(b"retained on bounded cleanup failure")
            stage.make_read_only()
    assert not (tmp_path / "output").exists()
    assert (nested / "file").read_bytes() == b"retained on bounded cleanup failure"


@pytest.mark.parametrize("replaced", ["root", "child"])
def test_cleanup_preserves_name_replaced_after_emptying_owned_directory(
    tmp_path, monkeypatch, replaced
):
    original = stages._change_tree
    with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
        child = stage.path / "child"
        child.mkdir()
        (child / "file").write_bytes(b"owned")
        target = stage.path if replaced == "root" else child
        target_inode = target.stat().st_ino

        def replace_after_emptying(descriptor, **kwargs):
            original(descriptor, **kwargs)
            if os.fstat(descriptor).st_ino == target_inode:
                target.rename(tmp_path / "retained")
                target.mkdir()
                (target / "foreign").write_bytes(b"foreign")

        monkeypatch.setattr(stages, "_change_tree", replace_after_emptying)
    assert (target / "foreign").read_bytes() == b"foreign"


def test_each_publisher_rejects_outer_stage_swapped_during_writes_then_restored(
    tmp_path, monkeypatch, publisher
):
    _module, publish = publisher
    original_open = stages._open_owned_child
    original_validate = stages.StagedDirectory.require_exact_files
    retained = tmp_path / "retained-original"
    foreign = tmp_path / "foreign-written-tree"
    swapped = False

    def swap_after_open(parent, name):
        nonlocal swapped
        descriptor = original_open(parent, name)
        if not swapped:
            swapped = True
            path = tmp_path / name
            path.rename(retained)
            path.mkdir()
        return descriptor

    def restore_before_validation(stage, expected):
        stage.path.rename(foreign)
        retained.rename(stage.path)
        original_validate(stage, expected)

    monkeypatch.setattr(stages, "_open_owned_child", swap_after_open)
    monkeypatch.setattr(
        stages.StagedDirectory, "require_exact_files", restore_before_validation
    )
    with pytest.raises(
        (OSError, ValueError), match="inventory does not match generated files"
    ):
        publish(tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert _files(foreign)


def test_runtime_rejects_foreign_reload_even_when_original_output_is_restored(
    tmp_path, monkeypatch
):
    foreign = _write_side(
        tmp_path / "foreign",
        role="baseline",
        artifact=_artifact("foreign.gguf", "e"),
        outputs=("A", "wrong"),
        image_marker="a",
    )
    original = runtime_import_authoring.load_runtime_import_side
    retained = tmp_path / "retained"
    destination = tmp_path / "output"

    def substitute_during_reload(path, **kwargs):
        path.rename(retained)
        foreign.directory.rename(path)
        try:
            return original(path, **kwargs)
        finally:
            path.rename(foreign.directory)
            retained.rename(path)

    monkeypatch.setattr(
        runtime_import_authoring, "load_runtime_import_side", substitute_during_reload
    )
    with pytest.raises(ValueError, match="reload does not match generated evidence"):
        _write_side(
            destination,
            role="baseline",
            artifact=_artifact("expected.gguf", "1"),
            outputs=("A", "B"),
            image_marker="a",
        )
    assert destination.is_dir()
    assert foreign.directory.is_dir()


@pytest.mark.parametrize("mutation", ["missing", "extra", "bytes", "symlink", "fifo"])
def test_exact_generated_files_reject_inventory_and_content_changes(tmp_path, mutation):
    expected = {"nested/file": b"expected payload"}
    with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
        nested = stage.path / "nested"
        nested.mkdir()
        leaf = nested / "file"
        leaf.write_bytes(expected["nested/file"])
        if mutation == "missing":
            leaf.unlink()
        elif mutation == "extra":
            (nested / "extra").write_bytes(b"unexpected")
        elif mutation == "bytes":
            leaf.write_bytes(b"tampered payload")
        elif mutation == "symlink":
            leaf.unlink()
            leaf.symlink_to(tmp_path / "outside")
        else:
            leaf.unlink()
            os.mkfifo(leaf)
        with pytest.raises(OSError, match="does not match generated"):
            stage.require_exact_files(expected)


def test_exact_generated_files_compare_large_payload_in_bounded_chunks(
    tmp_path, monkeypatch
):
    expected = {"payload": b"0123456789" * 20000}
    read_sizes = []
    original_read = os.read

    def read_chunk(descriptor, size):
        read_sizes.append(size)
        return original_read(descriptor, size)

    with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
        (stage.path / "payload").write_bytes(expected["payload"])
        monkeypatch.setattr(os, "read", read_chunk)
        stage.require_exact_files(expected)
    assert len(read_sizes) > 2
    assert max(read_sizes) <= 65536


@pytest.mark.parametrize("phase", ["open", "read", "directory"])
def test_exact_reader_detects_changes_during_validation(tmp_path, monkeypatch, phase):
    original_open, original_read = os.open, os.read
    changed = False
    with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
        leaf = stage.path / "payload"
        leaf.write_bytes(b"expected")

        def mutate():
            nonlocal changed
            if not changed:
                changed = True
                if phase == "directory":
                    (stage.path / "extra").write_bytes(b"unexpected")
                else:
                    leaf.unlink()
                    leaf.write_bytes(b"expected")

        def race_open(name, *args, **kwargs):
            if name == "payload":
                mutate()
            return original_open(name, *args, **kwargs)

        def race_read(descriptor, count):
            chunk = original_read(descriptor, count)
            mutate()
            return chunk

        if phase == "open":
            monkeypatch.setattr(os, "open", race_open)
        else:
            monkeypatch.setattr(os, "read", race_read)
        with pytest.raises(OSError, match="changed"):
            stage.require_exact_files({"payload": b"expected"})


def test_exact_reader_detects_child_directory_replacement_after_read(
    tmp_path, monkeypatch
):
    original = stages._require_exact_tree
    with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
        nested = stage.path / "nested"
        nested.mkdir()
        (nested / "file").write_bytes(b"expected")

        def replace_after_read(descriptor, expected, *, depth=0):
            original(descriptor, expected, depth=depth)
            if depth == 1:
                nested.rename(tmp_path / "retained")
                nested.mkdir()
                (nested / "file").write_bytes(b"expected")

        monkeypatch.setattr(stages, "_require_exact_tree", replace_after_read)
        with pytest.raises(OSError, match="directory identity changed while reading"):
            stage.require_exact_files({"nested/file": b"expected"})


@pytest.mark.parametrize("name", ["/absolute", "../outside", "nested/./file", ""])
def test_exact_reader_rejects_unbounded_generated_paths(tmp_path, name):
    with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
        with pytest.raises(ValueError, match="relative without traversal"):
            stage.require_exact_files({name: b"expected"})


def test_exact_reader_bounds_directory_depth(tmp_path):
    with staged_directory(tmp_path / "output", prefix=".stage-") as stage:
        nested = stage.path
        for _ in range(17):
            nested /= "nested"
            nested.mkdir()
        (nested / "file").write_bytes(b"expected")
        with pytest.raises(OSError, match="nesting exceeds validation limit"):
            stage.require_exact_files(
                {"/".join(["nested"] * 17 + ["file"]): b"expected"}
            )


def test_runtime_uses_one_generated_timestamp_for_expected_and_written_manifest(
    tmp_path, monkeypatch
):
    support = importlib.import_module("tests.runtime.test_runtime_import_authoring")
    monkeypatch.setattr(support, "_GENERATED_AT", None)
    side = _write_side(
        tmp_path / "output",
        role="baseline",
        artifact=_artifact("baseline.gguf", "1"),
        outputs=("A", "B"),
        image_marker="a",
    )
    assert isinstance(
        json.loads(side.side_evidence.runtime_manifest)["generated_at_utc"], str
    )


def test_native_preflight_cannot_validate_original_config_using_substitute_tree(
    tmp_path, monkeypatch
):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    *_rest, arguments = _publish(fixture)
    baseline = arguments["baseline_evidence"]
    assert isinstance(baseline, evidence_pack_publication.RuntimeSideEvidence)
    invalid_config = json.loads(baseline.runtime_config)
    invalid_config["role"] = "subject"
    arguments["baseline_evidence"] = replace(
        baseline,
        runtime_config=evidence_pack_publication.canonical_json_bytes(invalid_config),
    )
    original = evidence_pack_publication._preflight_runtime_side
    retained = tmp_path / "retained"
    foreign = tmp_path / "foreign"

    def substitute_during_preflight(stage, **kwargs):
        stage.rename(retained)
        shutil.copytree(retained, stage)
        config = (
            stage / evidence_pack_publication.EVIDENCE_PATHS["baseline_runtime_config"]
        )
        config.chmod(0o600)
        config.write_bytes(baseline.runtime_config)
        try:
            return original(stage, **kwargs)
        finally:
            stage.rename(foreign)
            retained.rename(stage)

    monkeypatch.setattr(
        evidence_pack_publication,
        "_preflight_runtime_side",
        substitute_during_preflight,
    )
    with pytest.raises(ValueError, match="config.*role"):
        evidence_pack_publication.publish_comparison_evidence(
            tmp_path / "output", **arguments
        )
    assert not (tmp_path / "output").exists()
