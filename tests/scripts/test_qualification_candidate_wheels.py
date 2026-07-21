from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from scripts import qualification_candidate_wheels as candidate_wheels
from scripts import runtime_qualification


def _wheel(path: Path, payload: bytes = b"wheel fixture\n") -> Path:
    path.write_bytes(payload)
    return path


def test_generator_publishes_canonical_runtime_accepted_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    second = _wheel(tmp_path / "z-second.whl", b"second\n")
    first = _wheel(tmp_path / "a-first.whl", b"first\n")
    output = tmp_path / "candidate-wheels.json"

    assert (
        candidate_wheels.main(
            [
                "--wheel",
                str(second),
                "--wheel",
                str(first),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    expected = {
        "format_version": "invarlock/qualification-candidate-wheels-v1",
        "wheels": [
            {
                "path": str(first),
                "sha256": "sha256:" + hashlib.sha256(first.read_bytes()).hexdigest(),
            },
            {
                "path": str(second),
                "sha256": "sha256:" + hashlib.sha256(second.read_bytes()).hexdigest(),
            },
        ],
    }
    canonical = json.dumps(expected, separators=(",", ":"), sort_keys=True) + "\n"
    assert output.read_text(encoding="utf-8") == canonical
    assert capsys.readouterr().out == canonical
    assert stat.S_IMODE(output.stat().st_mode) == 0o600

    manifest_digest, specs = runtime_qualification._candidate_wheel_specs(  # noqa: SLF001
        output
    )
    assert manifest_digest == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    assert [(spec.path, spec.sha256) for spec in specs] == [
        (first, expected["wheels"][0]["sha256"]),
        (second, expected["wheels"][1]["sha256"]),
    ]


@pytest.mark.parametrize("kind", ["same-path", "hard-link"])
def test_generator_rejects_duplicate_file_identity(
    tmp_path: Path,
    kind: str,
) -> None:
    wheel = _wheel(tmp_path / "candidate.whl")
    repeated = wheel
    if kind == "hard-link":
        repeated = tmp_path / "alias.whl"
        os.link(wheel, repeated)
    output = tmp_path / "candidate-wheels.json"

    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="repeated",
    ):
        candidate_wheels.create_manifest([wheel, repeated], output=output)

    assert not output.exists()


def test_generator_rejects_symlinked_wheel_and_parent(
    tmp_path: Path,
) -> None:
    wheel = _wheel(tmp_path / "candidate.whl")
    wheel_link = tmp_path / "candidate-link.whl"
    wheel_link.symlink_to(wheel)

    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="without symbolic links",
    ):
        candidate_wheels.create_manifest(
            [wheel_link], output=tmp_path / "candidate-wheels.json"
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="parent must be one real directory",
    ):
        candidate_wheels.create_manifest(
            [wheel], output=parent_link / "candidate-wheels.json"
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda root: root / "missing.whl", "unavailable"),
        (lambda root: _wheel(root / "candidate.zip"), "real .whl path"),
        (
            lambda root: root / "directory.whl",
            "bounded regular file",
        ),
    ],
)
def test_generator_rejects_non_wheel_inputs(
    tmp_path: Path,
    factory,
    message: str,
) -> None:
    candidate = factory(tmp_path)
    if candidate.name == "directory.whl":
        candidate.mkdir()

    with pytest.raises(candidate_wheels.CandidateWheelManifestError, match=message):
        candidate_wheels.create_manifest(
            [candidate], output=tmp_path / "candidate-wheels.json"
        )


def test_generator_rejects_oversized_and_invalid_inventory(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.whl"
    with oversized.open("wb") as handle:
        handle.truncate(candidate_wheels._MAX_WHEEL_BYTES + 1)  # noqa: SLF001

    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="bounded regular file",
    ):
        candidate_wheels.create_manifest(
            [oversized], output=tmp_path / "candidate-wheels.json"
        )
    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="requires 1-8 wheels",
    ):
        candidate_wheels.create_manifest([], output=tmp_path / "candidate-wheels.json")
    wheels = [_wheel(tmp_path / f"candidate-{index}.whl") for index in range(9)]
    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="requires 1-8 wheels",
    ):
        candidate_wheels.create_manifest(
            wheels, output=tmp_path / "candidate-wheels.json"
        )


def test_generator_rejects_wheel_change_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = _wheel(tmp_path / "candidate.whl")
    real_read = candidate_wheels.os.read
    changed = False

    def read_then_change(descriptor: int, amount: int) -> bytes:
        nonlocal changed
        payload = real_read(descriptor, amount)
        if payload and not changed:
            with wheel.open("ab") as handle:
                handle.write(b"changed\n")
            changed = True
        return payload

    monkeypatch.setattr(candidate_wheels.os, "read", read_then_change)

    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="changed while it was read",
    ):
        candidate_wheels.create_manifest(
            [wheel], output=tmp_path / "candidate-wheels.json"
        )


def test_generator_never_clobbers_existing_or_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = _wheel(tmp_path / "candidate.whl")
    output = tmp_path / "candidate-wheels.json"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="already exists",
    ):
        candidate_wheels.create_manifest([wheel], output=output)
    assert output.read_text(encoding="utf-8") == "existing\n"

    output.unlink()
    real_link = candidate_wheels.os.link

    def racing_link(source, destination, *, follow_symlinks):
        Path(destination).write_text("racer\n", encoding="utf-8")
        return real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(candidate_wheels.os, "link", racing_link)
    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="already exists",
    ):
        candidate_wheels.create_manifest([wheel], output=output)
    assert output.read_text(encoding="utf-8") == "racer\n"
    assert not list(tmp_path.glob(".candidate-wheels.json.*.tmp"))


def test_main_reports_creation_error_without_output(tmp_path: Path) -> None:
    output = tmp_path / "candidate-wheels.json"

    with pytest.raises(SystemExit, match="candidate wheel is unavailable"):
        candidate_wheels.main(
            [
                "--wheel",
                str(tmp_path / "missing.whl"),
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


def test_generator_reports_descriptor_open_failure_and_leaves_no_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = _wheel(tmp_path / "candidate.whl")
    output = tmp_path / "candidate-wheels.json"
    monkeypatch.setattr(
        candidate_wheels.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")),
    )

    with pytest.raises(
        candidate_wheels.CandidateWheelManifestError,
        match="candidate wheel could not be read",
    ):
        candidate_wheels.create_manifest([wheel], output=output)

    assert not output.exists()


def test_generator_closes_temporary_descriptor_when_permission_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = _wheel(tmp_path / "candidate.whl")
    output = tmp_path / "candidate-wheels.json"
    real_close = candidate_wheels.os.close
    closed: list[int] = []

    def recording_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(candidate_wheels.os, "close", recording_close)
    monkeypatch.setattr(
        candidate_wheels.os,
        "fchmod",
        lambda *_args: (_ for _ in ()).throw(OSError("permission setup failed")),
    )

    with pytest.raises(OSError, match="permission setup failed"):
        candidate_wheels.create_manifest([wheel], output=output)

    assert closed
    assert not output.exists()
    assert not list(tmp_path.glob(".candidate-wheels.json.*.tmp"))
