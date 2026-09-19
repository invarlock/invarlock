from __future__ import annotations

import json
import runpy
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import scripts.release.first_party_distribution_validation as validation_module
from scripts.release.first_party_distribution_validation import (
    FirstPartyDistribution,
    _artifact_pair,
    _contained_distribution_directory,
    _real_directory,
    _validate_artifact_directory,
    validate_first_party_distributions,
)
from scripts.release.release_distribution_validation import ReleasePreflightError

ROOT = Path(__file__).resolve().parents[2]
VERSION = str(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
)


@pytest.fixture(scope="module")
def built_core(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("first-party-core")
    source = root / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".venv*", "__pycache__", "*.egg-info", "build", "dist", "reports"
        ),
    )
    dist = root / "dist"
    dist.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist)],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return dist


def test_single_core_pair_matches_exact_source(built_core: Path) -> None:
    results = validate_first_party_distributions(
        repo_root=ROOT,
        expected_version=VERSION,
        core_dist_dir=built_core,
    )
    assert len(results) == 1
    assert results[0].project == "core"
    assert results[0].distribution == "invarlock"
    assert results[0].version == VERSION


def test_artifact_directory_requires_one_wheel_and_source_archive(
    tmp_path: Path,
) -> None:
    (tmp_path / "invarlock-0.15.0-py3-none-any.whl").write_bytes(b"wheel")
    with pytest.raises(ReleasePreflightError, match="one wheel/sdist pair"):
        _validate_artifact_directory(tmp_path, expected_pairs=1, label="core")


def test_artifact_pair_rejects_ambiguous_names(tmp_path: Path) -> None:
    (tmp_path / "invarlock-0.15.0-py3-none-any.whl").write_bytes(b"wheel")
    (tmp_path / "invarlock-0.15.0.tar.gz").write_bytes(b"sdist")
    (tmp_path / "invarlock-0.15.0.post1.tar.gz").write_bytes(b"sdist")
    with pytest.raises(ReleasePreflightError, match="ambiguous"):
        _artifact_pair(dist_dir=tmp_path, distribution_name="invarlock")


def test_real_directory_rejects_missing_file_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(ReleasePreflightError, match="missing"):
        _real_directory(tmp_path / "missing", label="candidate")

    regular_file = tmp_path / "file"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ReleasePreflightError, match="one real directory"):
        _real_directory(regular_file, label="candidate")

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ReleasePreflightError, match="symbolic link"):
        _real_directory(symlink, label="candidate")


def test_distribution_directory_must_remain_inside_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    outside = tmp_path / "outside"
    checkout.mkdir()
    outside.mkdir()

    with pytest.raises(ReleasePreflightError, match="inside the checkout"):
        _contained_distribution_directory(checkout, outside)


def test_artifact_directory_rejects_symbolic_links(tmp_path: Path) -> None:
    wheel = tmp_path / "invarlock-0.16.1-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    (tmp_path / "wheel-link").symlink_to(wheel)

    with pytest.raises(ReleasePreflightError, match="symbolic links"):
        _validate_artifact_directory(tmp_path, expected_pairs=1, label="core")


def test_validation_rejects_core_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "invarlock-0.16.1-py3-none-any.whl").write_bytes(b"wheel")
    (tmp_path / "invarlock-0.16.1.tar.gz").write_bytes(b"sdist")
    monkeypatch.setattr(
        validation_module,
        "read_distribution_project",
        lambda _root: ("invarlock", "0.16.1"),
    )

    with pytest.raises(ReleasePreflightError, match="version does not match"):
        validate_first_party_distributions(
            repo_root=tmp_path,
            expected_version="0.16.2",
            core_dist_dir=tmp_path,
        )


def test_main_emits_the_single_distribution_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = tmp_path / "checkout"
    dist = checkout / "dist"
    dist.mkdir(parents=True)
    expected = FirstPartyDistribution(
        project="core",
        distribution="invarlock",
        version="0.16.1",
        wheel="invarlock-0.16.1-py3-none-any.whl",
        sdist="invarlock-0.16.1.tar.gz",
    )
    monkeypatch.setattr(
        validation_module,
        "validate_first_party_distributions",
        lambda **_kwargs: [expected],
    )

    assert (
        validation_module.main(
            [
                "--repo-root",
                str(checkout),
                "--expected-version",
                "0.16.1",
                "--core-dist-dir",
                "dist",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "format_version": "invarlock/distribution-validation-v1",
        "ok": True,
        "distributions": [expected.__dict__],
    }


def test_main_reports_release_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    monkeypatch.setattr(
        validation_module,
        "validate_first_party_distributions",
        lambda **_kwargs: (_ for _ in ()).throw(ReleasePreflightError("invalid pair")),
    )

    with pytest.raises(SystemExit, match="2"):
        validation_module.main(
            [
                "--repo-root",
                str(tmp_path),
                "--expected-version",
                "0.16.1",
                "--core-dist-dir",
                "dist",
            ]
        )


def test_direct_script_entry_point_delegates_to_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(validation_module.__file__), "--help"])
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(str(validation_module.__file__), run_name="__main__")
