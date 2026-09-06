from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.release import first_party_distribution_validation as validation_module
from scripts.release import release_distribution_validation as distribution_validation
from scripts.release.first_party_distribution_validation import (
    FirstPartyDistribution,
    _artifact_pair,
    _contained_distribution_directory,
    _real_directory,
    _validate_artifact_directory,
    validate_first_party_addin_distributions,
    validate_first_party_distributions,
)
from scripts.release.release_distribution_validation import ReleasePreflightError

ROOT = Path(__file__).resolve().parents[2]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]["version"]


@pytest.fixture(scope="module")
def built_addins(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fixture_root = tmp_path_factory.mktemp("first-party-addins")
    dist = fixture_root / "dist"
    dist.mkdir()
    sources = fixture_root / "sources"
    sources.mkdir()
    for project in ("diagnostics", "gguf", "multimodal", "tensorrt_llm"):
        isolated_source = sources / project
        shutil.copytree(
            ROOT / "addins" / project,
            isolated_source,
            ignore=shutil.ignore_patterns(
                ".DS_Store",
                ".venv*",
                "__pycache__",
                "*.egg-info",
                "build",
                "dist",
            ),
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                str(dist),
                str(isolated_source),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    return dist


@pytest.fixture(scope="module")
def built_core(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fixture_root = tmp_path_factory.mktemp("first-party-core")
    isolated_source = fixture_root / "source"
    shutil.copytree(
        ROOT,
        isolated_source,
        ignore=shutil.ignore_patterns(
            ".addins-smoke-*",
            ".coverage*",
            ".DS_Store",
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv*",
            "__pycache__",
            "*.egg-info",
            "artifacts",
            "build",
            "dist",
            "reports",
            "site",
        ),
    )
    dist = fixture_root / "dist"
    dist.mkdir()
    metadata_paths = [
        isolated_source / directory / name
        for directory in (".", "src", "src/invarlock", "src/invarlock/_data/contracts")
        for name in (".DS_Store", "._payload", "Thumbs.db", "desktop.ini")
    ]
    for path in metadata_paths:
        path.write_bytes(b"desktop metadata")
    # A direct wheel build must also exclude metadata before sdist filtering.
    direct_wheel = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist),
            str(isolated_source),
        ],
        cwd=isolated_source,
        check=False,
        capture_output=True,
        text=True,
    )
    assert direct_wheel.returncode == 0, direct_wheel.stderr
    with zipfile.ZipFile(next(dist.glob("*.whl"))) as archive:
        assert not any(
            distribution_validation._is_os_metadata_path(name)
            for name in archive.namelist()
        )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(dist),
            str(isolated_source),
        ],
        cwd=isolated_source,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    with tarfile.open(next(dist.glob("*.tar.gz")), "r:gz") as archive:
        assert not any(
            distribution_validation._is_os_metadata_path(name)
            for name in archive.getnames()
        )
    assert all(path.read_bytes() == b"desktop metadata" for path in metadata_paths)
    return dist


def test_first_party_addin_artifacts_match_exact_source(
    built_addins: Path,
) -> None:
    results = validate_first_party_addin_distributions(
        repo_root=ROOT,
        expected_version=VERSION,
        dist_dir=built_addins,
    )

    assert {result.project for result in results} == {
        "diagnostics",
        "gguf",
        "multimodal",
        "tensorrt_llm",
    }


def test_first_party_artifacts_include_core_and_all_addins(
    built_core: Path, built_addins: Path
) -> None:
    results = validate_first_party_distributions(
        repo_root=ROOT,
        expected_version=VERSION,
        core_dist_dir=built_core,
        addin_dist_dir=built_addins,
    )

    assert [result.project for result in results] == [
        "core",
        "diagnostics",
        "gguf",
        "multimodal",
        "tensorrt_llm",
    ]
    assert len({result.distribution for result in results}) == 5


def test_first_party_artifacts_reject_core_namespace_injection(
    built_core: Path, built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "core"
    shutil.copytree(built_core, copied)
    wheel = next(copied.glob("invarlock-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        files = {
            member.filename: archive.read(member)
            for member in archive.infolist()
            if not member.is_dir()
        }
    files["invarlock/injected.py"] = b"EXECUTED = True\n"
    _rewrite_record(files)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    with pytest.raises(ReleasePreflightError, match="exact checkout"):
        validate_first_party_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            core_dist_dir=copied,
            addin_dist_dir=built_addins,
        )


def test_first_party_addin_rejects_a_symlinked_distribution_directory(
    built_addins: Path, tmp_path: Path
) -> None:
    linked = tmp_path / "linked-addins"
    linked.symlink_to(built_addins, target_is_directory=True)

    with pytest.raises(ReleasePreflightError, match="symbolic link"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            dist_dir=linked,
        )


def test_cli_distribution_directory_must_remain_inside_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ReleasePreflightError, match="inside the checkout"):
        _contained_distribution_directory(checkout, outside)


def test_contained_distribution_directory_accepts_relative_real_directory(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    dist = checkout / "dist"
    dist.mkdir(parents=True)

    assert _contained_distribution_directory(checkout, Path("dist")) == dist


def test_real_directory_rejects_missing_path_and_regular_file(tmp_path: Path) -> None:
    with pytest.raises(ReleasePreflightError, match="missing"):
        _real_directory(tmp_path / "missing", label="artifact directory")

    regular_file = tmp_path / "regular"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ReleasePreflightError, match="real directory"):
        _real_directory(regular_file, label="artifact directory")


def test_artifact_directory_rejects_wrong_count_and_symlink(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ReleasePreflightError, match="1 wheel/sdist pair"):
        _validate_artifact_directory(
            empty, expected_pairs=1, label="core distribution directory"
        )

    linked = empty / "linked.whl"
    linked.symlink_to(tmp_path / "missing-wheel")
    with pytest.raises(ReleasePreflightError, match="symbolic links"):
        _validate_artifact_directory(
            empty, expected_pairs=1, label="core distribution directory"
        )


def test_artifact_pair_rejects_unmatched_distribution(tmp_path: Path) -> None:
    with pytest.raises(ReleasePreflightError, match="artifact pair is ambiguous"):
        _artifact_pair(dist_dir=tmp_path, distribution_name="invarlock")


def test_distribution_versions_must_match_release(
    built_core: Path, built_addins: Path
) -> None:
    with pytest.raises(ReleasePreflightError, match="core version"):
        validate_first_party_distributions(
            repo_root=ROOT,
            expected_version="0.0.0",
            core_dist_dir=built_core,
            addin_dist_dir=built_addins,
        )

    with pytest.raises(ReleasePreflightError, match="add-in version"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version="0.0.0",
            dist_dir=built_addins,
        )


def test_distribution_names_must_be_unique(
    built_core: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = FirstPartyDistribution(
        project="duplicate",
        distribution="invarlock",
        version=VERSION,
        wheel="duplicate.whl",
        sdist="duplicate.tar.gz",
    )
    monkeypatch.setattr(
        validation_module,
        "validate_first_party_addin_distributions",
        lambda **_kwargs: [duplicate],
    )

    with pytest.raises(ReleasePreflightError, match="names must be unique"):
        validate_first_party_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            core_dist_dir=built_core,
            addin_dist_dir=built_core,
        )


def test_cli_emits_one_core_and_four_addin_results(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    core_dist = checkout / "dist"
    addin_dist = core_dist / "addins"
    addin_dist.mkdir(parents=True)
    expected = [
        FirstPartyDistribution(
            project="core",
            distribution="invarlock",
            version=VERSION,
            wheel=f"invarlock-{VERSION}-py3-none-any.whl",
            sdist=f"invarlock-{VERSION}.tar.gz",
        )
    ]
    expected.extend(
        FirstPartyDistribution(
            project=project,
            distribution=f"invarlock-{project}",
            version=VERSION,
            wheel=f"invarlock_{project}-{VERSION}-py3-none-any.whl",
            sdist=f"invarlock_{project}-{VERSION}.tar.gz",
        )
        for project in ("diagnostics", "gguf", "multimodal", "tensorrt_llm")
    )
    observed: dict[str, object] = {}

    def validate(**kwargs: object) -> list[FirstPartyDistribution]:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(
        validation_module, "validate_first_party_distributions", validate
    )
    monkeypatch.setattr(
        validation_module,
        "read_distribution_project",
        lambda _root: ("invarlock", VERSION),
    )

    assert (
        validation_module.main(
            [
                "--repo-root",
                str(checkout),
                "--core-dist-dir",
                "dist",
                "--addin-dist-dir",
                "dist/addins",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["format_version"] == "invarlock/distribution-validation-v1"
    assert payload["ok"] is True
    assert [item["project"] for item in payload["distributions"]] == [
        item.project for item in expected
    ]
    assert observed == {
        "repo_root": checkout,
        "expected_version": VERSION,
        "core_dist_dir": core_dist,
        "addin_dist_dir": addin_dist,
    }


def test_cli_reports_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "dist/addins").mkdir(parents=True)

    def reject(**_kwargs: object) -> list[FirstPartyDistribution]:
        raise ReleasePreflightError("distribution mismatch")

    monkeypatch.setattr(
        validation_module,
        "validate_first_party_distributions",
        reject,
    )
    monkeypatch.setattr(
        validation_module,
        "read_distribution_project",
        lambda _root: ("invarlock", VERSION),
    )

    with pytest.raises(SystemExit, match="2"):
        validation_module.main(
            [
                "--repo-root",
                str(checkout),
                "--core-dist-dir",
                "dist",
                "--addin-dist-dir",
                "dist/addins",
            ]
        )


def _rewrite_record(files: dict[str, bytes]) -> None:
    record_name = next(name for name in files if name.endswith(".dist-info/RECORD"))
    rows: list[list[str]] = []
    for name, content in sorted(files.items()):
        if name == record_name:
            rows.append([name, "", ""])
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode()
        rows.append([name, "sha256=" + digest.rstrip("="), str(len(content))])
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    files[record_name] = output.getvalue().encode()


def _read_wheel_files(wheel: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel) as archive:
        return {
            member.filename: archive.read(member)
            for member in archive.infolist()
            if not member.is_dir()
        }


def _write_wheel_files(wheel: Path, files: dict[str, bytes]) -> None:
    _rewrite_record(files)
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def test_first_party_addin_rejects_removed_wheel_dependency(
    built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "addins"
    shutil.copytree(built_addins, copied)
    wheel = next(copied.glob("invarlock_runtime_gguf-*.whl"))
    files = _read_wheel_files(wheel)
    metadata = next(name for name in files if name.endswith(".dist-info/METADATA"))
    files[metadata] = b"\n".join(
        line
        for line in files[metadata].split(b"\n")
        if not line.startswith(b"Requires-Dist: invarlock")
    )
    _write_wheel_files(wheel, files)

    with pytest.raises(ReleasePreflightError, match="wheel metadata Requires-Dist"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            dist_dir=copied,
        )


def test_first_party_addin_rejects_injected_wheel_dependency(
    built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "addins"
    shutil.copytree(built_addins, copied)
    wheel = next(copied.glob("invarlock_diagnostics-*.whl"))
    files = _read_wheel_files(wheel)
    metadata = next(name for name in files if name.endswith(".dist-info/METADATA"))
    headers, separator, body = files[metadata].partition(b"\n\n")
    assert separator
    files[metadata] = (
        headers + b"\nRequires-Dist: unbound-package>=1" + separator + body
    )
    _write_wheel_files(wheel, files)

    with pytest.raises(ReleasePreflightError, match="wheel metadata Requires-Dist"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            dist_dir=copied,
        )


def test_first_party_core_rejects_substituted_requires_python(
    built_core: Path, built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "core"
    shutil.copytree(built_core, copied)
    wheel = next(copied.glob("invarlock-*.whl"))
    files = _read_wheel_files(wheel)
    metadata = next(name for name in files if name.endswith(".dist-info/METADATA"))
    files[metadata] = files[metadata].replace(
        b"Requires-Python: >=3.12", b"Requires-Python: >=3.13", 1
    )
    _write_wheel_files(wheel, files)

    with pytest.raises(ReleasePreflightError, match="wheel metadata Requires-Python"):
        validate_first_party_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            core_dist_dir=copied,
            addin_dist_dir=built_addins,
        )


def test_first_party_addin_rejects_substituted_optional_dependency_marker(
    built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "addins"
    shutil.copytree(built_addins, copied)
    wheel = next(copied.glob("invarlock_runtime_hf_vision_text-*.whl"))
    files = _read_wheel_files(wheel)
    metadata = next(name for name in files if name.endswith(".dist-info/METADATA"))
    assert b'extra == "runtime"' in files[metadata]
    files[metadata] = files[metadata].replace(
        b'extra == "runtime"', b'extra == "unbound"', 1
    )
    _write_wheel_files(wheel, files)

    with pytest.raises(ReleasePreflightError, match="wheel metadata Requires-Dist"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            dist_dir=copied,
        )


def test_first_party_addin_rejects_substituted_sdist_dependency(
    built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "addins"
    shutil.copytree(built_addins, copied)
    sdist = next(copied.glob("invarlock_runtime_gguf-*.tar.gz"))
    rewritten = sdist.with_suffix(".rewritten")
    with (
        tarfile.open(sdist, "r:gz") as source,
        tarfile.open(rewritten, "w:gz") as destination,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member) if member.isreg() else None
            payload = extracted.read() if extracted is not None else None
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO"):
                assert payload is not None
                expected = f"Requires-Dist: invarlock=={VERSION}".encode()
                assert expected in payload
                payload = payload.replace(
                    expected,
                    b"Requires-Dist: invarlock==99.0.0",
                    1,
                )
                member.size = len(payload)
            destination.addfile(
                member,
                io.BytesIO(payload) if payload is not None else None,
            )
    rewritten.replace(sdist)

    with pytest.raises(ReleasePreflightError, match="sdist metadata Requires-Dist"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            dist_dir=copied,
        )


def test_first_party_addin_rejects_validly_recorded_source_substitution(
    built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "addins"
    shutil.copytree(built_addins, copied)
    wheel = next(copied.glob("invarlock_diagnostics-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        files = {
            member.filename: archive.read(member)
            for member in archive.infolist()
            if not member.is_dir()
        }
    source = "invarlock_addins/diagnostics/observations.py"
    files[source] += b"\nSUBSTITUTED = True\n"
    _rewrite_record(files)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    with pytest.raises(ReleasePreflightError, match="exact checkout"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            dist_dir=copied,
        )


def test_first_party_addin_rejects_validly_recorded_namespace_injection(
    built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "addins"
    shutil.copytree(built_addins, copied)
    wheel = next(copied.glob("invarlock_diagnostics-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        files = {
            member.filename: archive.read(member)
            for member in archive.infolist()
            if not member.is_dir()
        }
    files["invarlock_addins/injected.py"] = b"EXECUTED = True\n"
    _rewrite_record(files)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    with pytest.raises(ReleasePreflightError, match="source-unbound namespace"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            dist_dir=copied,
        )


def test_first_party_addin_rejects_oversized_wheel_metadata(
    built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "addins"
    shutil.copytree(built_addins, copied)
    wheel = next(copied.glob("invarlock_diagnostics-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        files = {
            member.filename: archive.read(member)
            for member in archive.infolist()
            if not member.is_dir()
        }
    dist_info = next(
        name.rsplit("/", 1)[0] for name in files if name.endswith(".dist-info/METADATA")
    )
    files[f"{dist_info}/oversized.txt"] = b"x" * (1024 * 1024 + 1)
    _rewrite_record(files)
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    with pytest.raises(ReleasePreflightError, match="dist-info member is too large"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            dist_dir=copied,
        )


def test_first_party_addin_rejects_sdist_entry_point_substitution(
    built_addins: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "addins"
    shutil.copytree(built_addins, copied)
    sdist = next(copied.glob("invarlock_runtime_gguf-*.tar.gz"))
    rewritten = sdist.with_suffix(".rewritten")
    with (
        tarfile.open(sdist, "r:gz") as source,
        tarfile.open(rewritten, "w:gz") as destination,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member) if member.isreg() else None
            payload = extracted.read() if extracted is not None else None
            if member.name.endswith(".egg-info/entry_points.txt"):
                payload = (
                    b"[invarlock.runtime_providers]\n"
                    b"llama_cpp = invarlock_addins.gguf.provider:provider\n"
                    b"substituted = invarlock_addins.gguf.provider:provider\n"
                )
                member.size = len(payload)
            destination.addfile(
                member,
                io.BytesIO(payload) if payload is not None else None,
            )
    rewritten.replace(sdist)

    with pytest.raises(ReleasePreflightError, match="sdist entry points"):
        validate_first_party_addin_distributions(
            repo_root=ROOT,
            expected_version=VERSION,
            dist_dir=copied,
        )


@pytest.mark.parametrize(
    ("pyproject", "message"),
    [
        ("project = []\n", "project metadata is unreadable"),
        ('[project]\nname = ""\nversion = "1.0"\n', "project identity is invalid"),
        (
            '[project]\nname = "example"\nversion = "1.0"\nrequires-python = 1\n',
            "requires-python is invalid",
        ),
        (
            '[project]\nname = "example"\nversion = "1.0"\ndependencies = "bad"\n',
            "dependencies are invalid",
        ),
        (
            '[project]\nname = "example"\nversion = "1.0"\n'
            'optional-dependencies = "bad"\n',
            "optional dependencies are invalid",
        ),
        (
            '[project]\nname = "example"\nversion = "1.0"\n'
            '[project.optional-dependencies]\nfeature = "bad"\n',
            "optional dependencies are invalid",
        ),
        (
            '[project]\nname = "example"\nversion = "1.0"\n'
            "[project.optional-dependencies]\nFeature = []\nfeature = []\n",
            "names are ambiguous",
        ),
    ],
)
def test_checkout_package_metadata_rejects_malformed_project_tables(
    tmp_path: Path, pyproject: str, message: str
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(pyproject, encoding="utf-8")

    with pytest.raises(ReleasePreflightError, match=message):
        distribution_validation._expected_package_metadata(project_root)
