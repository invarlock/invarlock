from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

_VERSION = "0.12.1"
_SHA = "a" * 40


def _load_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "release" / "release_preflight.py"
    spec = importlib.util.spec_from_file_location("release_preflight_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module() -> ModuleType:
    return _load_module()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_distribution_pair(root: Path) -> tuple[Path, Path, Path]:
    package_source = root / "src" / "invarlock"
    package_source.mkdir(parents=True)
    init_source = f"__version__ = {_VERSION!r}\n".encode()
    (package_source / "__init__.py").write_bytes(init_source)
    contract_source = b'{"policy": "trusted"}\n'
    contract_path = package_source / "_data" / "contracts" / "policy.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(contract_source)
    pyproject = (
        f"[project]\nname = 'invarlock'\nversion = '{_VERSION}'\n"
        "\n[project.scripts]\n"
        'invarlock = "invarlock.cli.app:app"\n'
    ).encode()
    entry_points = b"[console_scripts]\ninvarlock = invarlock.cli.app:app\n"
    (root / "pyproject.toml").write_bytes(pyproject)
    dist = root / "dist"
    dist.mkdir()
    wheel = dist / f"invarlock-{_VERSION}-py3-none-any.whl"
    wheel_files = {
        "invarlock/__init__.py": init_source,
        "invarlock/_data/contracts/policy.json": contract_source,
        f"invarlock-{_VERSION}.dist-info/METADATA": (
            f"Metadata-Version: 2.3\nName: invarlock\nVersion: {_VERSION}\n"
        ).encode(),
        f"invarlock-{_VERSION}.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"invarlock-{_VERSION}.dist-info/entry_points.txt": entry_points,
    }
    record_name = f"invarlock-{_VERSION}.dist-info/RECORD"
    record_lines = [
        ",".join(
            (
                name,
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                .decode("ascii")
                .rstrip("="),
                str(len(content)),
            )
        )
        for name, content in wheel_files.items()
    ]
    record_lines.append(f"{record_name},,")
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in wheel_files.items():
            archive.writestr(name, content)
        archive.writestr(record_name, "\n".join(record_lines) + "\n")
    sdist = dist / f"invarlock-{_VERSION}.tar.gz"
    metadata = (
        f"Metadata-Version: 2.3\nName: invarlock\nVersion: {_VERSION}\n"
    ).encode()
    with tarfile.open(sdist, "w:gz") as archive:
        project_member = tarfile.TarInfo(f"invarlock-{_VERSION}/pyproject.toml")
        project_member.size = len(pyproject)
        archive.addfile(project_member, io.BytesIO(pyproject))
        source_member = tarfile.TarInfo(
            f"invarlock-{_VERSION}/src/invarlock/__init__.py"
        )
        source_member.size = len(init_source)
        archive.addfile(source_member, io.BytesIO(init_source))
        contract_member = tarfile.TarInfo(
            f"invarlock-{_VERSION}/src/invarlock/_data/contracts/policy.json"
        )
        contract_member.size = len(contract_source)
        archive.addfile(contract_member, io.BytesIO(contract_source))
        member = tarfile.TarInfo(f"invarlock-{_VERSION}/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))
        egg_info_member = tarfile.TarInfo(
            f"invarlock-{_VERSION}/src/invarlock.egg-info/PKG-INFO"
        )
        egg_info_member.size = len(metadata)
        archive.addfile(egg_info_member, io.BytesIO(metadata))
        entry_point_member = tarfile.TarInfo(
            f"invarlock-{_VERSION}/src/invarlock.egg-info/entry_points.txt"
        )
        entry_point_member.size = len(entry_points)
        archive.addfile(entry_point_member, io.BytesIO(entry_points))
    manifest = root / "wheel-sdist-hashes.txt"
    manifest.write_text(
        f"{_digest(wheel)}  {wheel.name}\n{_digest(sdist)}  {sdist.name}\n",
        encoding="utf-8",
    )
    return dist, manifest, wheel


def _config(module: ModuleType, tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    dist, hashes, _ = _write_distribution_pair(checkout)
    return module.ReleasePreflightConfig(
        repo_root=checkout,
        release_sha=_SHA,
        expected_version=_VERSION,
        dist_dir=dist,
        hash_manifest=hashes,
    )


def _refresh_hash_manifest(config: object) -> None:
    dist_dir = config.dist_dir
    wheel = next(dist_dir.glob("*.whl"))
    sdist = next(dist_dir.glob("*.tar.gz"))
    config.hash_manifest.write_text(
        f"{_digest(wheel)}  {wheel.name}\n{_digest(sdist)}  {sdist.name}\n",
        encoding="utf-8",
    )


def _rewrite_wheel_with_extra(config: object, name: str, content: bytes) -> None:
    wheel = next(config.dist_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        files = {
            member.filename: archive.read(member)
            for member in archive.infolist()
            if not member.is_dir() and not member.filename.endswith(".dist-info/RECORD")
        }
    files[name] = content
    record_name = next(name for name in files if name.endswith(".dist-info/METADATA"))
    record_name = record_name.rsplit("/", 1)[0] + "/RECORD"
    record_lines = [
        ",".join(
            (
                member_name,
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(value).digest())
                .decode("ascii")
                .rstrip("="),
                str(len(value)),
            )
        )
        for member_name, value in files.items()
    ]
    record_lines.append(f"{record_name},,")
    with zipfile.ZipFile(wheel, "w") as archive:
        for member_name, value in files.items():
            archive.writestr(member_name, value)
        archive.writestr(record_name, "\n".join(record_lines) + "\n")
    _refresh_hash_manifest(config)


def _rewrite_sdist_with_extra(config: object, name: str, content: bytes) -> None:
    sdist = next(config.dist_dir.glob("*.tar.gz"))
    with tarfile.open(sdist, "r:gz") as archive:
        files = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isreg() and archive.extractfile(member) is not None
        }
    files[name] = content
    with tarfile.open(sdist, "w:gz") as archive:
        for member_name, value in files.items():
            member = tarfile.TarInfo(member_name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    _refresh_hash_manifest(config)


def _passing_import(module: ModuleType) -> object:
    return module.InstalledWheelImport(
        module_file=Path("/isolated/site-packages/invarlock/__init__.py"),
        module_version=_VERSION,
        distribution_name="invarlock",
        distribution_version=_VERSION,
        distribution_root=Path("/isolated/site-packages"),
    )


def test_release_preflight_runs_all_independent_gates(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    calls: list[str] = []

    def fake_git(_: Path, *args: str) -> str:
        calls.append("git:" + args[0])
        return _SHA if args[0] == "rev-parse" else ""

    def fake_probe(received: object, wheel: Path) -> object:
        assert received == config
        assert wheel == next(config.dist_dir.glob("*.whl"))
        calls.append("installed-wheel")
        return _passing_import(module)

    def fake_negative_audit(received: object) -> None:
        assert received == config
        calls.append("negative-audit")

    monkeypatch.setattr(module, "_git_output", fake_git)
    monkeypatch.setattr(module, "_probe_installed_wheel", fake_probe)
    monkeypatch.setattr(
        module, "_run_current_negative_evidence_audit", fake_negative_audit
    )

    summary = module.run_release_preflight(config)

    assert calls == [
        "git:rev-parse",
        "git:status",
        "installed-wheel",
        "negative-audit",
        "git:rev-parse",
        "git:status",
    ]
    assert summary["ok"] is True
    assert summary["installed_wheel_import"] == "isolated_venv_from_candidate_wheel"
    assert summary["current_negative_evidence"] == "passed"


def test_release_preflight_rejects_plain_text_shape_artifacts(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    wheel = next(config.dist_dir.glob("*.whl"))
    wheel.write_text("wheel", encoding="utf-8")
    config.hash_manifest.write_text(
        f"{_digest(wheel)}  {wheel.name}\n"
        f"{_digest(next(config.dist_dir.glob('*.tar.gz')))}  "
        f"{next(config.dist_dir.glob('*.tar.gz')).name}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_git_output",
        lambda _root, *args: _SHA if args[0] == "rev-parse" else "",
    )

    with pytest.raises(module.ReleasePreflightError, match="readable wheel archive"):
        module.run_release_preflight(config)


def test_distribution_hash_manifest_rejects_tampered_artifact(
    module: ModuleType, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    wheel = next(config.dist_dir.glob("*.whl"))
    with wheel.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(module.ReleasePreflightError, match="does not match"):
        module.validate_distributions(config)


def test_distribution_sources_must_match_the_exact_checkout(
    module: ModuleType, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    (config.repo_root / "src" / "invarlock" / "__init__.py").write_text(
        "__version__ = 'changed'\n", encoding="utf-8"
    )

    with pytest.raises(module.ReleasePreflightError, match="do not match"):
        module.validate_distributions(config)


def test_distribution_package_data_must_match_the_exact_checkout(
    module: ModuleType, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    contract = (
        config.repo_root / "src" / "invarlock" / "_data" / "contracts" / "policy.json"
    )
    contract.write_text('{"policy": "changed"}\n', encoding="utf-8")

    with pytest.raises(module.ReleasePreflightError, match="do not match"):
        module.validate_distributions(config)


def test_metadata_only_zip_cannot_pass_as_a_wheel(
    module: ModuleType, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    wheel = next(config.dist_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"invarlock-{_VERSION}.dist-info/METADATA",
            f"Metadata-Version: 2.3\nName: invarlock\nVersion: {_VERSION}\n",
        )
    sdist = next(config.dist_dir.glob("*.tar.gz"))
    config.hash_manifest.write_text(
        f"{_digest(wheel)}  {wheel.name}\n{_digest(sdist)}  {sdist.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(module.ReleasePreflightError, match="missing required"):
        module.validate_distributions(config)


@pytest.mark.parametrize(
    ("name", "expected_error"),
    (
        ("invarlock.pth", "must not contain .pth"),
        (f"invarlock-{_VERSION}.data/scripts/evil", "must not contain .data"),
        ("another_package/__init__.py", "unexpected top-level"),
        (f"invarlock-{_VERSION}.dist-info/evil.py", "dist-info must not contain"),
    ),
)
def test_wheel_rejects_extra_runtime_or_import_payloads(
    module: ModuleType,
    tmp_path: Path,
    name: str,
    expected_error: str,
) -> None:
    config = _config(module, tmp_path)
    _rewrite_wheel_with_extra(config, name, b"malicious payload\n")

    with pytest.raises(module.ReleasePreflightError, match=expected_error):
        module.validate_distributions(config)


@pytest.mark.parametrize(
    ("name", "expected_error"),
    (
        (f"invarlock-{_VERSION}/evil.pth", "must not contain .pth"),
        (
            f"invarlock-{_VERSION}/src/another_package/__init__.py",
            "unexpected source package",
        ),
        (
            f"invarlock-{_VERSION}/src/invarlock.egg-info/evil.py",
            "egg-info must not contain",
        ),
        (f"invarlock-{_VERSION}/unexpected.py", "supplemental source"),
    ),
)
def test_sdist_rejects_unexpected_or_executable_payloads(
    module: ModuleType,
    tmp_path: Path,
    name: str,
    expected_error: str,
) -> None:
    config = _config(module, tmp_path)
    _rewrite_sdist_with_extra(config, name, b"malicious payload\n")

    with pytest.raises(module.ReleasePreflightError, match=expected_error):
        module.validate_distributions(config)


def test_sdist_allows_only_inert_generated_setup_cfg(
    module: ModuleType, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    _rewrite_sdist_with_extra(
        config,
        f"invarlock-{_VERSION}/setup.cfg",
        b"[egg_info]\ntag_build = \ntag_date = 0\n",
    )

    artifacts = module.validate_distributions(config)

    assert artifacts.sdist.is_file()
    assert artifacts.hashes[artifacts.sdist.name] == _digest(artifacts.sdist)


def test_sdist_rejects_generated_setup_cfg_with_build_configuration(
    module: ModuleType, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    _rewrite_sdist_with_extra(
        config,
        f"invarlock-{_VERSION}/setup.cfg",
        b"[options]\npackages = find:\n",
    )

    with pytest.raises(
        module.ReleasePreflightError, match="setup.cfg contains unsupported"
    ):
        module.validate_distributions(config)


@pytest.mark.parametrize("artifact", ("wheel", "sdist"))
def test_distribution_entry_points_must_match_the_exact_checkout(
    module: ModuleType, tmp_path: Path, artifact: str
) -> None:
    config = _config(module, tmp_path)
    malicious_entry_points = b"[console_scripts]\ninvarlock = invarlock.evil:main\n"
    if artifact == "wheel":
        _rewrite_wheel_with_extra(
            config,
            f"invarlock-{_VERSION}.dist-info/entry_points.txt",
            malicious_entry_points,
        )
    else:
        _rewrite_sdist_with_extra(
            config,
            f"invarlock-{_VERSION}/src/invarlock.egg-info/entry_points.txt",
            malicious_entry_points,
        )

    with pytest.raises(
        module.ReleasePreflightError, match=f"{artifact} entry points do not match"
    ):
        module.validate_distributions(config)


def test_checkout_must_match_exact_clean_release_sha(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    monkeypatch.setattr(module, "_git_output", lambda *_: "b" * 40)

    with pytest.raises(module.ReleasePreflightError, match="does not match"):
        module.validate_clean_exact_checkout(config)


def test_candidate_wheel_is_installed_in_a_disposable_isolated_environment(
    module: ModuleType, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    artifacts = module.validate_distributions(config)

    imported = module._probe_installed_wheel(config, artifacts.wheel)

    assert imported.module_version == _VERSION
    assert imported.distribution_version == _VERSION
    assert config.repo_root.resolve() not in imported.module_file.parents
    assert config.repo_root.resolve() not in imported.distribution_root.parents


def test_negative_evidence_audit_is_pinned_to_canonical_checkout_tree(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    commands: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE", "1")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")

    module._run_current_negative_evidence_audit(config)

    command, kwargs = commands[0]
    assert command[command.index("--root") + 1] == str(
        (config.repo_root / "public_evidence").resolve()
    )
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE" not in environment
    assert environment["PYTHONPATH"] == str(config.repo_root / "src")


def test_make_release_preflight_is_distinct_from_artifact_shape_check() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")
    preflight = text.split("release-preflight:", 1)[1].split(
        "\nguard-validation-smoke:", 1
    )[0]

    assert "scripts/release/release_preflight.py" in preflight
    assert "RELEASE_PREFLIGHT_ARGS is required" in preflight
    assert "evidence_contracts.py" not in preflight
    assert ".PHONY: release-preflight" in text


def test_post_gate_checkout_recheck_rejects_source_toctou(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    status_calls = 0

    def fake_git(_: Path, *args: str) -> str:
        nonlocal status_calls
        if args[0] == "rev-parse":
            return _SHA
        status_calls += 1
        return "" if status_calls == 1 else " M src/invarlock/__init__.py"

    monkeypatch.setattr(module, "_git_output", fake_git)
    monkeypatch.setattr(
        module, "_probe_installed_wheel", lambda *_: _passing_import(module)
    )
    monkeypatch.setattr(module, "_run_current_negative_evidence_audit", lambda _: None)

    with pytest.raises(module.ReleasePreflightError, match="not clean"):
        module.run_release_preflight(config)


def test_installed_wheel_import_rejects_checkout_source(
    module: ModuleType, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    imported = module.InstalledWheelImport(
        module_file=config.repo_root / "src" / "invarlock" / "__init__.py",
        module_version=_VERSION,
        distribution_name="invarlock",
        distribution_version=_VERSION,
        distribution_root=config.repo_root / "src",
    )

    with pytest.raises(
        module.ReleasePreflightError, match="inside the release checkout"
    ):
        module.validate_installed_wheel_import(config, imported)


def test_preflight_summary_has_no_checkout_paths(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(module, tmp_path)
    monkeypatch.setattr(
        module,
        "_git_output",
        lambda _root, *args: _SHA if args[0] == "rev-parse" else "",
    )
    monkeypatch.setattr(
        module, "_probe_installed_wheel", lambda *_: _passing_import(module)
    )
    monkeypatch.setattr(module, "_run_current_negative_evidence_audit", lambda _: None)

    payload = json.dumps(module.run_release_preflight(config), sort_keys=True)

    assert str(config.repo_root) not in payload
