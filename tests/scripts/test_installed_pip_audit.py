from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import shutil
import stat
import zipfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.security import installed_audit_binding as binding
from scripts.security import run_pip_audit as audit

_ADVISORY = "GHSA-test-test-test"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record(files: dict[str, bytes], dist: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, data in sorted(files.items()):
        if name != dist + "/RECORD":
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .decode()
                .rstrip("=")
            )
            writer.writerow([name, "sha256=" + digest, str(len(data))])
    writer.writerow([dist + "/RECORD", "", ""])
    return stream.getvalue().encode()


def _wheel(path: Path, name: str, version: str) -> dict[str, bytes]:
    dist = f"{name}-{version}.dist-info"
    files = {
        name + "/__init__.py": b"VALUE = 1\n",
        dist
        + "/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
        dist
        + "/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        dist + "/top_level.txt": (name + "\n").encode(),
    }
    files[dist + "/RECORD"] = _record(files, dist)
    _archive(path, files)
    return files


def _archive(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def _install(root: Path, files: dict[str, bytes], dist: str) -> None:
    files = dict(files)
    files[dist + "/INSTALLER"] = b"pip\n"
    files[dist + "/REQUESTED"] = b""
    files[dist + "/RECORD"] = _record(files, dist)
    for name, data in files.items():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


@pytest.fixture
def surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "site"
    wheel = tmp_path / "accelerate-1.14.0-py3-none-any.whl"
    files = _wheel(wheel, "accelerate", "1.14.0")
    _install(root, files, "accelerate-1.14.0.dist-info")
    project_wheel = tmp_path / "invarlock-0.15.0-py3-none-any.whl"
    _install(
        root, _wheel(project_wheel, "invarlock", "0.15.0"), "invarlock-0.15.0.dist-info"
    )
    _install(root, _wheel(tmp_path / "pip.whl", "pip", "26.2"), "pip-26.2.dist-info")
    lock = tmp_path / "requirements/workflows/hf.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        f"accelerate==1.14.0 \\\n    --hash=sha256:{_digest(wheel.read_bytes())}\n"
    )
    bootstrap = lock.parent / "pip-bootstrap.txt"
    bootstrap.write_text("pip==26.2 --hash=sha256:" + "0" * 64 + "\n")
    policy = tmp_path / "allowlist.json"
    policy.write_text(
        json.dumps(
            {
                "owner": "test-owner",
                "entries": [
                    {
                        "advisory": _ADVISORY,
                        "packages": ["accelerate"],
                        "versions": ["1.14.0"],
                        "allowed_sources": ["requirements/workflows/hf.txt"],
                        "owner": "test-owner",
                        "expires": (date.today() + timedelta(days=14)).isoformat(),
                        "tracking_issue": "https://github.com/example/repo/issues/1",
                        "reason": "temporary test exception",
                        "compensating_control": "authenticated component and exact inventory",
                    }
                ],
            }
        )
    )
    report = tmp_path / "reports/audit.json"
    arguments = [
        "--allowlist",
        str(policy),
        "--path",
        str(root),
        "--installed-lock",
        "requirements/workflows/hf.txt",
        "--installed-lock-sha256",
        _digest(lock.read_bytes()),
        "--installed-bootstrap-lock",
        "requirements/workflows/pip-bootstrap.txt",
        "--installed-bootstrap-lock-sha256",
        _digest(bootstrap.read_bytes()),
        "--installed-wheel",
        str(wheel),
        "--installed-project-wheel",
        str(project_wheel),
        "--report",
        str(report),
    ]
    raw = {
        "dependencies": [
            {
                "name": name,
                "version": version,
                "vulns": (
                    [
                        {
                            "id": _ADVISORY,
                            "aliases": [],
                            "fix_versions": [],
                            "description": "retained original finding",
                        }
                    ]
                    if name == "accelerate"
                    else []
                ),
            }
            for name, version in [
                ("accelerate", "1.14.0"),
                ("invarlock", "0.15.0"),
                ("pip", "26.2"),
            ]
        ],
        "fixes": [],
    }
    calls = []

    def scanner(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": False, "capture_output": True, "timeout": 300}
        assert "--ignore-vuln" not in command
        return SimpleNamespace(
            returncode=1,
            stdout=json.dumps(raw).encode(),
            stderr=b"Found 1 known vulnerability\n",
        )

    monkeypatch.setattr(binding.subprocess, "run", scanner)
    return SimpleNamespace(
        root=root,
        wheel=wheel,
        files=files,
        dist="accelerate-1.14.0.dist-info",
        project_wheel=project_wheel,
        lock=lock,
        bootstrap=bootstrap,
        policy=policy,
        report=report,
        args=arguments,
        raw=raw,
        calls=calls,
    )


def _set(surface: SimpleNamespace, option: str, value: str) -> None:
    surface.args[surface.args.index(option) + 1] = value


def _refresh_lock(surface: SimpleNamespace) -> None:
    surface.lock.write_text(
        f"accelerate==1.14.0 --hash=sha256:{_digest(surface.wheel.read_bytes())}\n"
    )
    _set(surface, "--installed-lock-sha256", _digest(surface.lock.read_bytes()))


def test_exact_bound_finding_is_retained_and_separately_accepted(surface) -> None:
    assert audit.main(surface.args) == 0
    report = json.loads(surface.report.read_text())
    assert report["status"] == "accepted_exception"
    assert report["raw_findings"] == surface.raw
    assert json.loads(base64.b64decode(report["raw_stdout_base64"])) == surface.raw
    assert (
        report["accepted_findings"][0]["finding"]
        == surface.raw["dependencies"][0]["vulns"][0]
    )
    assert report["blocking_findings"] == []
    assert report["binding"]["installed_distribution_inventory"] == {
        "accelerate": "1.14.0",
        "invarlock": "0.15.0",
        "pip": "26.2",
    }
    assert surface.calls == [
        ["pip-audit", "--path", str(surface.root), "--format", "json"]
    ]


@pytest.mark.parametrize("existing", [False, True])
def test_report_is_private_on_creation_and_replacement(surface, existing) -> None:
    if existing:
        surface.report.parent.mkdir()
        surface.report.write_text("previous report" * 50000)
        surface.report.chmod(0o666)
    assert audit.main(surface.args) == 0
    assert stat.S_IMODE(surface.report.stat().st_mode) == 0o600
    assert json.loads(surface.report.read_text())["status"] == "accepted_exception"


@pytest.mark.parametrize(
    "mutation",
    [
        "bytes",
        "missing",
        "extra",
        "pyc",
        "symlink-file",
        "symlink-root",
        "symlink-parent",
        "duplicate",
        "egg-info",
        "alternate-module",
        "metadata",
        "record",
        "installer",
        "requested",
        "direct-url",
        "missing-distribution",
        "extra-distribution",
        "wrong-version",
        "project-metadata",
        "lock-sha",
        "bootstrap-sha",
        "wheel-sha",
        "unapproved-source",
        "absolute-lock",
        "expired",
    ],
)
def test_binding_mutations_block_before_scanner(surface, mutation, tmp_path) -> None:
    target = surface.root / "accelerate/__init__.py"
    metadata = surface.root / surface.dist
    if mutation == "bytes":
        target.write_bytes(b"tampered\n")
    elif mutation == "missing":
        target.unlink()
    elif mutation in {"extra", "pyc"}:
        (
            target.parent / ("extra.pyc" if mutation == "pyc" else "extra.py")
        ).write_bytes(b"extra")
    elif mutation == "symlink-file":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(replacement)
    elif mutation in {"symlink-root", "symlink-parent"}:
        alias = tmp_path / "alias"
        alias.symlink_to(
            surface.root if mutation == "symlink-root" else tmp_path,
            target_is_directory=True,
        )
        _set(
            surface,
            "--path",
            str(alias if mutation == "symlink-root" else alias / "site"),
        )
    elif mutation == "duplicate":
        shutil.copytree(metadata, surface.root / "alias-1.14.0.dist-info")
    elif mutation == "egg-info":
        (surface.root / "accelerate.egg-info").mkdir()
    elif mutation == "alternate-module":
        (surface.root / "accelerate.py").write_bytes(b"extra")
    elif mutation == "metadata":
        (metadata / "METADATA").write_bytes(
            b"Name: accelerate\nName: evil\nVersion: 1.14.0\n"
        )
    elif mutation == "record":
        (metadata / "RECORD").write_bytes(b"wrong,,\n")
    elif mutation == "installer":
        (metadata / "INSTALLER").write_bytes(b"unknown\n")
    elif mutation == "requested":
        (metadata / "REQUESTED").write_bytes(b"unexpected")
    elif mutation == "direct-url":
        (metadata / "direct_url.json").write_text('{"dir_info":{"editable":true}}')
    elif mutation == "missing-distribution":
        shutil.rmtree(surface.root / "pip-26.2.dist-info")
    elif mutation == "extra-distribution":
        _install(
            surface.root,
            _wheel(tmp_path / "extra.whl", "plugin", "1"),
            "plugin-1.dist-info",
        )
    elif mutation == "wrong-version":
        (surface.root / "pip-26.2.dist-info/METADATA").write_bytes(
            b"Name: pip\nVersion: 26.3\n"
        )
    elif mutation == "project-metadata":
        path = surface.root / "invarlock-0.15.0.dist-info/METADATA"
        path.write_bytes(path.read_bytes() + b"Requires-Dist: evil\n")
    elif mutation in {"lock-sha", "bootstrap-sha"}:
        _set(
            surface,
            "--installed-lock-sha256"
            if mutation == "lock-sha"
            else "--installed-bootstrap-lock-sha256",
            "0" * 64,
        )
    elif mutation == "wheel-sha":
        surface.wheel.write_bytes(surface.wheel.read_bytes() + b"changed")
    elif mutation == "unapproved-source":
        other = surface.lock.parent / "other.txt"
        other.write_bytes(surface.lock.read_bytes())
        _set(surface, "--installed-lock", "requirements/workflows/other.txt")
    elif mutation == "absolute-lock":
        _set(surface, "--installed-lock", str(surface.lock))
    elif mutation == "expired":
        policy = json.loads(surface.policy.read_text())
        policy["entries"][0]["expires"] = (date.today() - timedelta(days=1)).isoformat()
        surface.policy.write_text(json.dumps(policy))
    assert audit.main(surface.args) == 1
    report = json.loads(surface.report.read_text())
    assert report["status"] == "blocked" and report["error"]
    assert surface.calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        "record-hash",
        "record-inventory",
        "record-duplicate",
        "metadata-name",
        "metadata-version",
        "relocation",
        "pth",
        "top-level",
        "zip-duplicate",
        "zip-symlink",
    ],
)
def test_authenticated_but_invalid_wheels_are_rejected(surface, mutation) -> None:
    files = dict(surface.files)
    if mutation == "record-hash":
        files["accelerate/__init__.py"] += b"changed"
    elif mutation == "record-inventory":
        files[surface.dist + "/RECORD"] = b"not-present,,\n"
    elif mutation == "record-duplicate":
        files[surface.dist + "/RECORD"] += files[surface.dist + "/RECORD"].splitlines(
            keepends=True
        )[0]
    else:
        if mutation == "metadata-name":
            files[surface.dist + "/METADATA"] = b"Name: wrong\nVersion: 1.14.0\n"
        elif mutation == "metadata-version":
            files[surface.dist + "/METADATA"] = b"Name: accelerate\nVersion: 2\n"
        elif mutation == "relocation":
            files["accelerate-1.14.0.data/purelib/extra.py"] = b"extra"
        elif mutation == "pth":
            files["accelerate/extra.pth"] = b"import evil"
        elif mutation == "top-level":
            files[surface.dist + "/top_level.txt"] = b"accelerate\nother\n"
        files[surface.dist + "/RECORD"] = _record(files, surface.dist)
    _archive(surface.wheel, files)
    if mutation in {"zip-duplicate", "zip-symlink"}:
        with zipfile.ZipFile(surface.wheel, "a") as archive:
            if mutation == "zip-duplicate":
                with pytest.warns(UserWarning):
                    archive.writestr("accelerate/__init__.py", b"duplicate")
            else:
                info = zipfile.ZipInfo("accelerate/link")
                info.external_attr = 0o120777 << 16
                archive.writestr(info, b"target")
    _refresh_lock(surface)
    assert audit.main(surface.args) == 1
    assert not surface.calls


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-advisory",
        "alias-injection",
        "other-package",
        "wrong-version",
        "missing-package",
        "duplicate-package",
        "skipped",
        "invalid-json",
        "duplicate-json",
        "error-exit",
        "success-with-findings",
        "no-findings-error",
        "scanner-exception",
        "mid-scan-tamper",
    ],
)
def test_scanner_errors_and_unapproved_findings_fail_closed(
    surface, mutation, monkeypatch
) -> None:
    raw = surface.raw
    finding = raw["dependencies"][0]["vulns"][0]
    code = 1
    if mutation in {"unknown-advisory", "alias-injection"}:
        finding["id"] = "GHSA-unapproved"
        if mutation == "alias-injection":
            finding["aliases"] = [_ADVISORY]
    elif mutation == "other-package":
        raw["dependencies"][0]["vulns"] = []
        raw["dependencies"][1]["vulns"] = [finding]
    elif mutation == "wrong-version":
        raw["dependencies"][0]["version"] = "2"
    elif mutation == "missing-package":
        raw["dependencies"].pop()
    elif mutation == "duplicate-package":
        raw["dependencies"].append(raw["dependencies"][0])
    elif mutation == "skipped":
        raw["dependencies"][0] = {"name": "accelerate", "skip_reason": "unknown"}
    elif mutation == "error-exit":
        code = 2
    elif mutation == "success-with-findings":
        code = 0
    elif mutation == "no-findings-error":
        raw["dependencies"][0]["vulns"] = []
    stdout = json.dumps(raw).encode()
    if mutation == "invalid-json":
        stdout = b"not JSON"
    elif mutation == "duplicate-json":
        stdout = b'{"dependencies":[],"dependencies":[]}'

    def scanner(*_args, **_kwargs):
        if mutation == "scanner-exception":
            raise OSError("scanner unavailable")
        if mutation == "mid-scan-tamper":
            (surface.root / "accelerate/__init__.py").write_bytes(b"changed")
        return SimpleNamespace(
            returncode=code, stdout=stdout, stderr=b"scanner diagnostic"
        )

    monkeypatch.setattr(binding.subprocess, "run", scanner)
    assert audit.main(surface.args) == 1
    report = json.loads(surface.report.read_text())
    assert report["status"] == "blocked"
    if mutation != "scanner-exception":
        assert base64.b64decode(report["raw_stdout_base64"]) == stdout
    assert report["blocking_findings"] or report.get("error")


def test_clean_complete_scan_passes_without_accepted_findings(
    surface, monkeypatch
) -> None:
    surface.raw["dependencies"][0]["vulns"] = []
    monkeypatch.setattr(
        binding.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout=json.dumps(surface.raw).encode(), stderr=b""
        ),
    )
    assert audit.main(surface.args) == 0
    assert json.loads(surface.report.read_text())["status"] == "clean"


def test_installed_metadata_extras_have_exact_content(surface) -> None:
    dist = surface.root / surface.dist
    direct = {
        "url": surface.wheel.as_uri(),
        "archive_info": {
            "hash": "sha256=" + _digest(surface.wheel.read_bytes()),
            "hashes": {"sha256": _digest(surface.wheel.read_bytes())},
        },
    }
    (dist / "direct_url.json").write_text(json.dumps(direct))
    files = {
        p.relative_to(surface.root).as_posix(): p.read_bytes() for p in dist.iterdir()
    }
    files["accelerate/__init__.py"] = (
        surface.root / "accelerate/__init__.py"
    ).read_bytes()
    (dist / "RECORD").write_bytes(_record(files, surface.dist))
    assert audit.main(surface.args) == 0


def test_partial_bound_options_cannot_enable_an_exception(surface) -> None:
    index = surface.args.index("--installed-lock-sha256")
    del surface.args[index : index + 2]
    assert audit.main(surface.args) == 1
    assert not surface.calls


def test_unreadable_installed_subtree_blocks(surface, monkeypatch) -> None:
    def walk(_path, *, followlinks, onerror):
        assert followlinks is False
        onerror(PermissionError("unreadable unexpected subtree"))
        return iter(())

    monkeypatch.setattr(binding.os, "walk", walk)
    assert audit.main(surface.args) == 1
    assert "unreadable unexpected subtree" in surface.report.read_text()
    assert not surface.calls


def test_nonfinite_scanner_json_blocks_and_retains_raw(surface, monkeypatch) -> None:
    stdout = b'{"dependencies": [], "fixes": NaN}'
    monkeypatch.setattr(
        binding.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=1, stdout=stdout, stderr=b""),
    )
    assert audit.main(surface.args) == 1
    report = json.loads(surface.report.read_text())
    assert "non-finite" in report["error"]
    assert base64.b64decode(report["raw_stdout_base64"]) == stdout


def test_scanner_timeout_retains_partial_output(surface, monkeypatch) -> None:
    def scanner(*_args, **_kwargs):
        raise binding.subprocess.TimeoutExpired(
            "pip-audit", 300, output=b"partial", stderr=b"timeout"
        )

    monkeypatch.setattr(binding.subprocess, "run", scanner)
    assert audit.main(surface.args) == 1
    report = json.loads(surface.report.read_text())
    assert report["status"] == "blocked"
    assert base64.b64decode(report["raw_stdout_base64"]) == b"partial"
    assert base64.b64decode(report["raw_stderr_base64"]) == b"timeout"


def test_missing_report_cannot_enable_bound_mode(surface, capsys) -> None:
    index = surface.args.index("--report")
    del surface.args[index : index + 2]
    assert audit.main(surface.args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert not surface.calls


def test_declared_console_script_record_is_not_component_payload(surface) -> None:
    files = dict(surface.files)
    files["accelerate/utils/real.py"] = b"VALUE = 2\n"
    files[surface.dist + "/entry_points.txt"] = (
        b"[console_scripts]\naccelerate = accelerate:main\n"
    )
    files[surface.dist + "/RECORD"] = _record(files, surface.dist)
    _archive(surface.wheel, files)
    _refresh_lock(surface)
    _install(surface.root, files, surface.dist)
    record = surface.root / surface.dist / "RECORD"
    record.write_bytes(
        record.read_bytes() + b"../../bin/accelerate,sha256=generated,123\n"
    )
    assert audit.main(surface.args) == 0
    receipt = json.loads(surface.report.read_text())["binding"]
    assert receipt["generated_console_scripts_not_byte_bound"] == [
        "../../bin/accelerate"
    ]
    assert "accelerate/utils/real.py" in receipt["installed_component_files"]
    record.write_bytes(
        record.read_bytes() + b"../../bin/unapproved,sha256=generated,123\n"
    )
    assert audit.main(surface.args) == 1
    assert "unexpected installed RECORD entry" in surface.report.read_text()


def test_valid_install_without_optional_pip_metadata(surface) -> None:
    dist = surface.root / surface.dist
    (dist / "INSTALLER").unlink()
    (dist / "REQUESTED").unlink()
    (dist / "RECORD").write_bytes(_record(surface.files, surface.dist))
    assert audit.main(surface.args) == 0
    receipt = json.loads(surface.report.read_text())
    assert receipt["status"] == "accepted_exception"
    assert set(receipt["binding"]["installed_component_files"]) == set(surface.files)


def test_commented_multiline_lock_preserves_exact_artifact_binding(surface) -> None:
    surface.lock.write_text(
        "# Locked dependency generated for the approved workflow.\n\n"
        "accelerate==1.14.0 \\\n"
        "    # Retain the approved wheel digest.\n"
        f"    --hash=sha256:{_digest(surface.wheel.read_bytes())}\n"
    )
    _set(surface, "--installed-lock-sha256", _digest(surface.lock.read_bytes()))
    assert audit.main(surface.args) == 0
    receipt = json.loads(surface.report.read_text())
    assert receipt["binding"]["lock_sha256"] == _digest(surface.lock.read_bytes())
    assert receipt["binding"]["wheel_sha256"] == _digest(surface.wheel.read_bytes())


def test_wheel_directory_entries_preserve_exact_payload_inventory(surface) -> None:
    with zipfile.ZipFile(surface.wheel, "a") as archive:
        archive.writestr("accelerate/", b"")
        archive.writestr(surface.dist + "/", b"")
    _refresh_lock(surface)
    assert audit.main(surface.args) == 0
    receipt = json.loads(surface.report.read_text())
    expected = set(surface.files) | {
        surface.dist + "/INSTALLER",
        surface.dist + "/REQUESTED",
    }
    assert set(receipt["binding"]["installed_component_files"]) == expected


def test_report_descriptor_closes_when_stream_adoption_fails(surface, monkeypatch):
    import errno
    import os

    opened = []
    real_open = os.open

    def track_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == surface.report:
            opened.append(fd)
        return fd

    def fail_fdopen(*_args, **_kwargs):
        raise OSError(errno.EIO, "report stream failed")

    monkeypatch.setattr(binding.os, "open", track_open)
    monkeypatch.setattr(binding.os, "fdopen", fail_fdopen)
    with pytest.raises(OSError, match="report stream failed"):
        audit.main(surface.args)
    assert len(opened) == 1
    try:
        os.fstat(opened[0])
    except OSError as exc:
        assert exc.errno == errno.EBADF
    else:
        os.close(opened[0])
        pytest.fail("report descriptor leaked when stream creation failed")


@pytest.fixture
def hardened_surface(surface, monkeypatch):
    from scripts.security import hardened_accelerate_audit as hardened

    shutil.rmtree(surface.root / surface.dist)
    surface.dist = f"accelerate-{hardened.HARDENED_VERSION}.dist-info"
    surface.files = _wheel(surface.wheel, "accelerate", hardened.HARDENED_VERSION)
    _install(surface.root, surface.files, surface.dist)
    surface.lock.write_text(
        f"accelerate=={hardened.HARDENED_VERSION} --hash=sha256:{_digest(surface.wheel.read_bytes())}\n"
    )
    _set(surface, "--installed-lock-sha256", _digest(surface.lock.read_bytes()))
    surface.policy.write_text(json.dumps({"owner": "test", "entries": []}))
    surface.raw["dependencies"][0]["vulns"][0]["id"] = sorted(
        hardened.REMEDIATED_ADVISORIES
    )[0]
    trusted_data = surface.wheel.read_bytes()

    def verify(path):
        if path.read_bytes() != trusted_data:
            raise ValueError("hardened wheel differs from trusted derivation")
        return {
            "package": "accelerate",
            "version": hardened.HARDENED_VERSION,
            "upstream_version": hardened.UPSTREAM_VERSION,
            "wheel_sha256": _digest(trusted_data),
            "derivation": {"verified": True},
            "remediated_advisories": sorted(hardened.REMEDIATED_ADVISORIES),
        }

    monkeypatch.setattr(hardened, "verify_wheel", verify)
    surface.scan_requirements = []

    def scanner(command, **kwargs):
        surface.calls.append(command)
        surface.scan_requirements.append(
            Path(command[command.index("--requirement") + 1]).read_text()
        )
        assert "--no-deps" in command and "--disable-pip" in command
        assert "--ignore-vuln" not in command
        return SimpleNamespace(
            returncode=1,
            stdout=json.dumps(surface.raw).encode(),
            stderr=b"upstream finding",
        )

    monkeypatch.setattr(binding.subprocess, "run", scanner)
    return surface


def test_hardened_installed_audit_scans_upstream_identity_and_retains_raw(
    hardened_surface,
):
    surface = hardened_surface
    assert audit.main(surface.args) == 0
    report = json.loads(surface.report.read_text())
    assert report["status"] == "remediated"
    assert report["accepted_findings"] == []
    assert report["raw_findings"] == surface.raw
    assert (
        report["remediated_findings"][0]["finding"]
        == surface.raw["dependencies"][0]["vulns"][0]
    )
    assert (
        report["binding"]["installed_distribution_inventory"]["accelerate"]
        == "1.14.0+invarlock.1"
    )
    assert report["binding"]["scan_distribution_inventory"]["accelerate"] == "1.14.0"
    assert surface.scan_requirements == [
        "accelerate==1.14.0\ninvarlock==0.15.0\npip==26.2\n"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-id",
        "alias",
        "derived-scan",
        "skip",
        "modified-wheel",
        "modified-payload",
        "additional-hash",
        "missing-package",
    ],
)
def test_hardened_installed_remediation_fails_closed(hardened_surface, mutation):
    surface = hardened_surface
    finding = surface.raw["dependencies"][0]["vulns"][0]
    if mutation in {"unknown-id", "alias"}:
        if mutation == "alias":
            finding["aliases"] = [finding["id"]]
        finding["id"] = "GHSA-new-advisory"
    elif mutation == "derived-scan":
        surface.raw["dependencies"][0]["version"] = "1.14.0+invarlock.1"
    elif mutation == "skip":
        surface.raw["dependencies"][0] = {
            "name": "accelerate",
            "skip_reason": "unknown local version",
        }
    elif mutation == "modified-wheel":
        surface.wheel.write_bytes(surface.wheel.read_bytes() + b"changed")
        surface.lock.write_text(
            f"accelerate==1.14.0+invarlock.1 --hash=sha256:{_digest(surface.wheel.read_bytes())}\n"
        )
        _set(surface, "--installed-lock-sha256", _digest(surface.lock.read_bytes()))
    elif mutation == "modified-payload":
        (surface.root / "accelerate/__init__.py").write_bytes(b"changed")
    elif mutation == "additional-hash":
        surface.lock.write_text(
            surface.lock.read_text().strip() + " --hash=sha256:" + "0" * 64 + "\n"
        )
        _set(surface, "--installed-lock-sha256", _digest(surface.lock.read_bytes()))
    elif mutation == "missing-package":
        surface.raw["dependencies"].pop()
    assert audit.main(surface.args) == 1
    report = json.loads(surface.report.read_text())
    assert report["status"] == "blocked"
    assert report.get("error") or report["blocking_findings"]


def test_plain_path_cannot_skip_local_accelerate(hardened_surface):
    with pytest.raises(SystemExit, match="requires all installed binding"):
        audit.main(
            [
                "--path",
                str(hardened_surface.root),
                "--allowlist",
                str(hardened_surface.policy),
            ]
        )
    assert hardened_surface.calls == []


def test_source_only_lock_cannot_authorize_installed_runtime(hardened_surface):
    surface = hardened_surface
    source = surface.lock.parent / "accelerate-upstream-wheel.txt"
    source.write_bytes(surface.lock.read_bytes())
    _set(
        surface,
        "--installed-lock",
        "requirements/workflows/accelerate-upstream-wheel.txt",
    )
    assert audit.main(surface.args) == 1
    assert "source-only wheel lock is not a runtime" in surface.report.read_text()
    assert surface.calls == []


def test_inventory_normalizes_distribution_spelling_but_rejects_ambiguity(tmp_path):
    root = tmp_path / "site"
    metadata = root / "jaraco.classes-3.4.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Name: jaraco.classes\nVersion: 3.4.0\n")
    assert binding._inventory(root)[0] == {"jaraco-classes": "3.4.0"}
    alias = root / "jaraco_classes-3.4.0.dist-info"
    shutil.copytree(metadata, alias)
    with pytest.raises(ValueError, match="duplicate installed distribution"):
        binding._inventory(root)
    shutil.rmtree(alias)
    (metadata / "METADATA").write_text("Name: other\nVersion: 3.4.0\n")
    with pytest.raises(ValueError, match="installed dist-info identity mismatch"):
        binding._inventory(root)


def _unpublished_project(surface):
    surface.raw["dependencies"][1] = {
        "name": "invarlock",
        "skip_reason": "Dependency not found on PyPI and could not be audited: invarlock (0.15.0)",
    }


@pytest.mark.parametrize("kind", ["exception", "remediation"])
def test_unpublished_project_retains_exact_bound_identity(request, kind):
    surface = request.getfixturevalue(
        "hardened_surface" if kind == "remediation" else "surface"
    )
    _unpublished_project(surface)
    assert audit.main(surface.args) == 0
    report = json.loads(surface.report.read_text())
    assert report["raw_findings"] == surface.raw
    assert report["unpublished_project"][0]["name"] == "invarlock"
    assert report["unpublished_project"][0]["version"] == "0.15.0"
    assert report["unpublished_project"][0]["wheel_sha256"] == _digest(
        surface.project_wheel.read_bytes()
    )
    assert "no PyPI vulnerability result" in report["unpublished_project"][0]["scope"]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-version",
        "other-name",
        "unknown-reason",
        "extra-field",
        "duplicate",
        "installed-tamper",
    ],
)
def test_unpublished_project_does_not_allow_arbitrary_skips(surface, mutation):
    _unpublished_project(surface)
    record = surface.raw["dependencies"][1]
    if mutation == "wrong-version":
        record["skip_reason"] = record["skip_reason"].replace("0.15.0", "0.16.0")
    elif mutation == "other-name":
        record["name"] = "other"
    elif mutation == "unknown-reason":
        record["skip_reason"] = "network error"
    elif mutation == "extra-field":
        record["vulns"] = []
    elif mutation == "duplicate":
        surface.raw["dependencies"].append(record.copy())
    else:
        (surface.root / "invarlock-0.15.0.dist-info/METADATA").write_text("changed")
    assert audit.main(surface.args) == 1
    assert json.loads(surface.report.read_text())["status"] == "blocked"


def test_unpublished_project_without_findings_is_not_reported_clean(
    surface, monkeypatch
):
    _unpublished_project(surface)
    surface.raw["dependencies"][0]["vulns"] = []
    monkeypatch.setattr(
        binding.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(surface.raw).encode(), stderr=b""
        ),
    )
    assert audit.main(surface.args) == 0
    report = json.loads(surface.report.read_text())
    assert report["status"] == "dependencies_clean"
    assert len(report["unpublished_project"]) == 1
