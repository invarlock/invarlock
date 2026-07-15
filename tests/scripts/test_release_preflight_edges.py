from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.release import release_preflight as preflight


def _config(tmp_path: Path, **changes: object) -> preflight.ReleasePreflightConfig:
    root = tmp_path / "checkout"
    root.mkdir(exist_ok=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='invarlock'\nversion='1.2.3'\n", encoding="utf-8"
    )
    values: dict[str, object] = {
        "repo_root": root,
        "release_sha": "a" * 40,
        "expected_version": "1.2.3",
        "dist_dir": root / "dist",
        "hash_manifest": root / "hashes.txt",
    }
    values.update(changes)
    return preflight.ReleasePreflightConfig(**values)


def _imported(tmp_path: Path, **changes: object) -> preflight.InstalledWheelImport:
    root = tmp_path / "site-packages"
    values: dict[str, object] = {
        "module_file": root / "invarlock" / "__init__.py",
        "module_version": "1.2.3",
        "distribution_name": "invarlock",
        "distribution_version": "1.2.3",
        "distribution_root": root,
    }
    values.update(changes)
    return preflight.InstalledWheelImport(**values)


def test_git_failure_and_checkout_identity_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "bad"),
    )
    with pytest.raises(preflight.ReleasePreflightError, match="unable to inspect"):
        preflight._git_output(config.repo_root, "status")

    with pytest.raises(preflight.ReleasePreflightError, match="lowercase 40-character"):
        preflight.validate_clean_exact_checkout(
            _config(tmp_path, release_sha="not-a-sha")
        )

    monkeypatch.setattr(
        preflight,
        "_git_output",
        lambda _root, *args: "a" * 40 if args[0] == "rev-parse" else "dirty",
    )
    with pytest.raises(preflight.ReleasePreflightError, match="not clean"):
        preflight.validate_clean_exact_checkout(config)


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("not = toml =", "unreadable"),
        ("[tool.demo]\nvalue=1\n", "does not match"),
        ("[project]\nversion='9.9.9'\n", "does not match"),
    ],
)
def test_checkout_version_rejects_unreadable_or_wrong_metadata(
    tmp_path: Path, contents: str, expected: str
) -> None:
    config = _config(tmp_path)
    (config.repo_root / "pyproject.toml").write_text(contents, encoding="utf-8")
    with pytest.raises(preflight.ReleasePreflightError, match=expected):
        preflight._validate_checkout_version(config)


def test_execution_environment_removes_caller_runtime_bypasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHOME", "/untrusted")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "1")
    environment = preflight._sanitized_execution_environment(
        tmp_path, allow_checkout_source=False
    )
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert "INVARLOCK_ALLOW_NETWORK" not in environment
    source_environment = preflight._sanitized_execution_environment(
        tmp_path, allow_checkout_source=True
    )
    assert source_environment["PYTHONPATH"] == str(tmp_path / "src")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not-json", "did not return JSON"),
        ("[]", "invalid payload"),
        ('{"module_file": "/tmp/module.py"}', "omitted required"),
    ],
)
def test_import_probe_requires_complete_json_identity(
    payload: str, expected: str
) -> None:
    with pytest.raises(preflight.ReleasePreflightError, match=expected):
        preflight._parse_import_probe(payload)


def test_import_probe_parses_complete_identity() -> None:
    imported = preflight._parse_import_probe(
        '{"module_file":"/tmp/site/invarlock/__init__.py",'
        '"module_version":"1.2.3","distribution_name":"invarlock",'
        '"distribution_version":"1.2.3","distribution_root":"/tmp/site"}'
    )
    assert imported.module_file.name == "__init__.py"
    assert imported.distribution_root == Path("/tmp/site").resolve()


@pytest.mark.parametrize(
    "error", [OSError("no process"), subprocess.TimeoutExpired([], 1)]
)
def test_isolated_command_launch_errors_are_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    with pytest.raises(preflight.ReleasePreflightError, match="isolated.*failed"):
        preflight._run_isolated_wheel_command([], cwd=tmp_path, timeout=1)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"distribution_name": "other"}, "name is not"),
        ({"module_version": "0"}, "version does not match"),
        ({"distribution_version": "0"}, "version does not match"),
        ({"distribution_root": Path("/tmp/elsewhere")}, "outside its distribution"),
    ],
)
def test_installed_wheel_identity_rejects_forked_metadata(
    tmp_path: Path, changes: dict[str, object], expected: str
) -> None:
    with pytest.raises(preflight.ReleasePreflightError, match=expected):
        preflight.validate_installed_wheel_import(
            _config(tmp_path), _imported(tmp_path, **changes)
        )


def test_installed_wheel_identity_accepts_external_distribution(tmp_path: Path) -> None:
    assert (
        preflight.validate_installed_wheel_import(
            _config(tmp_path), _imported(tmp_path)
        )
        is None
    )


@pytest.mark.parametrize(
    ("failed_step", "expected"),
    [
        (0, "create isolated"),
        (1, "install the candidate"),
        (2, "import probe failed"),
    ],
)
def test_installed_wheel_probe_rejects_each_failed_isolation_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_step: int,
    expected: str,
) -> None:
    calls = 0

    def command(*_args, **_kwargs):
        nonlocal calls
        current = calls
        calls += 1
        return subprocess.CompletedProcess(
            [],
            1 if current == failed_step else 0,
            "",
            "",
        )

    monkeypatch.setattr(preflight, "_run_isolated_wheel_command", command)
    monkeypatch.setattr(preflight, "_require_executable_file", lambda *_args: None)
    with pytest.raises(preflight.ReleasePreflightError, match=expected):
        preflight._probe_installed_wheel(_config(tmp_path), tmp_path / "candidate.whl")


def test_negative_evidence_subprocess_gate_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 2, "", "failure"),
    )
    with pytest.raises(preflight.ReleasePreflightError, match="negative-evidence"):
        preflight._run_current_negative_evidence_audit(config)


def test_config_from_args_resolves_checkout_relative_paths(tmp_path: Path) -> None:
    args = argparse.Namespace(
        repo_root=tmp_path,
        release_sha="a" * 40,
        expected_version="1.2.3",
        dist_dir=Path("dist"),
        hash_manifest=Path("hashes.txt"),
    )
    config = preflight._config_from_args(args)
    assert config.dist_dir == tmp_path / "dist"
    assert config.hash_manifest == tmp_path / "hashes.txt"


def test_argument_parser_accepts_complete_release_identity(tmp_path: Path) -> None:
    args = preflight._parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "--release-sha",
            "a" * 40,
            "--expected-version",
            "1.2.3",
            "--hash-manifest",
            "hashes.txt",
            "--json",
        ]
    )
    assert args.repo_root == tmp_path
    assert args.json is True


def test_main_reports_rejection_and_both_success_output_modes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = SimpleNamespace(json=False)
    monkeypatch.setattr(preflight, "_parse_args", lambda _argv: args)
    monkeypatch.setattr(preflight, "_config_from_args", lambda _args: object())
    monkeypatch.setattr(
        preflight,
        "run_release_preflight",
        lambda _config: (_ for _ in ()).throw(preflight.ReleasePreflightError("bad")),
    )
    assert preflight.main([]) == 1
    assert "release preflight rejected: bad" in capsys.readouterr().err

    monkeypatch.setattr(
        preflight, "run_release_preflight", lambda _config: {"ok": True}
    )
    assert preflight.main([]) == 0
    assert "Release preflight passed" in capsys.readouterr().out
    args.json = True
    assert preflight.main([]) == 0
    assert '"ok": true' in capsys.readouterr().out
