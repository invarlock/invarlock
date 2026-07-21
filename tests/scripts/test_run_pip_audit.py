from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "security" / "run_pip_audit.py"
    spec = importlib.util.spec_from_file_location(
        "tests_run_pip_audit",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_allowlist(
    path: Path,
    *,
    tracking_issue: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "owner": "security-maintainers",
                "entries": [
                    {
                        "advisory": "GHSA-test-test-test",
                        "allowed_sources": ["requirements/test.txt"],
                        "compensating_control": "isolated test surface",
                        "owner": "security-maintainers",
                        "expires": (date.today() + timedelta(days=7)).isoformat(),
                        "packages": ["example-package"],
                        "tracking_issue": tracking_issue,
                        "reason": "test fixture",
                        "versions": ["1.0.0"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_allowlist_rejects_non_issue_github_url(tmp_path: Path) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://github.com/pypa/pip/pull/13607",
    )

    with pytest.raises(SystemExit, match="must link to a GitHub tracking issue"):
        module._load_allowlist(allowlist)


def test_load_allowlist_rejects_non_github_tracking_url(tmp_path: Path) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://example.com/issues/13607",
    )

    with pytest.raises(SystemExit, match="must link to a GitHub tracking issue"):
        module._load_allowlist(allowlist)


def test_load_allowlist_accepts_empty_entries(tmp_path: Path) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps({"owner": "security-maintainers", "entries": []}),
        encoding="utf-8",
    )

    owner, entries = module._load_allowlist(allowlist)

    assert owner == "security-maintainers"
    assert entries == []


def test_allowlist_binds_package_version_and_source(tmp_path: Path) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://github.com/example/repo/issues/1",
    )

    _owner, entries = module._load_allowlist(allowlist)

    assert entries[0].packages == ("example-package",)
    assert entries[0].versions == ("1.0.0",)
    assert entries[0].allowed_sources == ("requirements/test.txt",)


@pytest.mark.parametrize(("version", "ignored"), [("1.0.0", True), ("2.0.0", False)])
def test_pip_audit_exception_applies_only_to_exact_requirement_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    ignored: bool,
) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://github.com/example/repo/issues/1",
    )
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    (requirements / "test.txt").write_text(
        f"example-package=={version}\n", encoding="utf-8"
    )
    observed: list[str] = []

    def run(command: list[str], *, check: bool):
        assert check is False
        observed.extend(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", run)

    assert (
        module.main(
            [
                "--allowlist",
                str(allowlist),
                "--requirement",
                "requirements/test.txt",
            ]
        )
        == 0
    )
    assert ("--ignore-vuln" in observed) is ignored


def test_pip_audit_exception_does_not_cross_requirement_or_path_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://github.com/example/repo/issues/1",
    )
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    (requirements / "test.txt").write_text("example-package==1.0.0\n", encoding="utf-8")
    (requirements / "other.txt").write_text(
        "example-package==1.0.0\n", encoding="utf-8"
    )
    commands: list[list[str]] = []

    def run(command: list[str], *, check: bool):
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", run)

    base = [
        "--allowlist",
        str(allowlist),
        "--requirement",
        "requirements/test.txt",
    ]
    assert module.main([*base, "--requirement", "requirements/other.txt"]) == 0
    assert "--ignore-vuln" not in commands[-1]

    install_path = tmp_path / "installed"
    install_path.mkdir()
    assert module.main([*base, "--path", str(install_path)]) == 0
    assert "--ignore-vuln" not in commands[-1]


@pytest.mark.parametrize("location", ["top", "entry"])
def test_load_allowlist_rejects_undocumented_fields(
    tmp_path: Path, location: str
) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://github.com/pypa/pip/issues/13607",
    )
    payload = json.loads(allowlist.read_text(encoding="utf-8"))
    if location == "top":
        payload["unexpected"] = True
    else:
        payload["entries"][0]["unexpected"] = True
    allowlist.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="unsupported fields"):
        module._load_allowlist(allowlist)


def test_load_allowlist_requires_a_compensating_control(tmp_path: Path) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://github.com/pypa/pip/issues/13607",
    )
    payload = json.loads(allowlist.read_text(encoding="utf-8"))
    payload["entries"][0]["compensating_control"] = ""
    allowlist.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="is incomplete"):
        module._load_allowlist(allowlist)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must contain an object"),
        ({"owner": "", "entries": []}, "owner missing"),
        ({"owner": "security-maintainers", "entries": {}}, "must be a list"),
        (
            {"owner": "security-maintainers", "entries": ["unexpected"]},
            "must be an object",
        ),
    ],
)
def test_load_allowlist_rejects_malformed_policy_shapes(
    tmp_path: Path, payload: object, message: str
) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match=message):
        module._load_allowlist(allowlist)


@pytest.mark.parametrize(
    ("days", "message"),
    [(-1, "expired"), (31, "exceeds 30 days")],
)
def test_load_allowlist_enforces_expiry_window(
    tmp_path: Path, days: int, message: str
) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://github.com/example/repo/issues/1",
    )
    payload = json.loads(allowlist.read_text(encoding="utf-8"))
    payload["entries"][0]["expires"] = (date.today() + timedelta(days=days)).isoformat()
    allowlist.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match=message):
        module._load_allowlist(allowlist)


def test_requirement_scan_handles_external_paths_and_ignores_unpinned_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        tracking_issue="https://github.com/example/repo/issues/1",
    )
    requirement = tmp_path / "external.txt"
    requirement.write_text(
        "# generated lock\nexample-package>=1.0.0\n", encoding="utf-8"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    observed: list[str] = []

    def run(command: list[str], *, check: bool):
        assert check is False
        observed.extend(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.chdir(workspace)
    monkeypatch.setattr(module.subprocess, "run", run)

    assert (
        module.main(
            [
                "--allowlist",
                str(allowlist),
                "--requirement",
                str(requirement),
            ]
        )
        == 7
    )
    assert observed == ["pip-audit", "-r", str(requirement)]
