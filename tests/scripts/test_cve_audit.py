from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType

import pytest


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "security" / "cve_audit.py"
    spec = importlib.util.spec_from_file_location(
        "tests_cve_audit",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_requirement_lock_parser_collects_pinned_packages(tmp_path: Path) -> None:
    module = _load_script_module()
    req = tmp_path / "requirements.txt"
    req.write_text(
        """\
requests==2.33.0 \\
    --hash=sha256:abc
    # via invarlock
not-a-pin>=1.0
urllib3==2.6.3; python_version >= "3.12"
""",
        encoding="utf-8",
    )

    components = module.parse_requirement_lock(req, tmp_path)

    assert [(c.name, c.version) for c in components] == [
        ("requests", "2.33.0"),
        ("urllib3", "2.6.3"),
    ]


def test_uv_lock_parser_collects_pypi_packages(tmp_path: Path) -> None:
    module = _load_script_module()
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """\
version = 1

[[package]]
name = "urllib3"
version = "2.6.3"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "local-package"
version = "1.0.0"
source = { editable = "." }
""",
        encoding="utf-8",
    )

    components = module.parse_uv_lock(lock, tmp_path)

    assert [(c.name, c.version) for c in components] == [("urllib3", "2.6.3")]


def test_findings_include_all_matched_advisories_and_allowlist_classified() -> None:
    module = _load_script_module()
    component = module.Component(
        ecosystem="PyPI",
        name="urllib3",
        version="2.6.3",
        sources={"uv.lock"},
        used_by_src=True,
    )
    vuln = {
        "id": "GHSA-test",
        "aliases": ["CVE-2020-0001"],
        "published": "2020-05-01T00:00:00Z",
        "modified": "2020-05-02T00:00:00Z",
        "summary": "test advisory",
        "database_specific": {"severity": "HIGH"},
        "affected": [
            {
                "package": {"ecosystem": "PyPI", "name": "urllib3"},
                "ranges": [{"events": [{"introduced": "0"}, {"fixed": "2.7.0"}]}],
            }
        ],
    }

    findings = module.build_findings(
        [component],
        {component.key: [vuln]},
        allowlist={
            "CVE-2020-0001": {
                "allowed_sources": ["uv.lock"],
                "expires": "2026-06-01",
                "owner": "security-maintainers",
                "packages": ["urllib3"],
                "tracking_issue": "https://github.com/example/repo/issues/1",
                "reason": "fixture",
                "versions": ["2.6.3"],
            }
        },
        today=date(2026, 5, 15),
    )

    assert len(findings) == 1
    assert findings[0]["status"] == "accepted_until_2026-06-01"
    assert findings[0]["fixed_versions"] == ["2.7.0"]
    assert module.blocking_findings(findings) == []


def test_blocking_findings_excludes_only_current_acceptances() -> None:
    module = _load_script_module()
    accepted = {"status": "accepted_until_2026-07-13", "component": "torch"}
    expired = {"status": "unpatched_allowlist_expired", "component": "torch"}
    unpatched = {"status": "unpatched", "component": "urllib3"}

    assert module.blocking_findings([accepted, expired, unpatched]) == [
        expired,
        unpatched,
    ]


@pytest.mark.parametrize(
    ("name", "version", "sources"),
    [
        ("other", "2.6.3", {"uv.lock"}),
        ("urllib3", "2.6.2", {"uv.lock"}),
        ("urllib3", "2.6.3", {"requirements/other.txt"}),
        ("urllib3", "2.6.3", {"uv.lock", "requirements/other.txt"}),
    ],
)
def test_allowlist_never_expands_beyond_package_version_and_sources(
    name: str, version: str, sources: set[str]
) -> None:
    module = _load_script_module()
    component = module.Component(
        ecosystem="PyPI", name=name, version=version, sources=sources
    )
    allowlist = {
        "CVE-2020-0001": {
            "allowed_sources": ["uv.lock"],
            "expires": "2026-06-01",
            "owner": "security-maintainers",
            "packages": ["urllib3"],
            "tracking_issue": "https://github.com/example/repo/issues/1",
            "reason": "fixture",
            "versions": ["2.6.3"],
        }
    }

    status, entry = module.classify_status(
        ["CVE-2020-0001"],
        allowlist,
        date(2026, 5, 15),
        component=component,
    )

    assert status == "unpatched"
    assert entry is None


def test_load_allowlist_uses_strict_pip_audit_policy(tmp_path: Path) -> None:
    module = _load_script_module()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
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
                        "tracking_issue": "https://github.com/example/repo/pull/1",
                        "reason": "fixture",
                        "versions": ["1.0.0"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="must link to a GitHub tracking issue"):
        module.load_allowlist(allowlist)


def test_build_report_can_run_inventory_only(tmp_path: Path) -> None:
    module = _load_script_module()
    (tmp_path / "requirements" / "workflows").mkdir(parents=True)
    (tmp_path / "requirements" / "workflows" / "security.txt").write_text(
        "urllib3==2.6.3 \\\n    --hash=sha256:abc\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["urllib3>=2"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("import urllib3\n", encoding="utf-8")

    args = module.parse_args(["--repo-root", str(tmp_path), "--no-network"])
    report = module.build_report(args)

    assert report["inventory"]["component_count"] == 1
    assert report["inventory"]["src_used_component_count"] == 1
    assert "since" not in report
    assert report["findings"] == []


def test_parse_args_rejects_non_positive_batch_size() -> None:
    module = _load_script_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--batch-size", "0"])


def test_query_osv_batch_rejects_malformed_result_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    component = module.Component(
        ecosystem="PyPI",
        name="urllib3",
        version="2.6.3",
        sources={"uv.lock"},
    )

    monkeypatch.setattr(
        module,
        "_read_json_url",
        lambda *_args, **_kwargs: {"results": []},
    )

    with pytest.raises(RuntimeError, match="returned 0 results for 1 components"):
        module.query_osv_batch([component], 100, enrich=False)
