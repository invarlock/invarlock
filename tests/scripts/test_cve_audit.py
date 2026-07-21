from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("   ", None),
        ("2026-05-01T12:30:00Z", date(2026, 5, 1)),
        ("2026-05-01 trailing", date(2026, 5, 1)),
        ("not-a-date", None),
    ],
)
def test_parse_date_handles_osv_formats_without_guessing(
    value: str | None, expected: date | None
) -> None:
    module = _load_script_module()
    assert module.parse_date(value) == expected


def test_inventory_parsers_ignore_untrusted_or_irrelevant_shapes(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    assert module.parse_uv_lock(tmp_path / "missing.lock", tmp_path) == []
    assert module.parse_pyproject_dependency_names(tmp_path / "missing.toml") == set()
    assert module.collect_src_import_package_names(tmp_path / "missing-src") == set()

    lock = tmp_path / "uv.lock"
    lock.write_text(
        """\
version = 1
package = [
  "malformed",
  { version = "1" },
  { name = "local", version = "1", source = { editable = "." } },
  { name = "direct", version = "2", source = "local" },
  { name = "pypi", version = "3", source = { registry = "https://pypi.org/simple" } },
]
""",
        encoding="utf-8",
    )
    assert [
        (item.name, item.version) for item in module.parse_uv_lock(lock, tmp_path)
    ] == [
        ("direct", "2"),
        ("pypi", "3"),
    ]

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """\
[project]
dependencies = ["Pillow>=10", "???"]
optional-dependencies.bad = "not-a-list"
optional-dependencies.good = ["opencv_python>=4", 7]
""",
        encoding="utf-8",
    )
    assert module.parse_pyproject_dependency_names(pyproject) == {
        "opencv-python",
        "pillow",
    }
    assert module._dependency_name(None) is None
    assert module._dependency_name("???") is None

    src = tmp_path / "src"
    src.mkdir()
    (src / "broken.py").write_text("def invalid(:\n", encoding="utf-8")
    (src / "imports.py").write_text(
        "import PIL.Image\nfrom yaml import safe_load\nfrom . import local\n",
        encoding="utf-8",
    )
    assert module.collect_src_import_package_names(src) == {"pillow", "pyyaml"}


def test_merge_inventory_unifies_sources_and_source_usage() -> None:
    module = _load_script_module()
    merged = module.merge_components(
        [
            module.Component("PyPI", "Example_Package", "1", {"uv.lock"}),
            module.Component(
                "PyPI",
                "example-package",
                "1",
                {"requirements/test.txt"},
                used_by_src=True,
            ),
        ]
    )

    assert len(merged) == 1
    assert merged[0].name == "example-package"
    assert merged[0].sources == {"uv.lock", "requirements/test.txt"}
    assert merged[0].used_by_src is True


def test_allowlist_loading_is_optional_and_preserves_strict_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    assert module.load_allowlist(tmp_path / "missing.json") == {}
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text("{}", encoding="utf-8")
    entry = SimpleNamespace(
        advisory="GHSA-test-test-test",
        expires=date(2026, 6, 1),
        packages=("example-package",),
        versions=("1.0.0",),
        allowed_sources=("uv.lock",),
        owner="security-maintainers",
        tracking_issue="https://github.com/example/repo/issues/1",
        reason="fixture",
    )
    monkeypatch.setattr(
        module, "load_pip_audit_allowlist", lambda _path: ("owner", [entry])
    )

    assert module.load_allowlist(allowlist) == {
        "GHSA-test-test-test": {
            "expires": "2026-06-01",
            "packages": ["example-package"],
            "versions": ["1.0.0"],
            "allowed_sources": ["uv.lock"],
            "owner": "security-maintainers",
            "tracking_issue": "https://github.com/example/repo/issues/1",
            "reason": "fixture",
        }
    }


def test_osv_request_binds_method_headers_timeout_and_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    observed: dict[str, object] = {}

    def urlopen(request: object, *, timeout: int) -> io.BytesIO:
        observed["request"] = request
        observed["timeout"] = timeout
        return io.BytesIO(b'{"results":[]}')

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)

    assert module._read_json_url(
        "https://example.invalid/query", method="POST", payload={"query": 1}
    ) == {"results": []}
    request = observed["request"]
    assert request.method == "POST"
    assert request.data == b'{"query": 1}'
    assert request.get_header("User-agent") == "invarlock-cve-audit"
    assert observed["timeout"] == module.REQUEST_TIMEOUT_SECONDS


def _component(module: ModuleType):
    return module.Component(
        ecosystem="PyPI",
        name="urllib3",
        version="2.6.3",
        sources={"uv.lock"},
    )


def test_query_osv_batch_fails_closed_on_transport_or_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    component = _component(module)
    with pytest.raises(ValueError, match="must be positive"):
        module.query_osv_batch([component], 0, enrich=False)

    monkeypatch.setattr(
        module,
        "_read_json_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("offline")
        ),
    )
    with pytest.raises(RuntimeError, match="failed at offset 0"):
        module.query_osv_batch([component], 1, enrich=False)

    monkeypatch.setattr(module, "_read_json_url", lambda *_args, **_kwargs: {})
    with pytest.raises(RuntimeError, match="missing results"):
        module.query_osv_batch([component], 1, enrich=False)


def test_query_osv_batch_filters_malformed_vulnerabilities_and_can_enrich(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    component = _component(module)
    monkeypatch.setattr(
        module,
        "_read_json_url",
        lambda *_args, **_kwargs: {
            "results": [{"vulns": [{"id": "GHSA-valid"}, "malformed"]}]
        },
    )
    observed: list[dict[tuple[str, str, str], list[dict[str, object]]]] = []

    def enrich(results: dict[tuple[str, str, str], list[dict[str, object]]]):
        observed.append(results)
        return results

    monkeypatch.setattr(module, "enrich_osv_results", enrich)

    result = module.query_osv_batch([component], 1, enrich=True)
    assert result == {component.key: [{"id": "GHSA-valid"}]}
    assert observed == [result]

    monkeypatch.setattr(
        module, "_read_json_url", lambda *_args, **_kwargs: {"results": [None]}
    )
    assert module.query_osv_batch([component], 1, enrich=False) == {component.key: []}


def test_osv_enrichment_caches_results_and_falls_back_on_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    keys = [("PyPI", "one", "1"), ("PyPI", "two", "2")]
    calls: list[str] = []

    def fetch(advisory: str) -> dict[str, object]:
        calls.append(advisory)
        return {"id": advisory, "summary": "enriched"}

    monkeypatch.setattr(module, "fetch_osv_vuln", fetch)
    result = module.enrich_osv_results(
        {
            keys[0]: [{"id": "GHSA-shared"}, {"summary": "missing id"}],
            keys[1]: [{"id": "GHSA-shared"}],
        }
    )
    assert calls == ["GHSA-shared"]
    assert result[keys[0]][1] == {"summary": "missing id"}

    original = {"id": "GHSA-offline", "summary": "batch result"}
    monkeypatch.setattr(
        module,
        "fetch_osv_vuln",
        lambda _advisory: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    assert module.enrich_osv_results({keys[0]: [original]})[keys[0]] == [original]


def test_osv_helpers_filter_cross_package_data_and_normalize_severity() -> None:
    module = _load_script_module()
    component = _component(module)
    vuln = {
        "affected": [
            "malformed",
            {"package": {"name": "other", "ecosystem": "PyPI"}},
            {"package": {"name": "urllib3", "ecosystem": "npm"}},
            {
                "package": {"name": "urllib3", "ecosystem": "PyPI"},
                "ranges": [
                    "malformed",
                    {
                        "events": [
                            "malformed",
                            {"introduced": "0"},
                            {"fixed": "2.7.0"},
                        ]
                    },
                ],
                "versions": ["2.6.3", ">=2.8.0", None],
            },
        ]
    }
    assert module.fixed_versions_for(vuln, component) == ["2.7.0", "2.8.0"]
    assert module.severity_for({"database_specific": {"severity": "HIGH"}}) == "high"
    assert module.severity_for({"severity": [{"type": "CVSS_V3"}]}) == "CVSS_V3"
    assert module.severity_for({"severity": "malformed"}) == "unknown"
    assert module.advisory_ids({"id": "GHSA-a", "aliases": "malformed"}) == ["GHSA-a"]
    assert module.advisory_ids({"id": "GHSA-a", "aliases": ["CVE-a", "", None]}) == [
        "CVE-a",
        "GHSA-a",
    ]


@pytest.mark.parametrize(
    ("expires", "status"),
    [
        ("not-a-date", "unpatched_allowlist_invalid"),
        ("2026-05-01", "unpatched_allowlist_expired"),
    ],
)
def test_allowlist_classification_rejects_invalid_or_expired_acceptance(
    expires: str, status: str
) -> None:
    module = _load_script_module()
    component = _component(module)
    entry = {
        "packages": ["urllib3"],
        "versions": ["2.6.3"],
        "allowed_sources": ["uv.lock"],
        "expires": expires,
    }
    observed, selected = module.classify_status(
        ["GHSA-missing", "GHSA-test"],
        {"GHSA-test": entry},
        date(2026, 5, 15),
        component=component,
    )
    assert observed == status
    assert selected is entry


def test_markdown_report_renders_blockers_sources_and_empty_inventory(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    report = {
        "generated_at": "2026-05-15T00:00:00+00:00",
        "inventory": {"component_count": 1},
        "findings": [
            {
                "component": "urllib3",
                "version": "2.6.3",
                "sources": ["a", "b", "c", "d"],
                "advisory": "GHSA-test",
                "published": None,
                "severity": "high",
                "status": "unpatched",
                "fixed_versions": [],
            }
        ],
    }
    path = tmp_path / "reports" / "audit.md"
    module.write_markdown_report(report, path)
    rendered = path.read_text(encoding="utf-8")
    assert "a, b, c, +1 more" in rendered
    assert "not listed" in rendered
    assert "Blocking findings: `1`" in rendered

    report["findings"] = []
    module.write_markdown_report(report, path)
    assert "No OSV findings" in path.read_text(encoding="utf-8")


def test_build_report_uses_network_query_and_inventory_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    component = _component(module)
    monkeypatch.setattr(module, "collect_inventory", lambda _root: [component])
    monkeypatch.setattr(module, "load_allowlist", lambda _path: {})
    observed: list[tuple[int, bool]] = []

    def query(components: list[object], batch_size: int, *, enrich: bool):
        assert components == [component]
        observed.append((batch_size, enrich))
        return {component.key: []}

    monkeypatch.setattr(module, "query_osv_batch", query)
    args = module.parse_args(
        ["--repo-root", str(tmp_path), "--batch-size", "7", "--no-enrich"]
    )
    report = module.build_report(args)

    assert observed == [(7, False)]
    assert report["sources"]["inventory_files"] == ["uv.lock"]


@pytest.mark.parametrize(("blockers", "expected"), [(False, 0), (True, 1)])
def test_main_writes_both_reports_and_returns_blocking_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blockers: bool,
    expected: int,
) -> None:
    module = _load_script_module()
    findings = [{"status": "unpatched"}] if blockers else []
    report = {
        "generated_at": "2026-05-15T00:00:00+00:00",
        "inventory": {"component_count": 0},
        "findings": findings,
    }
    monkeypatch.setattr(module, "build_report", lambda _args: report)
    monkeypatch.setattr(
        module,
        "write_markdown_report",
        lambda value, path: path.write_text(str(value), encoding="utf-8"),
    )

    assert (
        module.main(
            [
                "--repo-root",
                str(tmp_path),
                "--out-json",
                "reports/audit.json",
                "--out-md",
                "reports/audit.md",
                "--no-network",
            ]
        )
        == expected
    )
    assert (tmp_path / "reports" / "audit.json").is_file()
    assert (tmp_path / "reports" / "audit.md").is_file()
