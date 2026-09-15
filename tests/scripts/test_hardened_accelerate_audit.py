from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.security import cve_audit, installed_audit_binding, run_pip_audit
from scripts.security import hardened_accelerate_audit as hardened


def test_verifier_retains_authenticated_bytes_and_rejects_changes(
    tmp_path, monkeypatch
):
    from scripts.security import build_hardened_accelerate_wheel as builder

    wheel = tmp_path / "derived.whl"
    wheel.write_bytes(b"trusted fixture")
    called = []

    def verify(path):
        called.append(path)
        assert path.read_bytes() == b"trusted fixture"
        return {"wheel_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    monkeypatch.setattr(builder, "verify_hardened_wheel", verify)
    proof = hardened.verify_wheel(wheel)
    assert called == [wheel]
    assert proof["derivation"]["wheel_sha256"] == proof["wheel_sha256"]

    def mutate(path):
        path.write_bytes(b"changed during verification")
        return {}

    monkeypatch.setattr(builder, "verify_hardened_wheel", mutate)
    with pytest.raises(ValueError, match="changed during verification"):
        hardened.verify_wheel(wheel)


@pytest.fixture
def surface(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wheel = tmp_path / hardened.WHEEL_DIRECTORY / hardened.WHEEL_NAME
    wheel.parent.mkdir(parents=True)
    trusted = b"independently authenticated derived wheel fixture"
    wheel.write_bytes(trusted)
    digest = hashlib.sha256(trusted).hexdigest()
    proof = {
        "package": "accelerate",
        "version": hardened.HARDENED_VERSION,
        "upstream_version": hardened.UPSTREAM_VERSION,
        "wheel_sha256": digest,
        "derivation": {"verified": True},
        "remediated_advisories": sorted(hardened.REMEDIATED_ADVISORIES),
    }

    def verify(path):
        if path.read_bytes() != trusted:
            raise ValueError("derived wheel mismatch")
        return copy.deepcopy(proof)

    monkeypatch.setattr(hardened, "verify_wheel", verify)
    lock = tmp_path / "requirements/workflows/hf.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text(f"accelerate=={hardened.HARDENED_VERSION} --hash=sha256:{digest}\n")
    package = {
        "name": "accelerate",
        "version": hardened.HARDENED_VERSION,
        "source": {"registry": hardened.WHEEL_DIRECTORY},
        "wheels": [{"path": hardened.WHEEL_NAME}],
    }
    return SimpleNamespace(
        root=tmp_path, lock=lock, wheel=wheel, proof=proof, package=package
    )


def test_cve_scan_retains_upstream_identity_and_never_accepts_alias_injection(
    surface, monkeypatch
):
    component = cve_audit.parse_requirement_lock(surface.lock, surface.root)[0]
    observed = []
    raw = [
        {"id": "GHSA-4j2p-28q2-5m79"},
        {"id": "GHSA-new", "aliases": ["GHSA-4j2p-28q2-5m79"]},
    ]

    def query(_url, **kwargs):
        observed.append(kwargs["payload"])
        return {"results": [{"vulns": raw}]}

    monkeypatch.setattr(cve_audit, "_read_json_url", query)
    results = cve_audit.query_osv_batch([component], 10, enrich=False)
    assert observed[0]["queries"][0] == {
        "package": {"ecosystem": "PyPI", "name": "accelerate"},
        "version": "1.14.0",
    }
    findings = cve_audit.build_findings(
        [component], results, allowlist={}, today=date.today()
    )
    assert {f["advisory"]: f["status"] for f in findings} == {
        "GHSA-4j2p-28q2-5m79": "remediated",
        "GHSA-new": "unpatched",
    }
    assert len(cve_audit.blocking_findings(findings)) == 1
    assert all(
        f["version"] == hardened.HARDENED_VERSION and f["scan_version"] == "1.14.0"
        for f in findings
    )
    assert [f["raw_finding"] for f in findings] == raw


@pytest.mark.parametrize(
    "mutation", ["hash", "additional-hash", "duplicate", "include", "version", "wheel"]
)
def test_requirement_remediation_authenticates_entire_lock_surface(surface, mutation):
    if mutation == "hash":
        surface.lock.write_text(
            surface.lock.read_text().replace(surface.proof["wheel_sha256"], "0" * 64)
        )
    elif mutation == "additional-hash":
        surface.lock.write_text(
            surface.lock.read_text().strip() + " --hash=sha256:" + "0" * 64 + "\n"
        )
    elif mutation == "duplicate":
        surface.lock.write_text(surface.lock.read_text() * 2)
    elif mutation == "include":
        surface.lock.write_text(surface.lock.read_text() + "-r attacker.txt\n")
    elif mutation == "version":
        surface.lock.write_text(
            surface.lock.read_text().replace(
                hardened.HARDENED_VERSION, "1.14.0+unknown.1"
            )
        )
    elif mutation == "wheel":
        surface.wheel.write_bytes(b"changed")
    with pytest.raises(ValueError):
        cve_audit.parse_requirement_lock(surface.lock, surface.root)


@pytest.mark.parametrize(
    "mutation", [None, "path", "registry", "version", "sdist", "extra-wheel"]
)
def test_uv_source_requires_exact_authenticated_derived_wheel(surface, mutation):
    package = surface.package
    if mutation == "path":
        package["wheels"][0]["path"] = "../other.whl"
    elif mutation == "registry":
        package["source"]["registry"] = "https://pypi.org/simple"
    elif mutation == "version":
        package["version"] = "1.14.0+unknown"
    elif mutation == "sdist":
        package["sdist"] = {"path": "source.tar.gz"}
    elif mutation == "extra-wheel":
        package["wheels"].append({"path": "other.whl"})
    if mutation:
        with pytest.raises(ValueError):
            hardened.uv_remediation(package, surface.root)
    else:
        assert hardened.uv_remediation(package, surface.root) == {
            **surface.proof,
            "source_only": False,
        }


def test_source_wheel_is_build_input_only_and_new_advisories_still_block(surface):
    from scripts.security.build_hardened_accelerate_wheel import UPSTREAM_WHEEL_SHA256

    source = surface.root / hardened.SOURCE_LOCK
    source.write_text(f"accelerate==1.14.0 --hash=sha256:{UPSTREAM_WHEEL_SHA256}\n")
    component = cve_audit.parse_requirement_lock(source, surface.root)[0]
    findings = cve_audit.build_findings(
        [component],
        {component.key: [{"id": "PYSEC-2026-3804"}, {"id": "GHSA-new"}]},
        allowlist={},
        today=date.today(),
    )
    assert {f["status"] for f in findings} == {"remediated_build_input", "unpatched"}
    runtime = source.parent / "unpatched-runtime.txt"
    runtime.write_bytes(source.read_bytes())
    runtime_component = cve_audit.parse_requirement_lock(runtime, surface.root)[0]
    assert runtime_component.remediation is None
    merged = cve_audit.merge_components([component, runtime_component])
    assert merged[0].remediation is None
    surface.wheel.unlink()
    with pytest.raises(FileNotFoundError):
        cve_audit.parse_requirement_lock(source, surface.root)


def test_local_uv_component_is_included_in_upstream_audit(surface):
    lock = surface.root / "uv.lock"
    lock.write_text(
        'version = 1\n[[package]]\nname = "accelerate"\n'
        f'version = "{hardened.HARDENED_VERSION}"\n'
        f'source = {{ registry = "{hardened.WHEEL_DIRECTORY}" }}\n'
        f'wheels = [{{ path = "{hardened.WHEEL_NAME}" }}]\n'
    )
    components = cve_audit.parse_uv_lock(lock, surface.root)
    assert len(components) == 1
    assert components[0].scan_version == "1.14.0"
    assert components[0].remediation["wheel_sha256"] == surface.proof["wheel_sha256"]


@pytest.mark.parametrize(
    "requirement",
    ["accelerate>=1.14.0", "accelerate @ https://example.invalid/accelerate.whl"],
)
def test_accelerate_requirements_cannot_evade_exact_pin_validation(
    surface, requirement
):
    surface.lock.write_text(requirement + "\n")
    with pytest.raises(ValueError, match="exact authenticated pin"):
        cve_audit.parse_requirement_lock(surface.lock, surface.root)


@pytest.mark.parametrize("mutation", ["missing-version", "duplicate", "editable"])
def test_uv_inventory_rejects_ambiguous_or_unscannable_accelerate(surface, mutation):
    package = (
        '[[package]]\nname = "accelerate"\nversion = "1.14.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    if mutation == "missing-version":
        package = package.replace('version = "1.14.0"\n', "")
    elif mutation == "duplicate":
        package *= 2
    else:
        package = package.replace(
            '{ registry = "https://pypi.org/simple" }', '{ editable = "." }'
        )
    lock = surface.root / "uv.lock"
    lock.write_text("version = 1\n" + package)
    message = (
        "unsupported Accelerate lock source"
        if mutation == "editable"
        else "missing or duplicate Accelerate uv identity"
    )
    with pytest.raises(ValueError, match=message):
        cve_audit.parse_uv_lock(lock, surface.root)


@pytest.mark.parametrize("mutation", [None, "lock", "inventory"])
def test_cve_report_rechecks_authenticated_inputs_after_query(
    surface, monkeypatch, mutation
):
    def query(components, batch_size, *, enrich):
        assert len(components) == 1
        assert components[0].scan_version == "1.14.0"
        if mutation == "lock":
            surface.lock.write_text(surface.lock.read_text() + "# changed\n")
        elif mutation == "inventory":
            (surface.lock.parent / "added.txt").write_text("another==2.0\n")
        return {components[0].key: [{"id": "PYSEC-2026-3804"}]}

    monkeypatch.setattr(cve_audit, "query_osv_batch", query)
    args = cve_audit.parse_args(["--repo-root", str(surface.root)])
    if mutation:
        message = (
            "lock changed during CVE audit"
            if mutation == "lock"
            else "remediation binding changed during CVE audit"
        )
        with pytest.raises(ValueError, match=message):
            cve_audit.build_report(args)
    else:
        report = cve_audit.build_report(args)
        assert len(report["findings"]) == 1
        assert report["findings"][0]["status"] == "remediated"
        assert report["raw_osv_results"][0]["version"] == "1.14.0"
        assert report["sources"]["inventory_sha256"] == {
            "requirements/workflows/hf.txt": hashlib.sha256(
                surface.lock.read_bytes()
            ).hexdigest()
        }


def test_enrichment_cannot_replace_unknown_id_with_remediated_id(surface, monkeypatch):
    component = cve_audit.parse_requirement_lock(surface.lock, surface.root)[0]
    monkeypatch.setattr(
        cve_audit, "fetch_osv_vuln", lambda _id: {"id": "PYSEC-2026-3804"}
    )
    with pytest.raises(RuntimeError, match="changed the matched advisory"):
        cve_audit.enrich_osv_results({component.key: [{"id": "GHSA-new"}]})


@pytest.mark.parametrize(
    "mutation", [None, "unknown", "derived-scan", "tampered-wheel", "tampered-lock"]
)
def test_requirements_scanner_retains_raw_and_checks_binding_after_scan(
    surface, monkeypatch, mutation
):
    raw = {
        "dependencies": [
            {
                "name": "accelerate",
                "version": "1.14.0",
                "vulns": [{"id": "PYSEC-2026-3804", "fix_versions": [], "aliases": []}],
            }
        ]
    }
    if mutation == "unknown":
        raw["dependencies"][0]["vulns"][0]["id"] = "GHSA-new"
    elif mutation == "derived-scan":
        raw["dependencies"][0]["version"] = hardened.HARDENED_VERSION

    def scan(command, **_kwargs):
        assert (
            Path(command[command.index("--requirement") + 1]).read_text()
            == "accelerate==1.14.0\n"
        )
        if mutation == "tampered-wheel":
            surface.wheel.write_bytes(b"changed")
        elif mutation == "tampered-lock":
            surface.lock.write_text(surface.lock.read_text() + "# changed\n")
        return SimpleNamespace(
            returncode=1, stdout=json.dumps(raw).encode(), stderr=b"raw diagnostic"
        )

    monkeypatch.setattr(installed_audit_binding.subprocess, "run", scan)
    report = surface.root / "report.json"
    assert run_pip_audit.main(
        ["--requirement", str(surface.lock), "--report", str(report)]
    ) == (0 if mutation is None else 1)
    result = json.loads(report.read_text())
    assert result["status"] == ("remediated" if mutation is None else "blocked")
    assert result["raw_findings"] == raw


def test_requirement_scan_keeps_other_lock_dependencies_and_their_findings(
    surface, monkeypatch
):
    other = surface.lock.parent / "other.txt"
    other.write_text("another==2.0 --hash=sha256:" + "0" * 64 + "\n")
    raw = {
        "dependencies": [
            {"name": "accelerate", "version": "1.14.0", "vulns": []},
            {
                "name": "another",
                "version": "2.0",
                "vulns": [{"id": "PYSEC-2026-3804", "fix_versions": []}],
            },
        ]
    }
    monkeypatch.setattr(
        installed_audit_binding.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=1, stdout=json.dumps(raw).encode(), stderr=b""
        ),
    )
    report = surface.root / "report.json"
    assert (
        run_pip_audit.main(
            [
                "--requirement",
                str(surface.lock),
                "--requirement",
                str(other),
                "--report",
                str(report),
            ]
        )
        == 1
    )
    result = json.loads(report.read_text())
    assert result["remediated_findings"] == []
    assert result["blocking_findings"][0]["name"] == "another"
    assert result["scanner_requirements"] == "accelerate==1.14.0\nanother==2.0\n"
