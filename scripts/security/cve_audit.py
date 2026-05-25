#!/usr/bin/env python3
"""Audit locked dependencies against OSV advisories.

The audit is intentionally dependency-light so it can run from the repository
toolchain before the security environment is fully trusted.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns"
DEFAULT_OUTPUT_JSON = "reports/security/cve-audit.json"
DEFAULT_OUTPUT_MD = "reports/security/cve-audit.md"
REQUEST_TIMEOUT_SECONDS = 30

_REQ_PIN_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^\s\\;]+)"
)
_REQ_NAME_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)")
_IMPORT_ALIAS_TO_PACKAGE = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}


@dataclass
class Component:
    ecosystem: str
    name: str
    version: str
    sources: set[str] = field(default_factory=set)
    used_by_src: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.ecosystem, normalize_package_name(self.name), self.version)


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None


def discover_requirement_files(repo_root: Path) -> list[Path]:
    candidates = [
        *sorted((repo_root / "requirements" / "workflows").glob("*.txt")),
        *sorted((repo_root / "requirements" / "evidence-packs").glob("*.txt")),
    ]
    return [path for path in candidates if path.is_file()]


def parse_requirement_lock(path: Path, repo_root: Path) -> list[Component]:
    components: list[Component] = []
    rel = path.relative_to(repo_root).as_posix()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _REQ_PIN_RE.match(line)
        if not match:
            continue
        components.append(
            Component(
                ecosystem="PyPI",
                name=match.group("name"),
                version=match.group("version").strip(),
                sources={rel},
            )
        )
    return components


def parse_uv_lock(path: Path, repo_root: Path) -> list[Component]:
    if not path.exists():
        return []
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    components: list[Component] = []
    rel = path.relative_to(repo_root).as_posix()
    for package in payload.get("package", []):
        if not isinstance(package, dict):
            continue
        name = str(package.get("name", "")).strip()
        version = str(package.get("version", "")).strip()
        if not name or not version:
            continue
        source = package.get("source", {})
        if isinstance(source, dict):
            registry = str(source.get("registry", ""))
            if source and (not registry or "pypi.org" not in registry):
                continue
        components.append(
            Component(ecosystem="PyPI", name=name, version=version, sources={rel})
        )
    return components


def parse_pyproject_dependency_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    project = payload.get("project", {})
    if isinstance(project, dict):
        for dep in project.get("dependencies", []):
            name = _dependency_name(dep)
            if name:
                names.add(name)
        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for deps in optional.values():
                if isinstance(deps, list):
                    for dep in deps:
                        name = _dependency_name(dep)
                        if name:
                            names.add(name)
    return names


def _dependency_name(requirement: object) -> str | None:
    if not isinstance(requirement, str):
        return None
    match = _REQ_NAME_RE.match(requirement)
    if not match:
        return None
    return normalize_package_name(match.group("name"))


def collect_src_import_package_names(src_root: Path) -> set[str]:
    imports: set[str] = set()
    if not src_root.exists():
        return imports
    for path in sorted(src_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(_import_to_package(alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(_import_to_package(node.module))
    return {normalize_package_name(name) for name in imports if name}


def _import_to_package(module_name: str) -> str:
    top_level = module_name.split(".", 1)[0]
    return _IMPORT_ALIAS_TO_PACKAGE.get(top_level, top_level)


def merge_components(components: list[Component]) -> list[Component]:
    merged: dict[tuple[str, str, str], Component] = {}
    for component in components:
        key = component.key
        current = merged.get(key)
        if current is None:
            merged[key] = Component(
                ecosystem=component.ecosystem,
                name=normalize_package_name(component.name),
                version=component.version,
                sources=set(component.sources),
                used_by_src=component.used_by_src,
            )
        else:
            current.sources.update(component.sources)
            current.used_by_src = current.used_by_src or component.used_by_src
    return sorted(merged.values(), key=lambda c: (c.ecosystem, c.name, c.version))


def collect_inventory(repo_root: Path) -> list[Component]:
    components: list[Component] = []
    components.extend(parse_uv_lock(repo_root / "uv.lock", repo_root))
    for req_file in discover_requirement_files(repo_root):
        components.extend(parse_requirement_lock(req_file, repo_root))

    imported_packages = collect_src_import_package_names(repo_root / "src")
    pyproject_names = parse_pyproject_dependency_names(repo_root / "pyproject.toml")
    src_relevant_names = imported_packages | pyproject_names

    merged = merge_components(components)
    for component in merged:
        component.used_by_src = component.name in src_relevant_names
    return merged


def load_allowlist(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, str]] = {}
    for raw in payload.get("entries", []):
        if not isinstance(raw, dict):
            continue
        advisory = str(raw.get("advisory", "")).strip()
        if not advisory:
            continue
        entries[advisory] = {
            "expires": str(raw.get("expires", "")).strip(),
            "owner": str(raw.get("owner", "")).strip(),
            "tracking_issue": str(raw.get("tracking_issue", "")).strip(),
            "reason": str(raw.get("reason", "")).strip(),
        }
    return entries


def _read_json_url(
    url: str, *, method: str = "GET", payload: object | None = None
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "invarlock-cve-audit",
        },
        method=method,
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def query_osv_batch(
    components: list[Component], batch_size: int, *, enrich: bool
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    results: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for offset in range(0, len(components), batch_size):
        batch = components[offset : offset + batch_size]
        payload = {
            "queries": [
                {
                    "package": {
                        "ecosystem": component.ecosystem,
                        "name": component.name,
                    },
                    "version": component.version,
                }
                for component in batch
            ]
        }
        try:
            response_payload = _read_json_url(
                OSV_BATCH_URL, method="POST", payload=payload
            )
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OSV batch query failed at offset {offset}: {exc}"
            ) from exc
        for component, raw_result in zip(
            batch, response_payload.get("results", []), strict=False
        ):
            vulns = raw_result.get("vulns", []) if isinstance(raw_result, dict) else []
            results[component.key] = [v for v in vulns if isinstance(v, dict)]
    if enrich:
        return enrich_osv_results(results)
    return results


def fetch_osv_vuln(advisory_id: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(advisory_id, safe="")
    return _read_json_url(f"{OSV_VULN_URL}/{encoded}")


def enrich_osv_results(
    results: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    cache: dict[str, dict[str, Any]] = {}
    enriched: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for key, vulns in results.items():
        enriched_vulns: list[dict[str, Any]] = []
        for vuln in vulns:
            advisory_id = str(vuln.get("id", "")).strip()
            if not advisory_id:
                enriched_vulns.append(vuln)
                continue
            if advisory_id not in cache:
                try:
                    cache[advisory_id] = fetch_osv_vuln(advisory_id)
                except urllib.error.URLError:
                    cache[advisory_id] = vuln
            enriched_vulns.append(cache[advisory_id])
        enriched[key] = enriched_vulns
    return enriched


def fixed_versions_for(vuln: dict[str, Any], component: Component) -> list[str]:
    fixed: set[str] = set()
    for affected in vuln.get("affected", []):
        if not isinstance(affected, dict):
            continue
        package = affected.get("package", {})
        if isinstance(package, dict):
            package_name = normalize_package_name(str(package.get("name", "")))
            ecosystem = str(package.get("ecosystem", ""))
            if package_name and package_name != component.name:
                continue
            if ecosystem and ecosystem != component.ecosystem:
                continue
        for range_entry in affected.get("ranges", []):
            if not isinstance(range_entry, dict):
                continue
            for event in range_entry.get("events", []):
                if isinstance(event, dict) and event.get("fixed"):
                    fixed.add(str(event["fixed"]))
        for version in affected.get("versions", []):
            if isinstance(version, str) and version.startswith(">="):
                fixed.add(version.removeprefix(">="))
    return sorted(fixed)


def severity_for(vuln: dict[str, Any]) -> str:
    database_specific = vuln.get("database_specific", {})
    if isinstance(database_specific, dict):
        severity = database_specific.get("severity")
        if isinstance(severity, str) and severity:
            return severity.lower()
    severities = vuln.get("severity", [])
    if isinstance(severities, list) and severities:
        score = severities[0]
        if isinstance(score, dict):
            return str(score.get("score", score.get("type", "unknown")))
    return "unknown"


def advisory_ids(vuln: dict[str, Any]) -> list[str]:
    ids = [str(vuln.get("id", ""))]
    aliases = vuln.get("aliases", [])
    if isinstance(aliases, list):
        ids.extend(str(alias) for alias in aliases if alias)
    return sorted({item for item in ids if item})


def classify_status(
    ids: list[str], allowlist: dict[str, dict[str, str]], today: date
) -> tuple[str, dict[str, str] | None]:
    for advisory_id in ids:
        entry = allowlist.get(advisory_id)
        if entry is None:
            continue
        expires = parse_date(entry.get("expires"))
        if expires is None:
            return "unpatched_allowlist_invalid", entry
        if expires < today:
            return "unpatched_allowlist_expired", entry
        return f"accepted_until_{expires.isoformat()}", entry
    return "unpatched", None


def build_findings(
    components: list[Component],
    osv_results: dict[tuple[str, str, str], list[dict[str, Any]]],
    *,
    allowlist: dict[str, dict[str, str]],
    today: date,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for component in components:
        for vuln in osv_results.get(component.key, []):
            published = parse_date(vuln.get("published"))
            modified = parse_date(vuln.get("modified"))
            ids = advisory_ids(vuln)
            status, allowlist_entry = classify_status(ids, allowlist, today)
            findings.append(
                {
                    "component": component.name,
                    "version": component.version,
                    "ecosystem": component.ecosystem,
                    "sources": sorted(component.sources),
                    "used_by_src": component.used_by_src,
                    "advisory": vuln.get("id"),
                    "aliases": [item for item in ids if item != vuln.get("id")],
                    "published": published.isoformat() if published else None,
                    "modified": modified.isoformat() if modified else None,
                    "severity": severity_for(vuln),
                    "summary": vuln.get("summary"),
                    "fixed_versions": fixed_versions_for(vuln, component),
                    "status": status,
                    "allowlist": allowlist_entry,
                    "references": [
                        ref.get("url")
                        for ref in vuln.get("references", [])
                        if isinstance(ref, dict) and ref.get("url")
                    ],
                }
            )
    return sorted(
        findings,
        key=lambda item: (
            not bool(item["used_by_src"]),
            str(item["severity"]),
            str(item["component"]),
            str(item["advisory"]),
        ),
    )


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    findings = report["findings"]
    lines = [
        "# CVE Audit",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Components audited: `{report['inventory']['component_count']}`",
        f"- Findings: `{len(findings)}`",
        "",
    ]
    if findings:
        lines.extend(
            [
                "| Component | Version | Source | Advisory | Published | Severity | Status | Fixed |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for finding in findings:
            source = ", ".join(finding["sources"][:3])
            if len(finding["sources"]) > 3:
                source += f", +{len(finding['sources']) - 3} more"
            fixed = ", ".join(finding["fixed_versions"]) or "not listed"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(finding["component"]),
                        str(finding["version"]),
                        source,
                        str(finding["advisory"]),
                        str(finding["published"] or ""),
                        str(finding["severity"]),
                        str(finding["status"]),
                        fixed,
                    ]
                )
                + " |"
            )
    else:
        lines.append("No OSV findings matched the audited locked versions.")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This report matches exact locked package versions against OSV. It does not",
            "claim that arbitrary first-party source code has or lacks a CVE; source code",
            "review and SAST remain separate audit lanes.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    components = collect_inventory(repo_root)
    allowlist = load_allowlist(repo_root / args.allowlist)
    if args.no_network:
        osv_results = {component.key: [] for component in components}
    else:
        osv_results = query_osv_batch(
            components, batch_size=args.batch_size, enrich=not args.no_enrich
        )
    findings = build_findings(
        components,
        osv_results,
        allowlist=allowlist,
        today=date.today(),
    )
    source_files = sorted(
        {source for component in components for source in component.sources}
    )
    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "sources": {
            "advisory": ["OSV"],
            "inventory_files": source_files,
            "allowlist": args.allowlist,
        },
        "inventory": {
            "component_count": len(components),
            "src_used_component_count": sum(
                1 for component in components if component.used_by_src
            ),
            "source_file_count": len(source_files),
        },
        "findings": findings,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--allowlist", default="scripts/security/pip_audit_allowlist.json"
    )
    parser.add_argument("--out-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--out-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip per-advisory OSV enrichment after batch matching.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out_json = Path(args.repo_root).resolve() / args.out_json
    out_md = Path(args.repo_root).resolve() / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown_report(report, out_md)
    print(f"Audited {report['inventory']['component_count']} components")
    print(f"Findings: {len(report['findings'])}")
    print(f"JSON: {out_json}")
    print(f"Markdown: {out_md}")
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
