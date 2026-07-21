#!/usr/bin/env python3
"""Run pip-audit with a centrally owned, time-boxed allowlist."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

MAX_ALLOWLIST_DAYS = 30
_GITHUB_ISSUE_RE = re.compile(r"^https://github\.com/[^/]+/[^/]+/issues/[1-9]\d*$")


@dataclass(frozen=True)
class AllowlistEntry:
    advisory: str
    packages: tuple[str, ...]
    versions: tuple[str, ...]
    allowed_sources: tuple[str, ...]
    owner: str
    expires: date
    tracking_issue: str
    reason: str
    compensating_control: str


def load_pip_audit_allowlist(path: Path) -> tuple[str, list[AllowlistEntry]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Allowlist file must contain an object: {path}")
    if set(payload) != {"owner", "entries"}:
        raise SystemExit(f"Allowlist file has unsupported fields: {path}")

    owner = str(payload.get("owner", "")).strip()
    if not owner:
        raise SystemExit(f"Allowlist owner missing in {path}")

    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list):
        raise SystemExit(f"Allowlist entries in {path} must be a list")

    today = date.today()
    entries: list[AllowlistEntry] = []
    for index, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise SystemExit(f"Allowlist entry {index} in {path} must be an object")
        if set(raw_entry) != {
            "advisory",
            "allowed_sources",
            "compensating_control",
            "expires",
            "owner",
            "packages",
            "reason",
            "tracking_issue",
            "versions",
        }:
            raise SystemExit(
                f"Allowlist entry {index} in {path} has unsupported fields"
            )
        advisory = str(raw_entry.get("advisory", "")).strip()
        packages = raw_entry.get("packages")
        versions = raw_entry.get("versions")
        allowed_sources = raw_entry.get("allowed_sources")
        entry_owner = str(raw_entry.get("owner", "")).strip()
        expires_raw = str(raw_entry.get("expires", "")).strip()
        tracking_issue = str(raw_entry.get("tracking_issue", "")).strip()
        reason = str(raw_entry.get("reason", "")).strip()
        compensating_control = str(raw_entry.get("compensating_control", "")).strip()
        if (
            not advisory
            or not isinstance(packages, list)
            or not packages
            or not all(isinstance(item, str) and item.strip() for item in packages)
            or not isinstance(versions, list)
            or not versions
            or not all(isinstance(item, str) and item.strip() for item in versions)
            or not isinstance(allowed_sources, list)
            or not allowed_sources
            or not all(
                isinstance(item, str)
                and item.strip()
                and not Path(item).is_absolute()
                and ".." not in Path(item).parts
                for item in allowed_sources
            )
            or not entry_owner
            or not expires_raw
            or not tracking_issue
            or not reason
            or not compensating_control
        ):
            raise SystemExit(f"Allowlist entry {index} in {path} is incomplete")
        expires = date.fromisoformat(expires_raw)
        if expires < today:
            raise SystemExit(
                f"Allowlist entry {advisory} owned by {entry_owner} expired on {expires}"
            )
        if (expires - today).days > MAX_ALLOWLIST_DAYS:
            raise SystemExit(
                f"Allowlist entry {advisory} exceeds {MAX_ALLOWLIST_DAYS} days: {expires}"
            )
        if not _GITHUB_ISSUE_RE.fullmatch(tracking_issue):
            raise SystemExit(
                f"Allowlist entry {advisory} must link to a GitHub tracking issue"
            )
        entries.append(
            AllowlistEntry(
                advisory=advisory,
                packages=tuple(
                    sorted(
                        {
                            re.sub(r"[-_.]+", "-", item.strip()).lower()
                            for item in packages
                        }
                    )
                ),
                versions=tuple(sorted({item.strip() for item in versions})),
                allowed_sources=tuple(
                    sorted({item.strip() for item in allowed_sources})
                ),
                owner=entry_owner,
                expires=expires,
                tracking_issue=tracking_issue,
                reason=reason,
                compensating_control=compensating_control,
            )
        )

    return owner, entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allowlist",
        default="scripts/security/pip_audit_allowlist.json",
        help="Path to the pip-audit allowlist JSON file",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help=(
            "Installation path to audit. May be supplied multiple times "
            "to audit more than one installed surface."
        ),
    )
    parser.add_argument(
        "--requirement",
        action="append",
        default=[],
        help=(
            "Requirements file to audit without installing the surface first. "
            "May be supplied multiple times."
        ),
    )
    args = parser.parse_args(argv)

    owner, entries = load_pip_audit_allowlist(Path(args.allowlist))
    print(f"Using pip-audit allowlist owned by {owner}", file=sys.stderr)
    for entry in entries:
        print(
            f"Registered {entry.advisory} for {', '.join(entry.packages)} "
            f"until {entry.expires.isoformat()} "
            f"({entry.owner}; {entry.tracking_issue}): {entry.reason}",
            file=sys.stderr,
        )
        print(
            f"Compensating control for {entry.advisory}: {entry.compensating_control}",
            file=sys.stderr,
        )

    cmd = ["pip-audit"]
    for path in args.path:
        cmd.extend(["--path", path])
    for requirement in args.requirement:
        cmd.extend(["-r", requirement])
    requirement_pins: set[tuple[str, str, str]] = set()
    for requirement in args.requirement:
        requirement_path = Path(requirement)
        try:
            source = (
                requirement_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
            )
        except ValueError:
            source = requirement_path.as_posix()
        for line in requirement_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\;]+)", line)
            if match:
                package = re.sub(r"[-_.]+", "-", match.group(1)).lower()
                requirement_pins.add((package, match.group(2), source))
    for entry in entries:
        package_pins = {
            (package, version, source)
            for package, version, source in requirement_pins
            if package in entry.packages
        }
        applies = (
            not args.path
            and bool(package_pins)
            and all(
                version in entry.versions and source in entry.allowed_sources
                for _package, version, source in package_pins
            )
        )
        if applies:
            cmd.extend(["--ignore-vuln", entry.advisory])

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


_load_allowlist = load_pip_audit_allowlist


if __name__ == "__main__":
    raise SystemExit(main())
