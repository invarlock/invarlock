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
    owner: str
    expires: date
    tracking_issue: str
    reason: str


def _load_allowlist(path: Path) -> tuple[str, list[AllowlistEntry]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Allowlist file must contain an object: {path}")

    owner = str(payload.get("owner", "")).strip()
    if not owner:
        raise SystemExit(f"Allowlist owner missing in {path}")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise SystemExit(f"Allowlist entries missing in {path}")

    today = date.today()
    entries: list[AllowlistEntry] = []
    for index, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise SystemExit(f"Allowlist entry {index} in {path} must be an object")
        advisory = str(raw_entry.get("advisory", "")).strip()
        entry_owner = str(raw_entry.get("owner", "")).strip()
        expires_raw = str(raw_entry.get("expires", "")).strip()
        tracking_issue = str(raw_entry.get("tracking_issue", "")).strip()
        reason = str(raw_entry.get("reason", "")).strip()
        if (
            not advisory
            or not entry_owner
            or not expires_raw
            or not tracking_issue
            or not reason
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
                owner=entry_owner,
                expires=expires,
                tracking_issue=tracking_issue,
                reason=reason,
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

    owner, entries = _load_allowlist(Path(args.allowlist))
    print(f"Using pip-audit allowlist owned by {owner}", file=sys.stderr)
    for entry in entries:
        print(
            f"Allowing {entry.advisory} until {entry.expires.isoformat()} "
            f"({entry.owner}; {entry.tracking_issue}): {entry.reason}",
            file=sys.stderr,
        )

    cmd = ["pip-audit"]
    for path in args.path:
        cmd.extend(["--path", path])
    for requirement in args.requirement:
        cmd.extend(["-r", requirement])
    for entry in entries:
        cmd.extend(["--ignore-vuln", entry.advisory])

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
