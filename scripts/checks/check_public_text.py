#!/usr/bin/env python3
"""Scan tracked Markdown for private details and process-only wording."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    message: str


@dataclass(frozen=True)
class Finding:
    source: str
    line: int
    rule: Rule


RULES: tuple[Rule, ...] = (
    Rule(
        "root_ssh_target",
        re.compile(r"\broot@(?:\d{1,3}(?:\.\d{1,3}){3}|[A-Za-z0-9_.-]+\.[A-Za-z]{2,})"),
        "replace the SSH target with a capability-focused host description",
    ),
    Rule(
        "operator_ssh_target",
        re.compile(
            r"\b(?:ubuntu|admin|ec2-user|debian|centos)@"
            r"(?:\d{1,3}(?:\.\d{1,3}){3}|[A-Za-z0-9_.-]+\.[A-Za-z]{2,})"
        ),
        "replace the SSH target with a capability-focused host description",
    ),
    Rule(
        "absolute_home_path",
        re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+(?:/|\b)"),
        "use a repository-relative path or a generic location",
    ),
    Rule(
        "absolute_root_path",
        re.compile(r"/root/[^\s`'\"),;]+"),
        "use a repository-relative path or a generic location",
    ),
    Rule(
        "desktop_runtime_path",
        re.compile(
            r"(?:\.automation/|desktop-primary-runtime|Automation\.app|"
            r"/var/run/com\.apple\.security\.cryptexd/)"
        ),
        "remove local automation and desktop-runtime details",
    ),
    Rule(
        "private_temp_path",
        re.compile(r"/(?:private/tmp|var/folders)/[^\s`'\"),;]+"),
        "describe the temporary location without a machine-specific path",
    ),
    Rule(
        "full_path_environment",
        re.compile(r"(?<![A-Z_])PATH=(?:\"[^\"]*\"|'[^']*'|[^\s`]+)"),
        "publish the command without the machine-specific PATH value",
    ),
    Rule(
        "credential_like_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|"
            r"password|authorization|bearer)\b\s*[:=]\s*"
            r"['\"]?[A-Za-z0-9_./+=:-]{12,}"
        ),
        "remove the credential-like value and rotate it if it was published",
    ),
    Rule(
        "review_process_status",
        re.compile(
            r"(?i)\b(?:public\s+)?(?:pr\s+)?(?:privacy|trust)\s+gate\s+"
            r"(?:passed|failed|clean|clear|ok|approved|checked)\b|"
            r"\b(?:adversarial|privacy|trust)\s+(?:audit|review)\s+"
            r"(?:passed|failed|completed|clean|clear|approved)\b|"
            r"\b(?:checked|generated|implemented|reviewed)\s+by\s+"
            r"(?:an?\s+(?:automated\s+reviewer|(?:coding\s+)?agent))\b|"
            r"\b(?:automated|agent)[ -](?:generated|reviewed)\b"
        ),
        "state the public result without describing the review mechanism",
    ),
    Rule(
        "avoidable_remote_validation_claim",
        re.compile(
            r"(?i)\b(?:private|remote|private\s+remote)\s+"
            r"(?:[A-Za-z0-9_.-]+\s+){0,4}?"
            r"(?:validation|runner|run|smoke|evidence|test|testing)\s+host\b|"
            r"\bremote\s+(?:all-lane\s+)?validation\s+passed\b"
        ),
        "describe the validated capability without private execution logistics",
    ),
    Rule(
        "planning_or_workspace_note",
        re.compile(
            r"(?i)\b(?:implementation|engineering|product)\s+backlog\b|"
            r"\bbacklog\s+(?:lane|family|item|work)s?\b|"
            r"\bdependabot-equivalent\b|"
            r"\b(?:compact\s+)?(?:evaluator\s+)?release\s+focus\b|"
            r"\brelease\s+breadth\s+target\b|"
            r"\b(?:catalog|matrix)\s+is\s+reviewed\s+rather\s+than\s+"
            r"quota-driven\b|"
            r"\breserved\s+for\s+a\s+future\s+claim\b|"
            r"\bincreasing\s+to\s+\d+\s+records?\s+would\s+add\b|"
            r"\b(?:fresh[- ]worktree|worktree-aware)\s+(?:remote\s+)?"
            r"(?:guidance|launch(?:es|ing)?)\b|"
            r"\bnongated\s+(?:replacement\s+)?backlog\b|"
            r"\bearlier\s+internal\s+development\b"
        ),
        "replace planning or workspace notes with the shipped user-facing outcome",
    ),
)


def tracked_markdown_paths(root: Path) -> list[Path]:
    """Return every Markdown path tracked by Git under *root*."""

    result = subprocess.run(
        ["git", "ls-files", "-z", "--", ":(icase,glob)**/*.md"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        root / raw_path.decode("utf-8")
        for raw_path in result.stdout.split(b"\0")
        if raw_path
    ]


def findings_for_text(source: str, text: str) -> list[Finding]:
    """Return all public-text findings in *text*."""

    findings: list[Finding] = []
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            findings.append(
                Finding(
                    source=source,
                    line=text.count("\n", 0, match.start()) + 1,
                    rule=rule,
                )
            )
    return sorted(findings, key=lambda finding: (finding.line, finding.rule.name))


def scan_paths(paths: list[Path], root: Path) -> list[Finding]:
    """Scan *paths*, using repository-relative names where possible."""

    findings: list[Finding] = []
    for path in paths:
        try:
            source = path.relative_to(root).as_posix()
        except ValueError:
            source = "<external-file>"
        findings.extend(findings_for_text(source, path.read_text(encoding="utf-8")))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan tracked Markdown for private operational details and "
            "process-only wording."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional files to scan; defaults to every tracked Markdown file.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root used for discovery and displayed paths.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    try:
        paths = [path.resolve() for path in args.paths]
        if not paths:
            paths = tracked_markdown_paths(root)
        findings = scan_paths(paths, root)
    except (OSError, UnicodeError, subprocess.CalledProcessError) as exc:
        print(
            f"Public text check could not run ({type(exc).__name__}); "
            "verify the repository and UTF-8 inputs.",
            file=sys.stderr,
        )
        return 2

    for finding in findings:
        print(
            f"{finding.source}:{finding.line}: {finding.rule.name}: "
            f"{finding.rule.message}",
            file=sys.stderr,
        )

    if findings:
        print(
            f"Public text check failed: {len(findings)} finding(s).",
            file=sys.stderr,
        )
        return 1

    print(f"Public text check passed for {len(paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
