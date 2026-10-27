#!/usr/bin/env python3
"""Scan working-tree Markdown for private details and process-only wording."""

from __future__ import annotations

import argparse
import hashlib
import json
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
        "review_staffing_language",
        re.compile(
            r"(?i)\bhuman[\s-]+(?:approval|review|judg(?:ment|ement)|readable|output|report|summary)\b"
        ),
        "describe the policy or reference-label requirement without staffing language",
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


# Only this wording rule is stylistic. Operational/privacy rules always apply.
UNCHANGED_VENDOR_STYLE_RULES = frozenset({"review_staffing_language"})


def unchanged_vendored_markdown(path: Path, root: Path, raw: bytes) -> bool:
    """Recognize unchanged upstream bytes; missing or ambiguous evidence denies exemption."""

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate provenance key")
            value[key] = item
        return value

    try:
        relative = path.absolute().relative_to(root.resolve())
        if (
            relative.parts[:2] != ("native", "vendor")
            or len(relative.parts) < 4
            or path.suffix.lower() != ".md"
            or path.resolve() != path.absolute()
        ):
            return False
        package = root.resolve().joinpath(*relative.parts[:3])
        provenance_path = package / "source-provenance.json"
        if provenance_path.is_symlink():
            return False
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
        if not isinstance(provenance, dict):
            return False
        name, version = provenance["name"], provenance["version"]
        if (
            name != package.name
            or not isinstance(version, str)
            or not version
            or provenance["origin"]
            != "external upstream dependency; not InvarLock implementation"
            or provenance["source_url"]
            != f"https://static.crates.io/crates/{name}/{name}-{version}.crate"
            or re.fullmatch(r"[0-9a-f]{64}", provenance["archive_sha256"]) is None
        ):
            return False
        document = path.relative_to(package).as_posix()
        modified = provenance["modified_upstream_files"]
        if (
            not isinstance(modified, list)
            or not all(isinstance(item, str) for item in modified)
            or document in modified
        ):
            return False
        actual = hashlib.sha256(raw).hexdigest()
        for field in ("upstream_file_hashes", "patched_file_hashes"):
            filename = provenance[field]
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or filename in (".", "..")
            ):
                return False
            manifest_path = package / filename
            if manifest_path.is_symlink():
                return False
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                object_pairs_hook=unique_object,
            )
            if not isinstance(manifest, dict) or manifest.get(document) != actual:
                return False
        return True
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        return False


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


def working_tree_markdown_paths(root: Path) -> list[Path]:
    """Include nonignored additions and exclude only Git-reported deletions."""
    pathspec = ["--", ":(icase,glob)**/*.md"]
    additions = subprocess.run(
        ["git", "ls-files", "-z", "--others", "--exclude-standard", *pathspec],
        cwd=root,
        check=True,
        capture_output=True,
    )
    deletions = subprocess.run(
        ["git", "ls-files", "-z", "--deleted", *pathspec],
        cwd=root,
        check=True,
        capture_output=True,
    )
    added = {
        root / name.decode("utf-8") for name in additions.stdout.split(b"\0") if name
    }
    deleted = {
        root / name.decode("utf-8") for name in deletions.stdout.split(b"\0") if name
    }
    return sorted((set(tracked_markdown_paths(root)) | added) - deleted)


def scan_paths(paths: list[Path], root: Path) -> list[Finding]:
    """Scan *paths*, using repository-relative names where possible."""

    findings: list[Finding] = []
    for path in paths:
        try:
            source = path.relative_to(root).as_posix()
        except ValueError:
            source = "<external-file>"
        raw = path.read_bytes()
        selected = findings_for_text(source, raw.decode("utf-8"))
        if any(
            finding.rule.name in UNCHANGED_VENDOR_STYLE_RULES for finding in selected
        ) and unchanged_vendored_markdown(path, root, raw):
            selected = [
                finding
                for finding in selected
                if finding.rule.name not in UNCHANGED_VENDOR_STYLE_RULES
            ]
        findings.extend(selected)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan working-tree Markdown for private operational details and "
            "process-only wording."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional files to scan; defaults to tracked and nonignored new Markdown, excluding Git-reported deletions.",
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
        paths = [path.absolute() for path in args.paths]
        if not paths:
            paths = working_tree_markdown_paths(root)
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
