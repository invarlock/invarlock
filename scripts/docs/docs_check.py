#!/usr/bin/env python3
"""Consolidated documentation checks.

Runs the common docs validations in one place so CI and local developers can
invoke a single entry point. Use flags to select subsets or --all for the full
suite.

Examples:
  python scripts/docs/docs_check.py --all
  python scripts/docs/docs_check.py --build --links
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CURATED_LIVE_EXAMPLE_PATHS = (
    "README.md",
    "docs/user-guide/getting-started.md",
    "docs/user-guide/quickstart.md",
    "notebooks/invarlock_python_api.ipynb",
    "notebooks/invarlock_policy_tiers.ipynb",
)

DOCS_ROOT_NAME = "docs"
LINK_RE = re.compile(r"\]\(([^)]+)\)")
SNIPPET_RE = re.compile(r"--8<--\s*\"([^\"]+)\"")
ANCHOR_PATTERN = re.compile(r"\[[^\]]+\]\((#[^)]+)\)")
API_REF_PATTERN = re.compile(r"\b(invarlock(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b")
VERSION_PATTERN = re.compile(r'__version__\s*=\s*"([^"]+)"')
PYPROJECT_VERSION_PATTERN = re.compile(r'^\s*version\s*=\s*"([^"]+)"\s*$', re.M)
CITATION_VERSION_PATTERN = re.compile(
    r"^\s*version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*$", re.M
)
COMMANDS = {
    "invarlock evaluate",
    "invarlock report",
    "invarlock verify",
    "invarlock doctor",
    "invarlock advanced",
}
EXPECTED_CONFIG_KEYS = {
    "model:",
    "dataset:",
    "edit:",
    "auto:",
    "guards:",
    "eval:",
    "output:",
}
GUARD_HEADINGS = {
    "### Invariants Guard",
    "### Spectral Guard",
    "### RMT Guard",
    "### Variance Guard",
}
REMOVED_README_GUARANTEE_LABEL = "Statistical " + "guarantees"
REMOVED_REPORT_GUARANTEE_LABEL = "What the report " + "guarantees"
API_REF_IGNORE_LAST_SEGMENT = {
    "dev",
    "gif",
    "jpeg",
    "jpg",
    "json",
    "md",
    "png",
    "svg",
    "yaml",
    "yml",
}
API_REF_EXCLUDE_TOP_LEVEL_DIRS = {
    ".evaluate_tmp",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "plans",
    "reports",
    "runs",
    "tmp",
    "venv",
    "worktrees",
}

CLAIM_REQUIRED_BY_FILE = {
    "docs/user-guide/quickstart.md": [
        "machine-readable evaluation report",
        "invarlock verify",
        "report html",
    ],
    "docs/user-guide/getting-started.md": [
        "invarlock evaluate",
        "invarlock verify",
        "report html",
        "Assurance Case",
    ],
    "docs/user-guide/example-reports.md": [
        "Machine-readable evaluation report",
    ],
    "SUPPORT.md": ["evaluation workflows"],
    "scripts/smoke/run_tiny_all_matrix.sh": ["Evaluation Matrix"],
    "tests/integration/scripts/test_tiny_matrix_checklist.py": [
        "Evaluation Matrix",
    ],
    "docs/README.md": [
        "Assurance Case",
        "Evaluation Math Derivation",
        "Published assurance basis covers GPT-2 and BERT profiles.",
        "Mistral 7B",
        "Qwen2 7B",
        "Qwen2.5 7B",
        "Qwen2.5 14B",
        "pilot calibration configs",
        "Model Family Catalog",
    ],
    "mkdocs.yml": [
        "Model Family Catalog: reference/model-family-catalog.md",
        "Assurance Case: assurance/00-assurance-case.md",
        "Evaluation Math Derivation: assurance/01-eval-math-derivation.md",
    ],
    "README.md": ["docs/assurance/00-assurance-case.md"],
    "docs/user-guide/reading-report.md": ["Assurance Case"],
    "docs/reference/index.md": ["Assurance claims and derivations"],
    "docs/reference/contracts.md": ["Model family catalog"],
    "docs/reference/model-family-catalog.md": [
        "support tier",
        "coverage state",
        "Declared Support",
        "<=14B Text Candidate Inventory",
        "Recommended Additions",
    ],
    "docs/assurance/00-assurance-case.md": [
        "assurance case",
        "assurance claims",
        "published assurance tiers",
    ],
    "docs/assurance/04-guard-contracts.md": [
        "Published assurance basis covers GPT-2 and BERT profiles.",
        "Mistral 7B",
        "Qwen2 7B",
        "Qwen2.5 7B",
        "Qwen2.5 14B",
        "not part of the published",
    ],
    "docs/reference/calibration.md": [
        "Published assurance basis covers GPT-2 and BERT profiles.",
        "Mistral 7B",
        "Qwen2 7B",
        "Qwen2.5 7B",
        "Qwen2.5 14B",
    ],
    "docs/reference/guards.md": [
        "GPT-2",
        "BERT profiles",
        "Mistral 7B",
        "Qwen2 7B",
        "Qwen2.5 7B",
        "Qwen2.5 14B",
        "published assurance basis",
    ],
    "docs/reference/model-adapters.md": [
        "Adapter availability is broader than the published assurance basis.",
        "GPT-2",
        "BERT",
        "Mistral 7B",
        "Qwen2 7B",
        "Qwen2.5 7B",
        "Qwen2.5 14B",
    ],
    "docs/user-guide/evidence-packs.md": [
        "signed manifest",
        "strict verification",
        "PASS final verdict",
    ],
    "scripts/evidence_packs/run_pack.sh": [
        "signed manifest",
        "strict verification",
        "PASS final verdict",
    ],
    "scripts/evidence_packs/tests/test_run_pack.sh": [
        "signed manifest, strict verification, and a PASS final verdict",
    ],
    "Makefile": ["eval-loop:"],
}

CLAIM_BANNED_BY_FILE = {
    "README.md": [REMOVED_README_GUARANTEE_LABEL],
    "docs/user-guide/quickstart.md": ["machine-readable safety report"],
    "docs/user-guide/getting-started.md": [
        "make cert-loop",
        "safety guarantees",
        "[Safety Case]",
    ],
    "docs/user-guide/example-reports.md": ["Signed compliance payload"],
    "SUPPORT.md": ["certif" + "ication flow"],
    "scripts/smoke/run_tiny_all_matrix.sh": [
        "Certif" + "ication Matrix",
        "gpt2_cert_",
        "gpt2_editcert_quant8",
        "bert_mlm_cert",
        "distilbert_cls_cert",
    ],
    "tests/integration/scripts/test_tiny_matrix_checklist.py": [
        "Certif" + "ication Matrix",
    ],
    "mkdocs.yml": ["Safety Case:"],
    "docs/README.md": ["Safety Case", "safety claim"],
    "docs/reference/index.md": ["Safety claims and proofs"],
    "docs/reference/architecture.md": ["Safety Case Overview"],
    "docs/reference/reports.md": ["Safety Case", REMOVED_REPORT_GUARANTEE_LABEL],
    "docs/user-guide/reading-report.md": ["[Safety Case]"],
    "docs/user-guide/primary-metric-smoke.md": ["Evaluation Math Proof"],
    "docs/assurance/00-assurance-case.md": [
        "safety case",
        "safety claims",
        "safety tiers",
    ],
    "Makefile": ["cert-loop:"],
    "scripts/evidence_packs/lib/tasks/task_functions.sh": ["Certif" + "ication for "],
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def path_contains_all(path: Path, snippets: set[str]) -> bool:
    if not path.exists():
        return False
    text = read_text(path)
    return all(snippet in text for snippet in snippets)


def rel_to_repo(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = proc.communicate()
    return proc.returncode, out


def _raise_on_code(code: int) -> None:
    if code != 0:
        raise SystemExit(code)


def check_build() -> None:
    code, out = run([sys.executable, "-m", "mkdocs", "build", "--strict"])
    print(out, end="")
    _raise_on_code(code)


def _is_external(link: str) -> bool:
    lowered = link.lower()
    return lowered.startswith(("http://", "https://", "mailto:", "tel:"))


def _check_docs_links(root: Path) -> list[str]:
    docs_root = root / DOCS_ROOT_NAME
    missing: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        text = read_text(path)
        display_path = rel_to_repo(path, root)
        for snippet in SNIPPET_RE.findall(text):
            resolved = (docs_root / snippet).resolve()
            if not resolved.exists():
                missing.append(f"{display_path}: missing snippet -> {snippet}")
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith("#") or _is_external(target):
                continue
            if target.startswith("!include") or target.startswith("--8<--"):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            resolved = (path.parent / target_path).resolve()
            if not resolved.exists():
                missing.append(f"{display_path}: broken link -> {target}")
    return missing


def check_docs_links() -> int:
    missing = _check_docs_links(repo_root())
    if missing:
        for entry in missing:
            print(entry)
        print(f"Found {len(missing)} broken documentation links", file=sys.stderr)
        return 1
    print("Documentation links valid")
    return 0


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\-\s]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    return re.sub(r"-+", "-", s)


def check_internal_links_for_file(md_path: Path, root: Path | None = None) -> int:
    root = root or repo_root()
    if not md_path.exists():
        print(f"File not found: {md_path}")
        return 1

    text = read_text(md_path)
    anchors = {
        _slugify(match) for match in re.findall(r"^#+\s*(.*)", text, flags=re.MULTILINE)
    }

    missing: list[str] = []
    for match in ANCHOR_PATTERN.finditer(text):
        anchor = match.group(1)[1:].strip()
        slug = _slugify(anchor)
        if slug not in anchors:
            missing.append(slug)

    if missing:
        print(f"{rel_to_repo(md_path, root)}: missing anchors -> {missing}")
        return 1
    return 0


def check_links() -> None:
    root = repo_root()
    _raise_on_code(check_docs_links())

    for md_path in sorted((root / DOCS_ROOT_NAME).rglob("*.md")):
        _raise_on_code(check_internal_links_for_file(md_path, root))


def _iter_reference_markdown_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def _resolve_doc_reference(markdown_file: Path, link: str, docs_root: Path) -> Path:
    target = link.split("#", 1)[0].strip()
    if not target:
        return Path()
    if target.startswith("/"):
        return (docs_root / target.lstrip("/")).resolve()
    return (markdown_file.parent / target).resolve()


def _validate_doc_reference_file(markdown_file: Path, docs_root: Path) -> list[str]:
    errors: list[str] = []
    text = read_text(markdown_file)
    for match in LINK_RE.finditer(text):
        raw_link = match.group(1).strip()
        if not raw_link or raw_link.startswith("#") or _is_external(raw_link):
            continue
        candidate = _resolve_doc_reference(markdown_file, raw_link, docs_root)
        if not candidate:
            continue
        if candidate.exists():
            continue
        if candidate.suffix == "" and candidate.with_suffix(".md").exists():
            continue
        errors.append(f"{markdown_file.relative_to(docs_root)} -> {raw_link}")
    return errors


def _validate_doc_references(root: Path) -> list[str]:
    docs_root = root / DOCS_ROOT_NAME
    if not docs_root.exists():
        return []
    errors: list[str] = []
    for md_file in _iter_reference_markdown_files(docs_root):
        errors.extend(_validate_doc_reference_file(md_file, docs_root))
    return errors


def check_references() -> None:
    errors = _validate_doc_references(repo_root())
    if errors:
        print("Broken documentation links detected:")
        for item in errors:
            print(f"  - {item}")
        raise SystemExit(1)
    print("All documentation references resolved successfully.")


@dataclass(frozen=True)
class ApiRef:
    file: str
    line: int
    text: str


def _is_api_ref_excluded_top_level(name: str) -> bool:
    return name.startswith(".") or name in API_REF_EXCLUDE_TOP_LEVEL_DIRS


def iter_api_refs(paths: Iterable[Path]) -> list[ApiRef]:
    results: list[ApiRef] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for match in API_REF_PATTERN.finditer(line):
                symbol = match.group(1)
                last = symbol.rsplit(".", 1)[-1]
                if last in API_REF_IGNORE_LAST_SEGMENT:
                    continue
                start, end = match.span(1)
                before = line[start - 1] if start - 1 >= 0 else ""
                after = line[end] if end < len(line) else ""
                if before in {'"', "'"} and after in {'"', "'"}:
                    continue
                url_context = line[max(0, start - 12) : min(len(line), end + 4)]
                if "://" in url_context:
                    continue
                results.append(ApiRef(file=str(path), line=line_number, text=symbol))
    return results


def resolve_api_ref(symbol: str) -> tuple[bool, str | None]:
    parts = symbol.split(".")
    mod_path: str | None = None
    optional_dep_missing = False
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        try:
            importlib.import_module(candidate)
            mod_path = candidate
            break
        except ModuleNotFoundError as exc:
            name = getattr(exc, "name", "") or ""
            if not name.startswith("invarlock"):
                optional_dep_missing = True
                break
            continue
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            SyntaxError,
            ValueError,
        ):
            optional_dep_missing = True
            break
    if mod_path is None:
        if optional_dep_missing:
            return True, None
        return False, "module not found"

    obj = importlib.import_module(mod_path)
    for attr in parts[len(mod_path.split(".")) :]:
        if not hasattr(obj, attr):
            return False, f"attribute '{attr}' missing on {obj!r}"
        obj = getattr(obj, attr)
    return True, None


def _iter_api_ref_markdown_files(root: Path) -> list[Path]:
    md_files: list[Path] = []
    for path in root.glob("**/*.md"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if rel_parts and _is_api_ref_excluded_top_level(rel_parts[0]):
            continue
        md_files.append(path)
    return sorted(md_files, key=lambda p: str(p))


def check_api_refs() -> int:
    root = repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    seen: set[tuple[str, int, str]] = set()
    unique: list[ApiRef] = []
    for ref in iter_api_refs(_iter_api_ref_markdown_files(root)):
        key = (ref.file, ref.line, ref.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)

    ok = 0
    failed = 0
    failures = io.StringIO()
    tmp_dir = root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / "docs_api_refs_results.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for ref in unique:
            success, err = resolve_api_ref(ref.text)
            out.write(
                json.dumps(
                    {
                        "file": ref.file,
                        "line": ref.line,
                        "symbol": ref.text,
                        "ok": success,
                        "error": err,
                    }
                )
                + "\n"
            )
            if success:
                ok += 1
            else:
                failed += 1
                failures.write(
                    f"{ref.file}:{ref.line}: unresolved {ref.text} - {err}\n"
                )

    print(f"Checked {len(unique)} doc references; ok={ok}; failed={failed}")
    if failed:
        print("--- failures ---")
        print(failures.getvalue().rstrip())
        return 1
    return 0


def iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def validate_python_examples() -> int:
    root = repo_root()
    docs_root = root / DOCS_ROOT_NAME

    if not docs_root.exists():
        print("Docs directory not found; skipping validation.")
        return 0

    python_files = iter_python_files(docs_root)
    if not python_files:
        print("No Python documentation examples found.")
        return 0

    errors: list[str] = []
    for path in python_files:
        try:
            compile(read_text(path), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{rel_to_repo(path, root)}: {exc}")

    if errors:
        print("Python example validation failed:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print(f"Validated {len(python_files)} Python example(s).")
    return 0


def iter_yaml_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.suffix in {".yaml", ".yml"} and p.is_file()
    )


def validate_yaml_examples() -> int:
    import yaml

    root = repo_root()
    docs_root = root / DOCS_ROOT_NAME

    if not docs_root.exists():
        print("Docs directory not found; nothing to validate.")
        return 0

    yaml_files = iter_yaml_files(docs_root)
    if not yaml_files:
        print("No YAML documentation examples found.")
        return 0

    errors: list[str] = []
    for path in yaml_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            errors.append(f"{rel_to_repo(path, root)}: {exc}")

    if errors:
        print("Invalid YAML examples detected:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print(f"Validated {len(yaml_files)} YAML example(s).")
    return 0


def check_examples() -> None:
    _raise_on_code(validate_yaml_examples())
    _raise_on_code(validate_python_examples())

    cli_tester = repo_root() / "scripts" / "docs" / "test_cli_examples.py"
    if cli_tester.exists():
        code, out = run([sys.executable, "scripts/docs/test_cli_examples.py"])
        print(out, end="")
        _raise_on_code(code)
    else:
        print(
            "[docs_check] Skipping CLI examples test (scripts/docs/test_cli_examples.py not found)"
        )


def get_package_version(root: Path) -> str:
    init_path = root / "src" / "invarlock" / "__init__.py"
    match = VERSION_PATTERN.search(read_text(init_path))
    if not match:
        raise RuntimeError(
            "Could not determine package version from src/invarlock/__init__.py"
        )
    return match.group(1)


def check_version_consistency() -> int:
    root = repo_root()
    version = get_package_version(root)

    pyproject = root / "pyproject.toml"
    citation_cff = root / "CITATION.cff"

    missing: list[str] = []
    pyproject_match = PYPROJECT_VERSION_PATTERN.search(read_text(pyproject))
    if not pyproject_match or pyproject_match.group(1) != version:
        missing.append(rel_to_repo(pyproject, root))

    citation_match = CITATION_VERSION_PATTERN.search(read_text(citation_cff))
    if not citation_match or citation_match.group(1) != version:
        missing.append(rel_to_repo(citation_cff, root))

    if missing:
        print(f"Version {version} does not match in:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print(f"Metadata version strings match package version {version}.")
    return 0


def gather_documented_commands(doc_root: Path) -> set[str]:
    documented: set[str] = set()
    for md_file in doc_root.rglob("*.md"):
        text = read_text(md_file)
        for command in COMMANDS:
            if re.search(rf"\b{re.escape(command)}\b", text):
                documented.add(command)
    return documented


def check_cli_completeness() -> int:
    root = repo_root()
    docs_root = root / DOCS_ROOT_NAME
    if not docs_root.exists():
        print("Docs directory not found; skipping CLI completeness check.")
        return 0

    documented = gather_documented_commands(docs_root)
    missing = COMMANDS.difference(documented)

    if missing:
        print("The following CLI commands are not documented:")
        for command in sorted(missing):
            print(f"  - {command}")
        return 1

    print("All core CLI commands are documented.")
    return 0


def doc_contains_config_keys(path: Path) -> bool:
    return path_contains_all(path, EXPECTED_CONFIG_KEYS)


def check_config_schema_sync() -> int:
    root = repo_root()
    candidates = [
        root / "docs" / "reference" / "config-schema.md",
        root / "docs" / "README.md",
        root / "README.md",
    ]

    for candidate in candidates:
        if doc_contains_config_keys(candidate):
            print(f"Configuration schema documented in {rel_to_repo(candidate, root)}")
            return 0

    print("Configuration schema snippets not found in documentation.")
    return 1


def check_guard_completeness() -> int:
    root = repo_root()
    guards_doc = root / "docs" / "reference" / "guards.md"

    if not guards_doc.exists():
        print("Guard reference documentation not found.")
        return 1

    text = read_text(guards_doc)
    missing = [heading for heading in GUARD_HEADINGS if heading not in text]

    if missing:
        print("Missing guard sections in docs/reference/guards.md:")
        for heading in missing:
            print(f"  - {heading}")
        return 1

    print("All core guard sections are present in docs/reference/guards.md.")
    return 0


def _check_required_snippets(
    failures: list[str], root: Path, rel_path: str, required_snippets: list[str]
) -> None:
    text = read_text(root / rel_path)
    for snippet in required_snippets:
        if snippet not in text:
            failures.append(f"{rel_path}: missing required snippet: {snippet!r}")


def _check_banned_snippets(
    failures: list[str], root: Path, rel_path: str, banned_snippets: list[str]
) -> None:
    text = read_text(root / rel_path)
    for snippet in banned_snippets:
        if snippet in text:
            failures.append(f"{rel_path}: banned snippet present: {snippet!r}")


def check_claim_surface_consistency() -> int:
    root = repo_root()
    failures: list[str] = []

    for rel_path, required_snippets in CLAIM_REQUIRED_BY_FILE.items():
        _check_required_snippets(failures, root, rel_path, required_snippets)
    for rel_path, banned_snippets in CLAIM_BANNED_BY_FILE.items():
        _check_banned_snippets(failures, root, rel_path, banned_snippets)

    if failures:
        print("[check_claim_surface_consistency] FAIL", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        return 1

    print("[check_claim_surface_consistency] OK")
    return 0


def check_consistency() -> None:
    for check in (
        check_version_consistency,
        check_cli_completeness,
        check_claim_surface_consistency,
        check_config_schema_sync,
        check_guard_completeness,
    ):
        _raise_on_code(check())

    code, out = run([sys.executable, "scripts/docs/lint_assurance_xrefs.py"])
    print(out, end="")
    _raise_on_code(code)


def check_live() -> None:
    code, out = run([sys.executable, "scripts/docs/verify_live_examples.py"])
    print(out, end="")
    _raise_on_code(code)


def check_live_fast() -> None:
    code, out = run(
        [
            sys.executable,
            "scripts/docs/verify_live_examples.py",
            "--markdown-execution-mode",
            "host",
            "--skip-markdown-model-loading",
            "--skip-notebook-model-loading",
            "--paths",
            *CURATED_LIVE_EXAMPLE_PATHS,
        ]
    )
    print(out, end="")
    _raise_on_code(code)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolidated docs checks")
    p.add_argument("--all", action="store_true", help="Run all checks")
    p.add_argument("--build", action="store_true", help="Build MkDocs strictly")
    p.add_argument(
        "--links", action="store_true", help="Run link checks (global + internal)"
    )
    p.add_argument("--refs", action="store_true", help="Validate doc references")
    p.add_argument(
        "--api-refs",
        action="store_true",
        help="Validate documented invarlock.* API references",
    )
    p.add_argument(
        "--examples",
        action="store_true",
        help="Validate YAML/Python examples and CLI snippets if available",
    )
    p.add_argument(
        "--consistency",
        action="store_true",
        help="Run version/CLI/schema/guards/claim consistency checks",
    )
    p.add_argument(
        "--claim-surface",
        action="store_true",
        help="Run public claim-surface wording checks only",
    )
    p.add_argument(
        "--version-consistency",
        action="store_true",
        help="Run metadata version consistency checks only",
    )
    p.add_argument(
        "--cli-completeness",
        action="store_true",
        help="Run CLI documentation completeness checks only",
    )
    p.add_argument(
        "--config-schema-sync",
        action="store_true",
        help="Run config schema documentation checks only",
    )
    p.add_argument(
        "--guard-completeness",
        action="store_true",
        help="Run guard documentation completeness checks only",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Live-run runnable markdown CLI examples and notebooks",
    )
    p.add_argument(
        "--live-fast",
        action="store_true",
        help="Run the curated deterministic live-example subset",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not any(vars(args).values()):
        print("No checks selected. Use --all or individual flags.", file=sys.stderr)
        raise SystemExit(2)

    summary = {
        "build": None,
        "links": None,
        "refs": None,
        "api_refs": None,
        "examples": None,
        "consistency": None,
        "claim_surface": None,
        "version_consistency": None,
        "cli_completeness": None,
        "config_schema_sync": None,
        "guard_completeness": None,
        "live_fast": None,
        "live": None,
    }

    try:
        if args.all or args.build:
            check_build()
            summary["build"] = True
        if args.all or args.links:
            check_links()
            summary["links"] = True
        if args.all or args.refs:
            check_references()
            summary["refs"] = True
        if args.all or args.api_refs:
            _raise_on_code(check_api_refs())
            summary["api_refs"] = True
        if args.all or args.examples:
            check_examples()
            summary["examples"] = True
        if args.all or args.consistency:
            check_consistency()
            summary["consistency"] = True
        if args.claim_surface:
            _raise_on_code(check_claim_surface_consistency())
            summary["claim_surface"] = True
        if args.version_consistency:
            _raise_on_code(check_version_consistency())
            summary["version_consistency"] = True
        if args.cli_completeness:
            _raise_on_code(check_cli_completeness())
            summary["cli_completeness"] = True
        if args.config_schema_sync:
            _raise_on_code(check_config_schema_sync())
            summary["config_schema_sync"] = True
        if args.guard_completeness:
            _raise_on_code(check_guard_completeness())
            summary["guard_completeness"] = True
        if args.all or args.live_fast:
            check_live_fast()
            summary["live_fast"] = True
        if args.live:
            check_live()
            summary["live"] = True
    except SystemExit as e:
        print(
            json.dumps({"ok": False, "summary": summary, "exit": e.code}),
            file=sys.stderr,
        )
        raise

    print(json.dumps({"ok": True, "summary": summary}))


if __name__ == "__main__":
    main()
