from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INVENTORY = Path("scripts/scripts_inventory.toml")
LARGE_MAINTAINER_SCRIPT_LINE_THRESHOLD = 1000
IGNORED_PATH_PATTERNS = (
    "scripts/.DS_Store",
    "scripts/**/.DS_Store",
    "scripts/**/.coverage/**",
    "scripts/**/.mypy_cache/**",
    "scripts/**/.pytest_cache/**",
    "scripts/**/.ruff_cache/**",
    "scripts/**/__pycache__/**",
    "scripts/**/*.pyc",
)
IGNORED_REFERENCE_PATH_PATTERNS = (
    ".github/**/.DS_Store",
    "docs/**/.DS_Store",
    "tests/**/.DS_Store",
    "tests/**/.coverage/**",
    "tests/**/.mypy_cache/**",
    "tests/**/.pytest_cache/**",
    "tests/**/.ruff_cache/**",
    "tests/**/__pycache__/**",
    "tests/**/*.pyc",
)


@dataclass(frozen=True)
class ScriptFamily:
    name: str
    owner: str
    purpose: str
    stability: str
    audience: str
    expected_runtime: str
    network: str
    gpu: str
    invoked_by: tuple[str, ...]
    paths: tuple[str, ...]


@dataclass(frozen=True)
class LargeScriptReview:
    path: str
    max_lines: int
    rationale: str


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _path_is_ignored(rel_path: str) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in IGNORED_PATH_PATTERNS)


def _load_inventory(
    path: Path,
) -> tuple[list[ScriptFamily], dict[str, LargeScriptReview], list[str]]:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [], {}, [f"inventory unreadable: {path}: {exc}"]
    except tomllib.TOMLDecodeError as exc:
        return [], {}, [f"inventory invalid TOML: {path}: {exc}"]

    errors: list[str] = []
    if raw.get("version") != 1:
        errors.append("inventory version must be 1")

    family_entries = raw.get("families")
    if not isinstance(family_entries, list):
        errors.append("inventory must define a [[families]] array")
        return [], {}, errors

    families: list[ScriptFamily] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(family_entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"family #{index} must be a table")
            continue
        name = entry.get("name")
        owner = entry.get("owner")
        description = entry.get("description")
        stability = entry.get("stability")
        audience = entry.get("audience")
        expected_runtime = entry.get("expected_runtime")
        network = entry.get("network")
        gpu = entry.get("gpu")
        invoked_by = entry.get("invoked_by")
        paths = entry.get("paths")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"family #{index} has an invalid name")
            continue
        metadata = {
            "owner": owner,
            "description": description,
            "stability": stability,
            "audience": audience,
            "expected_runtime": expected_runtime,
            "network": network,
            "gpu": gpu,
        }
        for key, value in metadata.items():
            if not isinstance(value, str) or not value.strip():
                errors.append(f"family {name!r} has invalid {key}")
        if not isinstance(invoked_by, list) or not invoked_by:
            errors.append(f"family {name!r} must define non-empty invoked_by")
            invoked_by_values: tuple[str, ...] = ()
        else:
            invoked_by_values = tuple(str(item) for item in invoked_by if item)
        if name in seen_names:
            errors.append(f"family {name!r} is duplicated")
        seen_names.add(name)
        if not isinstance(paths, list) or not paths:
            errors.append(f"family {name!r} must define non-empty paths")
            continue
        normalized_paths: list[str] = []
        for pattern in paths:
            if not isinstance(pattern, str) or not pattern:
                errors.append(f"family {name!r} contains an invalid path pattern")
                continue
            if pattern.startswith("/") or ".." in Path(pattern).parts:
                errors.append(f"family {name!r} has unsafe path pattern {pattern!r}")
                continue
            if not pattern.startswith("scripts/"):
                errors.append(
                    f"family {name!r} path must start with scripts/: {pattern}"
                )
                continue
            normalized_paths.append(pattern)
        if normalized_paths:
            families.append(
                ScriptFamily(
                    name=name,
                    owner=str(owner or ""),
                    purpose=str(description or ""),
                    stability=str(stability or ""),
                    audience=str(audience or ""),
                    expected_runtime=str(expected_runtime or ""),
                    network=str(network or ""),
                    gpu=str(gpu or ""),
                    invoked_by=invoked_by_values,
                    paths=tuple(normalized_paths),
                )
            )
    large_reviews: dict[str, LargeScriptReview] = {}
    review_entries = raw.get("large_script_reviews", [])
    if review_entries is None:
        review_entries = []
    if not isinstance(review_entries, list):
        errors.append("large_script_reviews must be a [[large_script_reviews]] array")
    else:
        for index, entry in enumerate(review_entries, start=1):
            if not isinstance(entry, dict):
                errors.append(f"large_script_reviews #{index} must be a table")
                continue
            review_path = entry.get("path")
            max_lines = entry.get("max_lines")
            rationale = entry.get("rationale")
            if not isinstance(review_path, str) or not review_path.startswith(
                "scripts/"
            ):
                errors.append(
                    f"large_script_reviews #{index} has invalid scripts/ path"
                )
                continue
            if review_path.startswith("/") or ".." in Path(review_path).parts:
                errors.append(
                    f"large_script_reviews #{index} has unsafe path {review_path!r}"
                )
                continue
            if not isinstance(max_lines, int) or max_lines < (
                LARGE_MAINTAINER_SCRIPT_LINE_THRESHOLD + 1
            ):
                errors.append(
                    f"large_script_reviews {review_path!r} must set max_lines above "
                    f"{LARGE_MAINTAINER_SCRIPT_LINE_THRESHOLD}"
                )
                continue
            if not isinstance(rationale, str) or not rationale.strip():
                errors.append(
                    f"large_script_reviews {review_path!r} must include rationale"
                )
                continue
            if review_path in large_reviews:
                errors.append(f"large_script_reviews {review_path!r} is duplicated")
                continue
            large_reviews[review_path] = LargeScriptReview(
                path=review_path,
                max_lines=max_lines,
                rationale=rationale,
            )
    return families, large_reviews, errors


def _script_files(root: Path) -> list[str]:
    scripts_root = root / "scripts"
    if not scripts_root.is_dir():
        return []
    files: list[str] = []
    for path in scripts_root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = _normalize_rel(path, root)
        if _path_is_ignored(rel_path):
            continue
        files.append(rel_path)
    return sorted(files)


def _matching_families(rel_path: str, families: list[ScriptFamily]) -> list[str]:
    matches: list[str] = []
    for family in families:
        if any(fnmatch.fnmatch(rel_path, pattern) for pattern in family.paths):
            matches.append(family.name)
    return matches


def _reference_index(root: Path) -> dict[str, str]:
    search_roots = [
        root / "Makefile",
        root / ".github" / "workflows",
        root / "docs",
        root / "scripts",
        root / "tests",
    ]
    index: dict[str, str] = {}
    for candidate in search_roots:
        if candidate.is_file():
            paths = [candidate]
        elif candidate.is_dir():
            paths = [path for path in candidate.rglob("*") if path.is_file()]
        else:
            paths = []
        for path in paths:
            try:
                rel = _normalize_rel(path, root)
                if rel == DEFAULT_INVENTORY.as_posix():
                    continue
                if _path_is_ignored(rel) or any(
                    fnmatch.fnmatch(rel, pattern)
                    for pattern in IGNORED_REFERENCE_PATH_PATTERNS
                ):
                    continue
                index[rel] = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
    return index


def _script_rel_if_file(root: Path, path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        rel = _normalize_rel(path, root)
    except ValueError:
        return None
    if not rel.startswith("scripts/") or _path_is_ignored(rel):
        return None
    return rel


def _resolve_python_module(
    *,
    root: Path,
    importer: Path,
    module: str | None,
    level: int,
    aliases: tuple[str, ...] = (),
) -> set[str]:
    module_parts = tuple(part for part in (module or "").split(".") if part)
    base_dirs: list[Path] = []

    if level > 0:
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        base_dirs.append(base)
    elif module_parts[:1] == ("scripts",):
        base_dirs.append(root)
    else:
        scripts_root = root / "scripts"
        for parent in (importer.parent, *importer.parent.parents):
            if parent == root:
                break
            if parent == scripts_root or scripts_root in parent.parents:
                base_dirs.append(parent)
        base_dirs.append(scripts_root)

    candidates: set[str] = set()
    seen_bases: set[Path] = set()
    for base in base_dirs:
        if base in seen_bases:
            continue
        seen_bases.add(base)
        module_path = base.joinpath(*module_parts)
        for path in (module_path.with_suffix(".py"), module_path / "__init__.py"):
            rel = _script_rel_if_file(root, path)
            if rel is not None:
                candidates.add(rel)
        for alias in aliases:
            alias_path = module_path / f"{alias}.py"
            rel = _script_rel_if_file(root, alias_path)
            if rel is not None:
                candidates.add(rel)

    if not candidates and len(module_parts) == 1:
        module_filename = f"{module_parts[0]}.py"
        matches = [
            _normalize_rel(path, root)
            for path in (root / "scripts").rglob(module_filename)
            if path.is_file()
        ]
        if len(matches) == 1:
            candidates.add(matches[0])
    return candidates


def _import_reference_index(root: Path, index: dict[str, str]) -> dict[str, list[str]]:
    imports: dict[str, set[str]] = defaultdict(set)
    for rel_path, text in index.items():
        if not rel_path.endswith(".py"):
            continue
        source_path = root / rel_path
        try:
            tree = ast.parse(text, filename=rel_path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets: set[str] = set()
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets.update(
                        _resolve_python_module(
                            root=root,
                            importer=source_path,
                            module=alias.name,
                            level=0,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                alias_names = tuple(alias.name for alias in node.names)
                targets.update(
                    _resolve_python_module(
                        root=root,
                        importer=source_path,
                        module=node.module,
                        level=node.level,
                        aliases=alias_names,
                    )
                )
            for target in targets:
                if target != rel_path:
                    imports[target].add(rel_path)
    return {path: sorted(sources) for path, sources in imports.items()}


def _referenced_by(
    rel_path: str,
    index: dict[str, str],
    import_index: dict[str, list[str]],
) -> list[str]:
    referenced = {
        path for path, text in index.items() if path != rel_path and rel_path in text
    }
    referenced.update(import_index.get(rel_path, []))
    return sorted(referenced)


def _line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError:
        return 0


def _requires_large_script_review(family: ScriptFamily, line_count: int) -> bool:
    if line_count <= LARGE_MAINTAINER_SCRIPT_LINE_THRESHOLD:
        return False
    return family.stability == "maintainer-workflow" or family.audience == "maintainer"


def build_audit_payload(
    *,
    root: Path,
    inventory_path: Path,
) -> tuple[dict[str, object], list[str]]:
    families, large_reviews, errors = _load_inventory(inventory_path)
    files = _script_files(root)
    family_by_name = {family.name: family for family in families}
    refs = _reference_index(root)
    import_refs = _import_reference_index(root, refs)
    file_rows: list[dict[str, object]] = []
    for rel_path in files:
        matches = _matching_families(rel_path, families)
        if len(matches) != 1:
            continue
        family = family_by_name[matches[0]]
        path = root / rel_path
        line_count = _line_count(path)
        size_review_required = _requires_large_script_review(family, line_count)
        review = large_reviews.get(rel_path)
        referenced = _referenced_by(rel_path, refs, import_refs)
        file_rows.append(
            {
                "path": rel_path,
                "family": family.name,
                "owner": family.owner,
                "purpose": family.purpose,
                "stability": family.stability,
                "audience": family.audience,
                "expected_runtime": family.expected_runtime,
                "network": family.network,
                "gpu": family.gpu,
                "invoked_by": list(family.invoked_by),
                "referenced_by": referenced,
                "referenced": bool(referenced),
                "executable": path.stat().st_mode & 0o111 != 0,
                "line_count": line_count,
                "size_review_required": size_review_required,
                "size_reviewed": review is not None,
            }
        )
    return (
        {
            "format_version": "scripts-audit-v1",
            "files": file_rows,
            "unreferenced": [
                row["path"] for row in file_rows if not bool(row["referenced"])
            ],
        },
        errors,
    )


def check_inventory(
    *,
    root: Path,
    inventory_path: Path,
) -> tuple[bool, list[str], dict[str, int]]:
    families, large_reviews, errors = _load_inventory(inventory_path)
    files = _script_files(root)
    counts: dict[str, int] = defaultdict(int)
    matched_patterns: set[str] = set()
    observed_review_paths: set[str] = set()
    family_by_name = {family.name: family for family in families}

    for rel_path in files:
        matches = _matching_families(rel_path, families)
        if not matches:
            errors.append(f"unclassified script file: {rel_path}")
            continue
        if len(matches) > 1:
            errors.append(
                f"script file matches multiple families: {rel_path} -> "
                + ", ".join(matches)
            )
            continue
        family = family_by_name[matches[0]]
        counts[family.name] += 1
        line_count = _line_count(root / rel_path)
        review = large_reviews.get(rel_path)
        if review is not None:
            observed_review_paths.add(rel_path)
        if _requires_large_script_review(family, line_count):
            if review is None:
                errors.append(
                    f"large maintainer script lacks size review: {rel_path} "
                    f"({line_count} lines, threshold "
                    f"{LARGE_MAINTAINER_SCRIPT_LINE_THRESHOLD})"
                )
            elif line_count > review.max_lines:
                errors.append(
                    f"large maintainer script exceeds reviewed max_lines: {rel_path} "
                    f"({line_count} > {review.max_lines})"
                )
        for pattern in family.paths:
            if fnmatch.fnmatch(rel_path, pattern):
                matched_patterns.add(pattern)

    for rel_path in sorted(set(large_reviews) - observed_review_paths):
        errors.append(f"large script review references missing file: {rel_path}")

    for family in families:
        for pattern in family.paths:
            if pattern not in matched_patterns:
                errors.append(
                    f"inventory pattern matched no files: {family.name}: {pattern}"
                )

    return not errors, errors, dict(sorted(counts.items()))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that every scripts/ file is assigned to a maintained family."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_repo_root_from_script(),
        help="Repository root. Defaults to the current checkout.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Inventory TOML path. Defaults to scripts/scripts_inventory.toml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a per-file audit payload with inherited ownership metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.root.resolve()
    inventory_path = args.inventory
    if inventory_path is None:
        inventory_path = root / DEFAULT_INVENTORY
    elif not inventory_path.is_absolute():
        inventory_path = root / inventory_path

    if args.json:
        payload, errors = build_audit_payload(root=root, inventory_path=inventory_path)
        ok, check_errors, _counts = check_inventory(
            root=root, inventory_path=inventory_path
        )
        errors.extend(check_errors)
        print(json.dumps(payload, sort_keys=True))
        return 0 if ok and not errors else 1

    ok, errors, counts = check_inventory(root=root, inventory_path=inventory_path)
    if not ok:
        print("[check_scripts_inventory] FAIL", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    total = sum(counts.values())
    print(f"[check_scripts_inventory] OK ({total} files across {len(counts)} families)")
    for name, count in counts.items():
        print(f"  - {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
