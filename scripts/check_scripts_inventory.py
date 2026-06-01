from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INVENTORY = Path("scripts/scripts_inventory.toml")
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


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _path_is_ignored(rel_path: str) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in IGNORED_PATH_PATTERNS)


def _load_inventory(path: Path) -> tuple[list[ScriptFamily], list[str]]:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [], [f"inventory unreadable: {path}: {exc}"]
    except tomllib.TOMLDecodeError as exc:
        return [], [f"inventory invalid TOML: {path}: {exc}"]

    errors: list[str] = []
    if raw.get("version") != 1:
        errors.append("inventory version must be 1")

    family_entries = raw.get("families")
    if not isinstance(family_entries, list):
        errors.append("inventory must define a [[families]] array")
        return [], errors

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
    return families, errors


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


def _referenced_by(rel_path: str, index: dict[str, str]) -> list[str]:
    return sorted(
        path for path, text in index.items() if path != rel_path and rel_path in text
    )


def build_audit_payload(
    *,
    root: Path,
    inventory_path: Path,
) -> tuple[dict[str, object], list[str]]:
    families, errors = _load_inventory(inventory_path)
    files = _script_files(root)
    family_by_name = {family.name: family for family in families}
    refs = _reference_index(root)
    file_rows: list[dict[str, object]] = []
    for rel_path in files:
        matches = _matching_families(rel_path, families)
        if len(matches) != 1:
            continue
        family = family_by_name[matches[0]]
        referenced = _referenced_by(rel_path, refs)
        path = root / rel_path
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
    families, errors = _load_inventory(inventory_path)
    files = _script_files(root)
    counts: dict[str, int] = defaultdict(int)
    matched_patterns: set[str] = set()

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
        counts[matches[0]] += 1
        for family in families:
            if family.name != matches[0]:
                continue
            for pattern in family.paths:
                if fnmatch.fnmatch(rel_path, pattern):
                    matched_patterns.add(pattern)

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
