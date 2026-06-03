#!/usr/bin/env python3
"""Check repository config integrity.

This script verifies that:
- All `defaults: !include <path>` targets resolve to files.
- YAML config files parse cleanly.
- All adapter names referenced in configs are present in the plugin registry.
- CI-critical preset and edit surfaces still exist.

Exit code is non-zero on any failure.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_ROOT = REPO_ROOT / "configs"
RUNTIME_CONFIG_ROOT = REPO_ROOT / "src" / "invarlock" / "_data" / "runtime"

REQUIRED_CI_CONFIGS = (
    Path("configs/presets/causal_lm/wikitext2_512.yaml"),
    Path("configs/overlays/edits/quant_rtn/8bit_attn.yaml"),
)
REQUIRED_EDIT_CLASSES = {
    "quant_rtn": (
        REPO_ROOT / "src" / "invarlock" / "edits" / "quant_rtn.py",
        "RTNQuantEdit",
    ),
}

INCLUDE_RE = re.compile(r"^\s*defaults:\s*!include\s+(\S+)\s*$")


class ConfigLoader(yaml.SafeLoader):
    """Loader that tolerates repo-local `!include` tags during linting."""


def _include_constructor(loader: ConfigLoader, node: yaml.Node) -> dict[str, str]:
    return {"__include__": loader.construct_scalar(node)}


ConfigLoader.add_constructor("!include", _include_constructor)


def _repo_rel(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _iter_yaml_files(roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix in {".yaml", ".yml"}:
            files.add(root.resolve())
        elif root.is_dir():
            files.update(
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and path.suffix in {".yaml", ".yml"}
            )
    return sorted(files)


def _find_missing_includes(path: Path) -> list[tuple[Path, str, Path]]:
    missing: list[tuple[Path, str, Path]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = INCLUDE_RE.match(line)
        if not match:
            continue
        include = match.group(1)
        target = (path.parent / include).resolve()
        if not target.exists():
            missing.append((path, include, target))
    return missing


def _collect_adapter_names(value: Any) -> set[str]:
    adapters: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "adapter" and isinstance(item, str):
                adapters.add(item)
            adapters.update(_collect_adapter_names(item))
    elif isinstance(value, list):
        for item in value:
            adapters.update(_collect_adapter_names(item))
    return adapters


def find_includes_adapters_and_yaml_errors(
    roots: Iterable[Path],
) -> tuple[list[tuple[Path, str, Path]], set[str], list[str]]:
    missing_includes: list[tuple[Path, str, Path]] = []
    adapters: set[str] = set()
    yaml_errors: list[str] = []

    for path in _iter_yaml_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            yaml_errors.append(f"{_repo_rel(path)}: unable to read: {exc}")
            continue
        missing_includes.extend(_find_missing_includes(path))
        try:
            loaded = yaml.load(text, Loader=ConfigLoader)
        except yaml.YAMLError as exc:
            yaml_errors.append(f"{_repo_rel(path)}: invalid YAML: {exc}")
            continue
        adapters.update(_collect_adapter_names(loaded))

    return missing_includes, adapters, yaml_errors


def find_includes_and_adapters(
    root: Path,
) -> tuple[list[tuple[Path, str, Path]], set[str]]:
    missing, adapters, _errors = find_includes_adapters_and_yaml_errors([root])
    return missing, adapters


def registry_adapters() -> tuple[set[str], str | None]:
    src_root = REPO_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    try:
        from invarlock.core.registry import get_registry
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        return set(), f"unable to import plugin registry: {exc}"
    reg = get_registry()
    return set(reg.list_adapters()), None


def _module_defines_class(path: Path, class_name: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body
    )


def _check_ci_matrix_surfaces() -> list[str]:
    errors: list[str] = []
    for rel_path in REQUIRED_CI_CONFIGS:
        if not (REPO_ROOT / rel_path).is_file():
            errors.append(f"missing CI config surface: {rel_path.as_posix()}")

    for name, (path, class_name) in REQUIRED_EDIT_CLASSES.items():
        if not path.is_file():
            errors.append(f"missing edit module for {name}: {_repo_rel(path)}")
        elif not _module_defines_class(path, class_name):
            errors.append(f"missing edit class for {name}: {class_name}")
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_CONFIG_ROOT],
        help="Config roots or YAML files to audit. Defaults to configs/.",
    )
    parser.add_argument(
        "--ci-matrix",
        action="store_true",
        help="Also verify the CI-critical preset/edit surfaces and runtime YAML.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    roots = [path if path.is_absolute() else REPO_ROOT / path for path in args.roots]
    if args.ci_matrix and RUNTIME_CONFIG_ROOT.is_dir():
        roots.append(RUNTIME_CONFIG_ROOT)

    missing_includes, adapters, yaml_errors = find_includes_adapters_and_yaml_errors(
        roots
    )
    rc = 0

    print("YAML syntax:")
    if yaml_errors:
        for error in yaml_errors:
            print(f"  MISS {error}")
        rc = 1
    else:
        print("  All YAML files parse correctly.")

    print("Include targets:")
    if not missing_includes:
        print("  All defaults includes resolve correctly.")
    else:
        for p, inc, tgt in missing_includes:
            print(f"  MISS {_repo_rel(p)} -> {inc} (resolved {tgt})")
        rc = 1

    print("\nAdapters referenced:")
    for a in sorted(adapters):
        print(f"  {a}")
    reg, registry_error = registry_adapters()
    print("\nAdapter registry availability:")
    if registry_error is not None:
        print(f"  MISS {registry_error}")
        rc = 1
    for a in sorted(adapters):
        ok = a in reg
        print(f"  {'OK  ' if ok else 'MISS'} {a}")
        if not ok:
            rc = 1

    if args.ci_matrix:
        print("\nCI matrix surfaces:")
        ci_errors = _check_ci_matrix_surfaces()
        if ci_errors:
            for error in ci_errors:
                print(f"  MISS {error}")
            rc = 1
        else:
            print("  Required CI preset/edit surfaces are present.")

    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
