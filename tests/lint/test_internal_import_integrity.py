from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "invarlock"


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _internal_modules() -> set[str]:
    return {_module_name(path) for path in PACKAGE_ROOT.rglob("*.py")}


def _resolved_from_import(
    node: ast.ImportFrom,
    *,
    source_module: str,
    source_is_package: bool,
) -> str | None:
    if node.level == 0:
        return node.module

    package = source_module if source_is_package else source_module.rpartition(".")[0]
    relative_name = "." * node.level + (node.module or "")
    try:
        return importlib.util.resolve_name(relative_name, package)
    except ImportError:
        return None


def test_all_internal_source_import_targets_exist() -> None:
    modules = _internal_modules()
    offenders: list[str] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source_module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets: list[str | None] = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                targets.append(
                    _resolved_from_import(
                        node,
                        source_module=source_module,
                        source_is_package=path.name == "__init__.py",
                    )
                )

            for target in targets:
                if (
                    target is not None
                    and target.startswith("invarlock.")
                    and target not in modules
                ):
                    relative = path.relative_to(REPO_ROOT)
                    offenders.append(f"{relative}:{node.lineno} -> {target}")

    assert sorted(set(offenders)) == []
