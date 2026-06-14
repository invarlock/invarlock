from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"

EXPECTED_EXECUTABLE_DIRS = {
    "adapters",
    "calibration",
    "ci",
    "cli",
    "core",
    "docs",
    "edits",
    "eval",
    "guards",
    "integration",
    "lint",
    "observability",
    "plugins",
    "evidence_packs",
    "reporting",
    "runtime",
    "scripts",
}

EXPECTED_SUPPORT_DIRS = {"_stubs", "artifacts", "fixtures", "schemas"}
DEPRECATED_DIRS = {
    "api",
    "packaging",
    "security",
    "unit",
    "utils",
    "guards_property",
    "guards_differential",
}
TRANSITIONAL_TOKENS = (
    "additional",
    "extra",
    "more",
    "cases",
    "edgecases",
    "split",
    "tail",
    "part2",
)


def _tracked_test_files() -> list[Path]:
    return sorted(
        path for path in TESTS_ROOT.rglob("*.py") if "__pycache__" not in path.parts
    )


def test_top_level_test_dirs_match_owner_contract() -> None:
    actual = {
        path.name
        for path in TESTS_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    assert actual == EXPECTED_EXECUTABLE_DIRS | EXPECTED_SUPPORT_DIRS


def test_deprecated_test_dirs_are_absent() -> None:
    for dirname in DEPRECATED_DIRS:
        assert not (TESTS_ROOT / dirname).exists(), dirname


def test_eval_does_not_host_reporting_modules() -> None:
    eval_root = TESTS_ROOT / "eval"
    assert not list(eval_root.glob("test_report*.py"))


def test_helpers_are_owner_local() -> None:
    legacy_utils_import = "tests" + ".utils"
    for path in _tracked_test_files():
        text = path.read_text(encoding="utf-8")
        assert legacy_utils_import not in text, path.as_posix()


def test_transitional_test_names_are_absent() -> None:
    offenders: list[str] = []
    for path in _tracked_test_files():
        name = path.name
        if not name.startswith("test_"):
            continue
        stem = path.stem
        if stem.endswith("regression_matrix"):
            continue
        segments = stem.removeprefix("test_").split("_")
        if any(token in segments for token in TRANSITIONAL_TOKENS):
            offenders.append(path.as_posix())
    assert offenders == []


def test_no_test_file_exceeds_size_guideline() -> None:
    offenders = [
        f"{path.as_posix()}:{len(path.read_text(encoding='utf-8').splitlines())}"
        for path in _tracked_test_files()
        if len(path.read_text(encoding="utf-8").splitlines()) > 800
    ]
    assert offenders == []


def _is_test_module_ref(module_name: str) -> bool:
    return ".test_" in module_name or module_name.rsplit(".", 1)[-1].startswith("test_")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return ""


def test_test_modules_do_not_import_other_test_modules() -> None:
    offenders: list[str] = []
    for path in _tracked_test_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and _is_test_module_ref(node.module):
                    offenders.append(f"{path.as_posix()}:{node.lineno}:{node.module}")
                for alias in node.names:
                    if _is_test_module_ref(alias.name):
                        offenders.append(
                            f"{path.as_posix()}:{node.lineno}:{alias.name}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_test_module_ref(alias.name):
                        offenders.append(
                            f"{path.as_posix()}:{node.lineno}:{alias.name}"
                        )
            elif isinstance(node, ast.Call):
                call_name = _call_name(node.func)
                if call_name not in {
                    "__import__",
                    "importlib.import_module",
                    "importlib.util.spec_from_file_location",
                }:
                    continue
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    continue
                imported_module = node.args[0].value
                if isinstance(imported_module, str) and _is_test_module_ref(
                    imported_module
                ):
                    offenders.append(
                        f"{path.as_posix()}:{node.lineno}:{imported_module}"
                    )
                if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
                    continue
                imported_path = node.args[1].value
                if isinstance(imported_path, str) and _is_test_module_ref(
                    Path(imported_path).stem
                ):
                    offenders.append(f"{path.as_posix()}:{node.lineno}:{imported_path}")
    assert offenders == []
