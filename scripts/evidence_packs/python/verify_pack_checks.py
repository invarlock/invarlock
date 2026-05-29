from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CONTROL_FILES = {
    "checksums.sha256",
    "manifest.json",
    "manifest.signature.json",
    "metadata/manifest.json",
    "metadata/manifest.signature.json",
    "metadata/checksums.sha256",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_pack_path(value: str) -> str:
    path = value.strip()
    if path.startswith("*"):
        path = path[1:]
    if path.startswith("./"):
        path = path[2:]
    return path


def _checksum_paths(path: Path) -> set[str]:
    paths: set[str] = set()
    if not path.is_file():
        return paths
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 1:
            paths.add(_normalize_pack_path(parts[0]))
            continue
        paths.add(_normalize_pack_path(parts[1]))
    return {path for path in paths if path}


def _is_transport_artifact(rel_path: str) -> bool:
    return (
        rel_path == ".DS_Store"
        or rel_path.endswith("/.DS_Store")
        or rel_path.startswith("._")
        or "/._" in rel_path
        or rel_path.startswith("__MACOSX/")
    )


def _actual_pack_files(pack_dir: Path) -> set[str]:
    files: set[str] = set()
    for path in pack_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(pack_dir).as_posix()
        if _is_transport_artifact(rel_path):
            continue
        files.add(rel_path)
    return files


def cmd_manifest_field(args: argparse.Namespace) -> int:
    payload = _load_json(args.manifest)
    if not isinstance(payload, dict):
        return 1
    value = payload.get(args.field)
    if value is None:
        return 1
    if isinstance(value, str):
        print(value)
    else:
        print(str(value))
    return 0


def cmd_path_within(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    candidate = args.candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return 1
    return 0


def cmd_scenario_strictness(args: argparse.Namespace) -> int:
    payload = _load_json(args.scenarios)
    if not isinstance(payload, dict):
        return 1
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return 1
    for scenario in scenarios:
        if not isinstance(scenario, dict) or scenario.get("id") != args.scenario_id:
            continue
        strictness = scenario.get("strictness")
        if isinstance(strictness, str) and strictness:
            print(strictness)
            return 0
    return 1


def cmd_extra_files(args: argparse.Namespace) -> int:
    pack_dir = args.pack_dir.resolve()
    expected = _checksum_paths(pack_dir / "checksums.sha256") | CONTROL_FILES
    actual = _actual_pack_files(pack_dir)
    extras = sorted(actual - expected)
    if not extras:
        return 0

    prefix = "ERROR" if args.strict else "WARNING"
    print(
        f"{prefix}: Pack contains extra files not covered by checksums.sha256:",
        file=sys.stderr,
    )
    for rel_path in extras:
        print(f"  - {rel_path}", file=sys.stderr)
    return 1 if args.strict else 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured JSON/path checks for evidence-pack shell entrypoints."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_field = subparsers.add_parser("manifest-field")
    manifest_field.add_argument("manifest", type=Path)
    manifest_field.add_argument("field")
    manifest_field.set_defaults(func=cmd_manifest_field)

    path_within = subparsers.add_parser("path-within")
    path_within.add_argument("root", type=Path)
    path_within.add_argument("candidate", type=Path)
    path_within.set_defaults(func=cmd_path_within)

    scenario_strictness = subparsers.add_parser("scenario-strictness")
    scenario_strictness.add_argument("scenarios", type=Path)
    scenario_strictness.add_argument("scenario_id")
    scenario_strictness.set_defaults(func=cmd_scenario_strictness)

    extra_files = subparsers.add_parser("extra-files")
    extra_files.add_argument("pack_dir", type=Path)
    extra_files.add_argument("--strict", action="store_true")
    extra_files.set_defaults(func=cmd_extra_files)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
