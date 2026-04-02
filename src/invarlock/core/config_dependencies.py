from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_INCLUDE_MAX_DEPTH = 16


@dataclass(frozen=True)
class ConfigDependencyScan:
    config_paths: tuple[Path, ...]
    referenced_paths: tuple[Path, ...]


def allow_config_include_outside() -> bool:
    allow_outside = os.environ.get("INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE", "")
    return allow_outside.strip().lower() in {"1", "true", "yes", "on"}


def iter_absolute_path_strings(payload: Any) -> set[Path]:
    paths: set[Path] = set()
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return paths
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            paths.add(absolute_path_no_resolve(candidate))
        return paths
    if isinstance(payload, dict):
        for value in payload.values():
            paths.update(iter_absolute_path_strings(value))
        return paths
    if isinstance(payload, (list, tuple, set)):
        for item in payload:
            paths.update(iter_absolute_path_strings(item))
    return paths


def absolute_path_no_resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return Path(os.path.abspath(str(candidate)))
    return Path(os.path.abspath(str(Path.cwd() / candidate)))


def create_config_loader(
    base_dir: Path,
    *,
    include_stack: tuple[Path, ...] = (),
    max_include_depth: int = CONFIG_INCLUDE_MAX_DEPTH,
    dependency_paths: set[Path] | None = None,
):
    class Loader(yaml.SafeLoader):
        pass

    Loader._base_dir = Path(base_dir).resolve()
    Loader._include_stack = tuple(Path(p).resolve() for p in include_stack)
    Loader._max_include_depth = int(max_include_depth)
    Loader._dependency_paths = dependency_paths

    def _construct_include(loader: yaml.SafeLoader, node: yaml.Node):
        rel = loader.construct_scalar(node)
        path = (loader._base_dir / rel).resolve()
        active_stack = tuple(getattr(loader, "_include_stack", ()))
        active_max_depth = int(
            getattr(loader, "_max_include_depth", CONFIG_INCLUDE_MAX_DEPTH)
        )

        if path in active_stack:
            chain = " -> ".join(str(p) for p in (*active_stack, path))
            raise ValueError(f"Config !include cycle detected: {chain}")
        if len(active_stack) >= active_max_depth:
            chain = " -> ".join(str(p) for p in (*active_stack, path))
            raise ValueError(
                f"Config !include depth exceeds {active_max_depth}: {chain}"
            )

        if not allow_config_include_outside():
            try:
                path.relative_to(loader._base_dir)
            except ValueError as exc:
                raise ValueError(
                    "Config !include must stay within the config directory. "
                    "Set INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE=1 to override."
                ) from exc
        tracked_paths = getattr(loader, "_dependency_paths", None)
        if isinstance(tracked_paths, set):
            tracked_paths.add(path)
        with path.open(encoding="utf-8") as fh:
            inc_loader = create_config_loader(
                path.parent,
                include_stack=(*active_stack, path),
                max_include_depth=active_max_depth,
                dependency_paths=tracked_paths,
            )
            return yaml.load(fh, Loader=inc_loader)

    Loader.add_constructor("!include", _construct_include)
    return Loader


def load_raw_config_payload(
    path: str | Path,
    *,
    dependency_paths: set[Path] | None = None,
) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Configuration file not found: {candidate}")
    resolved = absolute_path_no_resolve(candidate)
    if isinstance(dependency_paths, set):
        dependency_paths.add(resolved)
    loader = create_config_loader(
        resolved.parent,
        dependency_paths=dependency_paths,
    )
    with resolved.open(encoding="utf-8") as fh:
        raw = yaml.load(fh, Loader=loader)
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping")
    return raw


def inspect_config_dependencies(path: str | Path) -> ConfigDependencyScan:
    dependency_paths: set[Path] = set()
    raw = load_raw_config_payload(path, dependency_paths=dependency_paths)
    referenced_paths = iter_absolute_path_strings(raw)
    return ConfigDependencyScan(
        config_paths=tuple(
            sorted(
                {Path(item).resolve(strict=False) for item in dependency_paths},
                key=str,
            )
        ),
        referenced_paths=tuple(
            sorted(
                {absolute_path_no_resolve(item) for item in referenced_paths},
                key=str,
            )
        ),
    )


__all__ = [
    "CONFIG_INCLUDE_MAX_DEPTH",
    "ConfigDependencyScan",
    "absolute_path_no_resolve",
    "allow_config_include_outside",
    "create_config_loader",
    "inspect_config_dependencies",
    "iter_absolute_path_strings",
    "load_raw_config_payload",
]
