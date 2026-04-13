"""Runtime config loading and profile helpers for InvarLock."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from importlib import resources as _ires
from pathlib import Path
from typing import Any

import yaml

from .config_runtime import InvarLockConfig, VarianceGuardConfig

_RUNTIME_RESOURCE_ERRORS = (
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
)
CONFIG_INCLUDE_MAX_DEPTH = 16


@dataclass(frozen=True)
class ConfigDependencyScan:
    config_paths: tuple[Path, ...]
    referenced_paths: tuple[Path, ...]


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def allow_config_include_outside() -> bool:
    allow_outside = os.environ.get("INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE", "")
    return allow_outside.strip().lower() in {"1", "true", "yes", "on"}


def absolute_path_no_resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return Path(os.path.abspath(str(candidate)))
    return Path(os.path.abspath(str(Path.cwd() / candidate)))


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


def _load_raw_config_payload(
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
    raw = _load_raw_config_payload(path, dependency_paths=dependency_paths)
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


def _load_runtime_yaml(*rel_parts: str) -> dict[str, Any] | None:
    """Load YAML from the runtime config locations."""
    root = os.getenv("INVARLOCK_CONFIG_ROOT")
    if root:
        p = Path(root) / "runtime"
        for part in rel_parts:
            p = p / part
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
                if not isinstance(data, dict):
                    raise ValueError("Runtime YAML must be a mapping")
                return data

    try:
        base = _ires.files("invarlock._data.runtime")
        res = base
        for part in rel_parts:
            res = res.joinpath(part)
        try:
            is_file = getattr(res, "is_file", None)
            read_text = getattr(res, "read_text", None)
            if callable(is_file) and is_file() and callable(read_text):
                text = read_text(encoding="utf-8")
                data = yaml.safe_load(text) or {}
                if not isinstance(data, dict):
                    return None
                return data
        except FileNotFoundError:
            pass
    except _RUNTIME_RESOURCE_ERRORS:
        pass
    return None


def load_config(path: str | Path) -> InvarLockConfig:
    raw = _load_raw_config_payload(path)
    defaults = raw.pop("defaults", None)
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping when present")
    if isinstance(defaults, dict):
        raw = _deep_merge(defaults, raw)

    edit_block = raw.get("edit")
    if isinstance(edit_block, dict) and "parameters" in edit_block:
        raise ValueError("edit.parameters is not supported; use edit.plan.")
    if isinstance(edit_block, dict) and "kind" in edit_block:
        raise ValueError(
            "edit.kind is not supported; use edit.name with a canonical edit plugin name."
        )

    if raw.get("assurance") is not None:
        raise ValueError(
            "assurance.* is not supported; configure measurement contracts under guards.* "
            "(e.g., guards.spectral.estimator, guards.rmt.activation.sampling)."
        )

    guards_block = raw.get("guards")
    if isinstance(guards_block, dict):
        for guard_name in ("spectral", "rmt"):
            node = guards_block.get(guard_name)
            if isinstance(node, dict) and "mode" in node:
                raise ValueError(
                    f"guards.{guard_name}.mode is not supported; remove it and configure "
                    "measurement-contract knobs under guard policy fields instead."
                )

    guards = raw.get("guards")
    if isinstance(guards, dict):
        var = guards.get("variance")
        if isinstance(var, dict):
            vkw = {
                k: var.get(k)
                for k in [
                    "clamp",
                    "mode",
                    "deadband",
                    "min_gain",
                    "min_rel_gain",
                    "min_abs_adjust",
                    "max_scale_step",
                    "min_effect_lognll",
                    "predictive_one_sided",
                    "topk_backstop",
                    "max_adjusted_modules",
                    "predictive_gate",
                    "target_modules",
                    "scope",
                    "calibration",
                    "absolute_floor_ppl",
                ]
            }
            if vkw.get("mode") is None:
                vkw["mode"] = "ci"
            guards["variance"] = VarianceGuardConfig(
                **{k: v for k, v in vkw.items() if v is not None}
            )
    return InvarLockConfig(raw)


def load_tiers() -> dict[str, Any]:
    data = _load_runtime_yaml("tiers.yaml")
    if data is not None:
        return data
    raise FileNotFoundError(
        "tiers.yaml not found in package runtime (and no INVARLOCK_CONFIG_ROOT override)"
    )


def apply_profile(cfg: InvarLockConfig, profile: str) -> InvarLockConfig:
    overrides: dict[str, Any] | None = _load_runtime_yaml("profiles", f"{profile}.yaml")
    if overrides is None:
        raise ValueError(f"Unknown profile: {profile}")
    base_cfg = cfg.model_dump()
    merged = _deep_merge(base_cfg, overrides)

    base_primary_metric = (
        base_cfg.get("primary_metric")
        if isinstance(base_cfg.get("primary_metric"), dict)
        else {}
    )
    merged_primary_metric = (
        merged.get("primary_metric")
        if isinstance(merged.get("primary_metric"), dict)
        else {}
    )
    if base_primary_metric and merged_primary_metric is not None:
        for key, value in base_primary_metric.items():
            merged_primary_metric[key] = copy.deepcopy(value)
        merged["primary_metric"] = merged_primary_metric

    return InvarLockConfig(merged)


__all__ = [
    "CONFIG_INCLUDE_MAX_DEPTH",
    "ConfigDependencyScan",
    "absolute_path_no_resolve",
    "allow_config_include_outside",
    "apply_profile",
    "create_config_loader",
    "inspect_config_dependencies",
    "iter_absolute_path_strings",
    "load_config",
    "load_tiers",
]
