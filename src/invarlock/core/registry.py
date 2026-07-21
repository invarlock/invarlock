"""Runtime-integration discovery for the paired evaluation engine."""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, PackageNotFoundError, entry_points
from importlib.metadata import version as metadata_version
from pathlib import Path
from typing import Any, cast

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.runtime_security import third_party_plugins_allowed

from .builtin_plugin_catalog import builtin_plugin_specs

__all__ = ["CoreRegistry", "PluginInfo", "get_registry"]

_DISCOVERY_ERRORS = (AttributeError, ImportError, RuntimeError, TypeError, ValueError)
_LOAD_ERRORS = (AttributeError, ImportError, RuntimeError, TypeError, ValueError)
_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_FIRST_PARTY_RUNTIME_ADDINS = {
    "hf_vision_text": (
        "invarlock-runtime-hf-vision-text",
        "invarlock_addins.multimodal.provider:HFVisionTextProvider",
    ),
    "llama_cpp": (
        "invarlock-runtime-gguf",
        "invarlock_addins.gguf.provider:LlamaCppProvider",
    ),
    "tensorrt_llm": (
        "invarlock-runtime-tensorrt-llm",
        "invarlock_addins.tensorrt_llm.provider:TensorRTLLMProvider",
    ),
}
_QUALIFICATION_CANDIDATE_SITE = "INVARLOCK_QUALIFICATION_CANDIDATE_SITE"


@dataclass
class PluginInfo:
    """Deferred runtime-provider entry-point metadata."""

    name: str
    module: str
    class_name: str
    required_deps: tuple[str, ...]
    available: bool
    package: str | None = None
    version: str | None = None
    entry_point: Any | None = None


def _select_entry_points(eps: Any) -> list[EntryPoint]:
    selected: Iterable[EntryPoint]
    if hasattr(eps, "select"):
        selected = cast(
            "Iterable[EntryPoint]",
            eps.select(group="invarlock.runtime_providers"),
        )
    else:
        selected = cast(
            "Iterable[EntryPoint]",
            eps.get("invarlock.runtime_providers", []),
        )
    return list(selected)


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _qualification_candidate_site() -> Path | None:
    value = os.environ.get(_QUALIFICATION_CANDIDATE_SITE)
    if value is None:
        return None
    lexical = Path(os.path.abspath(value))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("Qualification candidate site is unavailable") from exc
    if resolved != lexical or not resolved.is_dir():
        raise RuntimeError("Qualification candidate site is invalid")
    return resolved


def _entry_point_is_from(
    entry_point: EntryPoint,
    *,
    candidate_site: Path,
) -> bool:
    dist = getattr(entry_point, "dist", None)
    locate_file = getattr(dist, "locate_file", None)
    if not callable(locate_file):
        return False
    try:
        location = Path(locate_file("")).resolve(strict=True)
    except (OSError, TypeError, ValueError):
        return False
    return location == candidate_site or location.is_relative_to(candidate_site)


def _is_shipped_entry_point(entry_point: EntryPoint) -> bool:
    dist = getattr(entry_point, "dist", None)
    dist_name = getattr(dist, "name", None)
    dist_version = getattr(dist, "version", None)
    if (
        not isinstance(dist_name, str)
        or _normalized_distribution_name(dist_name) != "invarlock"
        or dist_version != INVARLOCK_VERSION
    ):
        return False
    value = getattr(entry_point, "value", None)
    return any(
        entry_point.name == spec.name and value == f"{spec.module}:{spec.class_name}"
        for spec in builtin_plugin_specs("runtime_providers")
    )


def _is_approved_first_party_addin(entry_point: EntryPoint) -> bool:
    expected = _FIRST_PARTY_RUNTIME_ADDINS.get(entry_point.name)
    if expected is None:
        return False
    dist = getattr(entry_point, "dist", None)
    dist_name = getattr(dist, "name", None)
    dist_version = getattr(dist, "version", None)
    value = getattr(entry_point, "value", None)
    expected_distribution, expected_value = expected
    if (
        not isinstance(dist_name, str)
        or _normalized_distribution_name(dist_name) != expected_distribution
        or dist_version != INVARLOCK_VERSION
        or value != expected_value
    ):
        raise RuntimeError(
            "Invalid first-party runtime add-in entry point: "
            f"{entry_point.name!r} must come from {expected_distribution!r} "
            f"at version {INVARLOCK_VERSION!r} and resolve to {expected_value!r}"
        )
    return True


class CoreRegistry:
    """Discover and instantiate runtime providers without importing backends."""

    def __init__(self) -> None:
        self._runtime_providers: dict[str, PluginInfo] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self._discover_plugins()
            self._initialized = True

    def _discover_plugins(self) -> None:
        self._register_builtins()
        try:
            candidates = _select_entry_points(entry_points())
            allow_third_party = third_party_plugins_allowed()
            candidate_site = _qualification_candidate_site()
            for entry_point in candidates:
                if (
                    candidate_site is not None
                    and entry_point.name in _FIRST_PARTY_RUNTIME_ADDINS
                    and not _entry_point_is_from(
                        entry_point,
                        candidate_site=candidate_site,
                    )
                ):
                    continue
                first_party = _is_approved_first_party_addin(entry_point)
                if first_party or allow_third_party:
                    self._register_entry_point(entry_point)
        except _DISCOVERY_ERRORS as error:
            raise RuntimeError(f"Runtime-provider discovery failed: {error}") from error

    def _register_builtins(self) -> None:
        for spec in builtin_plugin_specs("runtime_providers"):
            if spec.name in self._runtime_providers:
                raise RuntimeError(f"Duplicate built-in runtime provider: {spec.name}")
            missing = self._missing_dependencies(spec.required_deps)
            self._runtime_providers[spec.name] = PluginInfo(
                name=spec.name,
                module=spec.module,
                class_name=spec.class_name,
                required_deps=spec.required_deps,
                available=not missing,
                package="invarlock",
                version=INVARLOCK_VERSION,
            )

    @staticmethod
    def _missing_dependencies(dependencies: tuple[str, ...]) -> list[str]:
        missing: list[str] = []
        for dependency in dependencies:
            try:
                spec = importlib.util.find_spec(dependency)
            except _DISCOVERY_ERRORS:
                spec = None
            if spec is None:
                missing.append(dependency)
        return missing

    @staticmethod
    def _parse_entry_point(entry_point: EntryPoint) -> tuple[str, str]:
        value = getattr(entry_point, "value", None)
        if not isinstance(value, str):
            raise TypeError("runtime-provider entry point value must be a string")
        module, separator, class_name = value.partition(":")
        if not module or separator != ":" or not class_name:
            raise ValueError(f"malformed runtime-provider entry point: {value}")
        return module, class_name

    def _register_entry_point(self, entry_point: EntryPoint) -> None:
        if _NAME_RE.fullmatch(entry_point.name) is None:
            raise RuntimeError(
                f"Invalid runtime provider plugin name: {entry_point.name!r}"
            )
        if entry_point.name in self._runtime_providers:
            if _is_shipped_entry_point(entry_point):
                return
            raise RuntimeError(f"Duplicate runtime provider name: {entry_point.name}")
        module, class_name = self._parse_entry_point(entry_point)
        package: str | None = None
        version: str | None = None
        dist = getattr(entry_point, "dist", None)
        if dist is not None:
            dist_name = getattr(dist, "name", None)
            if isinstance(dist_name, str) and dist_name:
                package = dist_name
            dist_version = getattr(dist, "version", None)
            if isinstance(dist_version, str) and dist_version:
                version = dist_version
        if package is None:
            package = module.split(".")[0]
            try:
                version = metadata_version(package)
            except PackageNotFoundError:
                version = None
        self._runtime_providers[entry_point.name] = PluginInfo(
            name=entry_point.name,
            module=module,
            class_name=class_name,
            required_deps=(),
            available=True,
            package=package,
            version=version,
            entry_point=entry_point,
        )

    @staticmethod
    def _resolve_provider_class(info: PluginInfo) -> Any:
        if info.entry_point:
            return info.entry_point.load()
        module = importlib.import_module(info.module)
        return getattr(module, info.class_name)

    def _instantiate(self, info: PluginInfo) -> Any:
        from .runtime_provider import INVARLOCK_RUNTIME_PROVIDER_ABI, RuntimeProvider

        try:
            provider_class = self._resolve_provider_class(info)
            provider_module = importlib.import_module(provider_class.__module__)
            declared_abi = getattr(
                provider_module,
                "INVARLOCK_RUNTIME_PROVIDER_ABI",
                None,
            )
            if declared_abi != INVARLOCK_RUNTIME_PROVIDER_ABI:
                raise ImportError(
                    "ABI mismatch: runtime provider="
                    f"{declared_abi!r} != core={INVARLOCK_RUNTIME_PROVIDER_ABI}"
                )
            instance = provider_class()
        except _LOAD_ERRORS as error:
            raise ImportError(
                f"Failed to load runtime provider '{info.name}': {error}"
            ) from error
        if not isinstance(instance, RuntimeProvider):
            raise ImportError(
                f"Failed to load runtime provider '{info.name}': "
                f"Expected RuntimeProvider, got {type(instance)}"
            )
        if instance.name != info.name:
            raise ImportError(
                f"Failed to load runtime provider '{info.name}': "
                f"provider identity mismatch ({instance.name!r})"
            )
        if instance.abi_version != INVARLOCK_RUNTIME_PROVIDER_ABI:
            raise ImportError(
                f"Failed to load runtime provider '{info.name}': "
                f"instance ABI {instance.abi_version!r} does not match "
                f"{INVARLOCK_RUNTIME_PROVIDER_ABI!r}"
            )
        return instance

    def list_runtime_providers(self) -> list[str]:
        """List provider names without importing their implementations."""

        self._ensure_initialized()
        return list(self._runtime_providers)

    def get_runtime_provider(self, name: str) -> Any:
        """Instantiate one provider after exact identity and ABI validation."""

        self._ensure_initialized()
        info = self._runtime_providers.get(name)
        if info is None:
            raise KeyError(
                f"Unknown runtime provider {name!r}. "
                f"Available: {list(self._runtime_providers)}"
            )
        if not info.available:
            dependencies = ", ".join(info.required_deps) or "unspecified"
            raise ImportError(
                f"Runtime provider {name!r} unavailable; "
                f"required dependencies: {dependencies}"
            )
        return self._instantiate(info)

    def get_plugin_info(self, name: str, plugin_type: str) -> dict[str, Any]:
        """Return runtime-provider metadata without importing its backend."""

        if plugin_type != "runtime_providers":
            raise ValueError(f"Unknown plugin type: {plugin_type}")
        self._ensure_initialized()
        info = self._runtime_providers.get(name)
        if info is None:
            return {
                "name": name,
                "module": None,
                "class_name": None,
                "required_deps": (),
                "available": False,
                "package": None,
                "version": None,
                "entry_point": None,
            }
        return {
            "name": info.name,
            "module": info.module,
            "class_name": info.class_name,
            "required_deps": info.required_deps,
            "available": info.available,
            "package": info.package,
            "version": info.version,
            "entry_point": info.entry_point.name if info.entry_point else None,
        }


_global_registry = CoreRegistry()


def get_registry() -> CoreRegistry:
    """Return the process-global lazy runtime-provider registry."""

    return _global_registry
