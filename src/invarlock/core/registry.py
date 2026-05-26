"""
InvarLock Core Registry
===================

Unified plugin registry using entry point discovery.
Provides centralized access to adapters, edits, and guards.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import (
    EntryPoint,
    PackageNotFoundError,
    entry_points,
)
from importlib.metadata import (
    version as metadata_version,
)
from typing import Any, cast

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.runtime_security import third_party_plugins_allowed

from .abi import INVARLOCK_CORE_ABI
from .api import Guard, ModelAdapter, ModelEdit
from .builtin_plugin_catalog import builtin_plugin_specs
from .exceptions import DependencyError, PluginError

__all__ = ["PluginInfo", "CoreRegistry", "get_registry"]

_DISCOVERY_ERRORS = (AttributeError, ImportError, RuntimeError, TypeError, ValueError)
_PLUGIN_LOAD_ERRORS = (
    AttributeError,
    DependencyError,
    ImportError,
    PluginError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class PluginInfo:
    """Plugin information from entry points."""

    name: str
    module: str
    class_name: str
    available: bool
    status: str
    package: str | None = None
    version: str | None = None
    entry_point: Any | None = None
    support_tier: str = "third_party"
    strict_assurance_allowed: bool = False
    published_basis: bool = False
    deployment_claim: bool = False


def _select_entry_points(eps: Any, group: str) -> list[EntryPoint]:
    """Return entry points for a given group across importlib versions."""

    selected: Iterable[EntryPoint]
    if hasattr(eps, "select"):
        selected = cast("Iterable[EntryPoint]", eps.select(group=group))
    else:
        selected = cast("Iterable[EntryPoint]", eps.get(group, []))
    return list(selected)


class CoreRegistry:
    """
    Central registry for InvarLock plugins using entry point discovery.

    Discovers and manages adapters, edits, and guards through
    setuptools entry points without requiring imports.
    """

    def __init__(self):
        self._adapters: dict[str, PluginInfo] = {}
        self._edits: dict[str, PluginInfo] = {}
        self._guards: dict[str, PluginInfo] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of plugin discovery."""
        if self._initialized:
            return

        self._discover_plugins()
        self._initialized = True

    def _discover_plugins(self) -> None:
        """Register built-ins and discover third-party plugins."""
        self._register_builtin_plugins()
        if not third_party_plugins_allowed():
            return

        try:
            eps = entry_points()
            self._register_entry_points(
                self._adapters,
                _select_entry_points(eps, "invarlock.adapters"),
                "adapters",
            )
            self._register_entry_points(
                self._edits,
                _select_entry_points(eps, "invarlock.edits"),
                "edits",
            )
            self._register_entry_points(
                self._guards,
                _select_entry_points(eps, "invarlock.guards"),
                "guards",
            )
        except _DISCOVERY_ERRORS as error:
            raise RuntimeError(f"Plugin discovery failed: {error}") from error

    def _register_builtin_plugins(self) -> None:
        """Register the shipped plugin surface explicitly."""

        def _register_builtin(
            registry: dict[str, PluginInfo],
            name: str,
            module: str,
            class_name: str,
            required_deps: list[str] | None = None,
        ) -> None:
            if name in registry:
                raise RuntimeError(f"Duplicate built-in plugin registration: {name}")
            actual_available = True
            actual_status = "Built-in"
            if required_deps:
                missing = self._check_runtime_dependencies(required_deps)
                if missing:
                    actual_available = False
                    actual_status = f"Needs extra: {', '.join(missing)}"

            registry[name] = PluginInfo(
                name=name,
                module=module,
                class_name=class_name,
                available=actual_available,
                status=actual_status,
                package="invarlock",
                version=INVARLOCK_VERSION,
            )

        registries = {
            "adapters": self._adapters,
            "edits": self._edits,
            "guards": self._guards,
        }
        for plugin_type, registry in registries.items():
            for spec in builtin_plugin_specs(plugin_type):
                _register_builtin(
                    registry,
                    spec.name,
                    spec.module,
                    spec.class_name,
                    required_deps=list(spec.required_deps) or None,
                )
                registry[spec.name].support_tier = spec.support_tier
                registry[
                    spec.name
                ].strict_assurance_allowed = spec.strict_assurance_allowed
                registry[spec.name].published_basis = spec.published_basis
                registry[spec.name].deployment_claim = spec.deployment_claim

    def _register_entry_points(
        self,
        registry: dict[str, PluginInfo],
        entry_points_for_group: list[EntryPoint],
        plugin_type: str,
    ) -> None:
        for entry_point in entry_points_for_group:
            if entry_point.name in registry:
                raise RuntimeError(
                    f"Duplicate {plugin_type.rstrip('s')} plugin name: {entry_point.name}"
                )
            registry[entry_point.name] = self._create_plugin_info(
                entry_point, plugin_type
            )

    def _check_runtime_dependencies(self, deps: list[str]) -> list[str]:
        """
        Check if runtime dependencies are actually present on the system.

        Uses importlib.util.find_spec to avoid importing packages and triggering
        heavy side effects (e.g., GPU-only extensions).

        Returns:
            List of missing dependency names.
        """
        missing: list[str] = []
        for dep in deps:
            try:
                spec = importlib.util.find_spec(dep)
            except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
                spec = None
            if spec is None:
                missing.append(dep)
        return missing

    def _parse_entry_point_value(self, entry_point: EntryPoint) -> tuple[str, str]:
        value = getattr(entry_point, "value", None)
        if not isinstance(value, str):
            raise TypeError("entry point value must be a string")
        module_path, separator, class_name = value.partition(":")
        if not module_path or separator != ":" or not class_name:
            raise ValueError(f"malformed entry point value: {value}")
        return module_path, class_name

    def _create_plugin_info(
        self, entry_point: EntryPoint, plugin_type: str
    ) -> PluginInfo:
        """Create plugin info from entry point."""
        _ = plugin_type
        module_path, class_name = self._parse_entry_point_value(entry_point)

        # Determine package/version metadata
        package_name: str | None = None
        version: str | None = None

        dist = getattr(entry_point, "dist", None)
        if dist is not None:
            metadata = getattr(dist, "metadata", None)
            if isinstance(metadata, dict):
                meta_name = metadata.get("Name")
                if isinstance(meta_name, str) and meta_name:
                    package_name = meta_name
            if not package_name:
                dist_name = getattr(dist, "name", None)
                if isinstance(dist_name, str) and dist_name:
                    package_name = dist_name
            dist_version = getattr(dist, "version", None)
            if isinstance(dist_version, str) and dist_version:
                version = dist_version

        if not package_name:
            package_name = module_path.split(".")[0]
            try:
                version = metadata_version(package_name)
            except PackageNotFoundError:
                version = None

        # Defer import to instantiation time to avoid heavy imports here
        return PluginInfo(
            name=entry_point.name,
            module=module_path,
            class_name=class_name,
            available=True,
            status="Deferred load",
            package=package_name,
            version=version,
            entry_point=entry_point,
        )

    def _resolve_plugin_class(self, info: PluginInfo) -> Any:
        if info.entry_point:
            return info.entry_point.load()
        module = importlib.import_module(info.module)
        return getattr(module, info.class_name)

    def _validate_plugin_abi(self, cls: Any) -> None:
        provider_mod = importlib.import_module(cls.__module__)
        plugin_abi = getattr(provider_mod, "INVARLOCK_CORE_ABI", None)
        if not isinstance(plugin_abi, str) or not plugin_abi.strip():
            raise ImportError(
                "ABI missing: plugin must declare "
                f"INVARLOCK_CORE_ABI={INVARLOCK_CORE_ABI}"
            )
        if plugin_abi != INVARLOCK_CORE_ABI:
            raise ImportError(
                f"ABI mismatch: plugin={plugin_abi} != core={INVARLOCK_CORE_ABI}"
            )

    def _instantiate_plugin(
        self,
        info: PluginInfo,
        *,
        expected_type: type[Any],
        kind: str,
    ) -> Any:
        try:
            cls = self._resolve_plugin_class(info)
            self._validate_plugin_abi(cls)
            instance = cls()
        except _PLUGIN_LOAD_ERRORS as error:
            raise ImportError(
                f"Failed to load {kind} '{info.name}': {error}"
            ) from error
        if not isinstance(instance, expected_type):
            raise ImportError(
                f"Failed to load {kind} '{info.name}': "
                f"Expected {expected_type.__name__}, got {type(instance)}"
            )
        return instance

    def list_adapters(self) -> list[str]:
        """List all registered adapter names."""
        self._ensure_initialized()
        return list(self._adapters.keys())

    def list_edits(self) -> list[str]:
        """List all registered edit names."""
        self._ensure_initialized()
        return list(self._edits.keys())

    def list_guards(self) -> list[str]:
        """List all registered guard names."""
        self._ensure_initialized()
        return list(self._guards.keys())

    def get_adapter(self, name: str) -> ModelAdapter:
        """Get an adapter instance by name."""
        self._ensure_initialized()

        if name not in self._adapters:
            available = list(self._adapters.keys())
            raise KeyError(f"Unknown adapter '{name}'. Available: {available}")

        info = self._adapters[name]
        if not info.available:
            raise ImportError(f"Adapter '{name}' unavailable: {info.status}")

        return cast(
            ModelAdapter,
            self._instantiate_plugin(
                info,
                expected_type=ModelAdapter,
                kind="adapter",
            ),
        )

    def get_edit(self, name: str) -> ModelEdit:
        """Get an edit instance by name."""
        self._ensure_initialized()

        if name not in self._edits:
            available = list(self._edits.keys())
            raise KeyError(f"Unknown edit '{name}'. Available: {available}")

        info = self._edits[name]
        if not info.available:
            raise ImportError(f"Edit '{name}' unavailable: {info.status}")

        return cast(
            ModelEdit,
            self._instantiate_plugin(
                info,
                expected_type=ModelEdit,
                kind="edit",
            ),
        )

    def get_guard(self, name: str) -> Guard:
        """Get a guard instance by name."""
        self._ensure_initialized()

        if name not in self._guards:
            available = list(self._guards.keys())
            raise KeyError(f"Unknown guard '{name}'. Available: {available}")

        info = self._guards[name]
        if not info.available:
            raise ImportError(f"Guard '{name}' unavailable: {info.status}")

        return cast(
            Guard,
            self._instantiate_plugin(
                info,
                expected_type=Guard,
                kind="guard",
            ),
        )

    def get_plugin_info(self, name: str, plugin_type: str) -> dict[str, Any]:
        """Get plugin information without instantiation."""
        self._ensure_initialized()

        if plugin_type == "adapters":
            registry = self._adapters
            entry_group = "invarlock.adapters"
        elif plugin_type == "edits":
            registry = self._edits
            entry_group = "invarlock.edits"
        elif plugin_type == "guards":
            registry = self._guards
            entry_group = "invarlock.guards"
        else:
            raise ValueError(f"Unknown plugin type: {plugin_type}")

        if name not in registry:
            return {"available": False, "status": "Not found", "module": "unknown"}

        info = registry[name]
        return {
            "available": info.available,
            "status": info.status,
            "module": info.module,
            "package": info.package,
            "version": info.version,
            "entry_point": info.entry_point.name if info.entry_point else None,
            "entry_point_group": entry_group if info.entry_point else None,
            "support_tier": info.support_tier,
            "strict_assurance_allowed": info.strict_assurance_allowed,
            "published_basis": info.published_basis,
            "deployment_claim": info.deployment_claim,
        }

    def get_plugin_metadata(self, name: str, plugin_type: str) -> dict[str, Any]:
        """Return comprehensive metadata for a plugin."""
        metadata = self.get_plugin_info(name, plugin_type)
        if metadata.get("module") == "unknown":
            raise KeyError(f"Unknown {plugin_type.rstrip('s')} plugin '{name}'")

        metadata.update(
            {
                "name": name,
                "type": plugin_type,
            }
        )
        return metadata

    # Typed-error wrappers that preserve existing behavior for existing methods
    def get_adapter_typed(self, name: str) -> ModelAdapter:
        try:
            return self.get_adapter(name)
        except (ImportError, KeyError) as e:  # pragma: no cover - exercised in tests
            details = {"name": name, "kind": "adapter", "reason": type(e).__name__}
            if isinstance(e, ImportError | ModuleNotFoundError):
                raise DependencyError(
                    code="E702", message="PLUGIN-DEPENDENCY-MISSING", details=details
                ) from e
            raise PluginError(
                code="E701", message="PLUGIN-LOAD-FAILED", details=details
            ) from e

    def get_edit_typed(self, name: str) -> ModelEdit:
        try:
            return self.get_edit(name)
        except (ImportError, KeyError) as e:  # pragma: no cover - exercised in tests
            details = {"name": name, "kind": "edit", "reason": type(e).__name__}
            if isinstance(e, ImportError | ModuleNotFoundError):
                raise DependencyError(
                    code="E702", message="PLUGIN-DEPENDENCY-MISSING", details=details
                ) from e
            raise PluginError(
                code="E701", message="PLUGIN-LOAD-FAILED", details=details
            ) from e

    def get_guard_typed(self, name: str) -> Guard:
        try:
            return self.get_guard(name)
        except (ImportError, KeyError) as e:  # pragma: no cover - exercised in tests
            details = {"name": name, "kind": "guard", "reason": type(e).__name__}
            if isinstance(e, ImportError | ModuleNotFoundError):
                raise DependencyError(
                    code="E702", message="PLUGIN-DEPENDENCY-MISSING", details=details
                ) from e
            raise PluginError(
                code="E701", message="PLUGIN-LOAD-FAILED", details=details
            ) from e

    def validate_configuration(
        self, adapter_name: str, edit_name: str, guard_names: list[str]
    ) -> tuple[bool, str]:
        """Validate that a configuration is available."""
        self._ensure_initialized()

        issues = []
        # Check adapter
        if adapter_name not in self._adapters:
            issues.append(f"Unknown adapter: {adapter_name}")
        elif not self._adapters[adapter_name].available:
            issues.append(f"Adapter unavailable: {adapter_name}")

        # Check edit
        if edit_name not in self._edits:
            issues.append(f"Unknown edit: {edit_name}")
        elif not self._edits[edit_name].available:
            issues.append(f"Edit unavailable: {edit_name}")

        # Check guards
        for guard_name in guard_names:
            if guard_name == "noop":
                continue  # noop is always available
            if guard_name not in self._guards:
                issues.append(f"Unknown guard: {guard_name}")
            elif not self._guards[guard_name].available:
                issues.append(f"Guard unavailable: {guard_name}")

        if issues:
            return False, "; ".join(issues)

        return True, "Configuration is valid"


# Global registry instance
_global_registry = CoreRegistry()


def get_registry() -> CoreRegistry:
    """Get the global plugin registry instance."""
    return _global_registry
