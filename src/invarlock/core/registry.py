"""
InvarLock Core Registry
===================

Unified plugin registry using entry point discovery.
Provides centralized access to adapters, edits, and guards.
"""

from __future__ import annotations

import importlib
import warnings
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

from .abi import INVARLOCK_CORE_ABI
from .api import Guard, ModelAdapter, ModelEdit
from .exceptions import DependencyError, PluginError

__all__ = ["PluginInfo", "CoreRegistry", "get_registry"]


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
        """Discover all plugins through entry points with fallback registration."""
        # Try entry points first
        try:
            eps = entry_points()

            # Discover adapters
            adapter_eps = _select_entry_points(eps, "invarlock.adapters")
            for ep in adapter_eps:
                info = self._create_plugin_info(ep, "adapters")
                self._adapters[ep.name] = info

            # Discover edits
            edit_eps = _select_entry_points(eps, "invarlock.edits")
            for ep in edit_eps:
                info = self._create_plugin_info(ep, "edits")
                self._edits[ep.name] = info

            # Discover guards
            guard_eps = _select_entry_points(eps, "invarlock.guards")
            for ep in guard_eps:
                info = self._create_plugin_info(ep, "guards")
                self._guards[ep.name] = info

        except Exception as e:
            warnings.warn(f"Plugin discovery failed: {e}", stacklevel=2)

        # Fallback registration for development
        self._register_fallback_plugins()

    def _register_fallback_plugins(self) -> None:
        """Register fallback plugins when entry points are not available."""

        def _fallback(
            registry: dict[str, PluginInfo],
            name: str,
            module: str,
            class_name: str,
            status: str = "Available (fallback)",
            required_deps: list[str] | None = None,
        ) -> None:
            if name not in registry:
                # Check runtime dependencies for optional plugins
                actual_available = True
                actual_status = status
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

        # Register built-in adapters
        _fallback(self._adapters, "hf_gpt2", "invarlock.adapters", "HF_GPT2_Adapter")
        _fallback(self._adapters, "hf_bert", "invarlock.adapters", "HF_BERT_Adapter")
        _fallback(self._adapters, "hf_llama", "invarlock.adapters", "HF_LLaMA_Adapter")
        _fallback(self._adapters, "hf_t5", "invarlock.adapters", "HF_T5_Adapter")
        _fallback(
            self._adapters, "hf_onnx", "invarlock.adapters", "HF_ORT_CausalLM_Adapter"
        )
        # Convenience auto adapters (delegate to built-ins)
        _fallback(
            self._adapters,
            "hf_causal_auto",
            "invarlock.adapters",
            "HF_Causal_Auto_Adapter",
        )
        _fallback(
            self._adapters, "hf_mlm_auto", "invarlock.adapters", "HF_MLM_Auto_Adapter"
        )
        # Optional plugin adapters (verify runtime dependencies)
        _fallback(
            self._adapters,
            "hf_gptq",
            "invarlock.plugins.hf_gptq_adapter",
            "HF_GPTQ_Adapter",
            status="Available (plugin)",
            required_deps=["auto_gptq"],
        )
        _fallback(
            self._adapters,
            "hf_awq",
            "invarlock.plugins.hf_awq_adapter",
            "HF_AWQ_Adapter",
            status="Available (plugin)",
            required_deps=["autoawq"],
        )
        _fallback(
            self._adapters,
            "hf_bnb",
            "invarlock.plugins.hf_bnb_adapter",
            "HF_BNB_Adapter",
            status="Available (plugin)",
            required_deps=["bitsandbytes"],
        )

        # Register built-in edits (quant-only core) and internal no-op
        _fallback(self._edits, "quant_rtn", "invarlock.edits", "RTNQuantEdit")
        _fallback(self._edits, "noop", "invarlock.edits.noop", "NoopEdit")

        # Register built-in guards
        _fallback(self._guards, "invariants", "invarlock.guards", "InvariantsGuard")
        _fallback(self._guards, "spectral", "invarlock.guards", "SpectralGuard")
        _fallback(self._guards, "variance", "invarlock.guards", "VarianceGuard")
        _fallback(self._guards, "rmt", "invarlock.guards", "RMTGuard")
        _fallback(self._guards, "hello_guard", "invarlock.plugins", "HelloGuard")

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
                spec = importlib.util.find_spec(dep)  # type: ignore[attr-defined]
            except Exception:
                spec = None
            if spec is None:
                missing.append(dep)
        return missing

    def _create_plugin_info(
        self, entry_point: EntryPoint, plugin_type: str
    ) -> PluginInfo:
        """Create plugin info from entry point."""
        try:
            # Parse module and class from entry point value
            module_path, class_name = entry_point.value.split(":")

            # Determine package/version metadata
            package_name: str | None = None
            version: str | None = None

            dist = getattr(entry_point, "dist", None)
            if dist is not None:
                package_name = getattr(dist, "metadata", {}).get("Name") or getattr(
                    dist, "name", None
                )
                version = getattr(dist, "version", None)

            if not package_name:
                package_name = module_path.split(".")[0]
                try:
                    version = metadata_version(package_name)
                except PackageNotFoundError:
                    version = None

            # Defer import to instantiation time to avoid heavy imports here
            available = True
            status = "Deferred load"

            return PluginInfo(
                name=entry_point.name,
                module=module_path,
                class_name=class_name,
                available=available,
                status=status,
                package=package_name,
                version=version,
                entry_point=entry_point,
            )

        except Exception as e:
            return PluginInfo(
                name=entry_point.name,
                module="unknown",
                class_name="unknown",
                available=False,
                status=f"Parse error: {e}",
                entry_point=entry_point,
            )

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

        try:
            if info.entry_point:
                cls = info.entry_point.load()
            else:
                # Fallback loading
                module = importlib.import_module(info.module)
                cls = getattr(module, info.class_name)
            # ABI compatibility check on the providing module
            try:  # pragma: no cover - simple guard
                provider_mod = importlib.import_module(cls.__module__)
                plugin_abi = getattr(provider_mod, "INVARLOCK_CORE_ABI", None)
                if plugin_abi is not None and str(plugin_abi) != INVARLOCK_CORE_ABI:
                    raise ImportError(
                        f"ABI mismatch: plugin={plugin_abi} != core={INVARLOCK_CORE_ABI}"
                    )
            except Exception as abi_exc:
                raise ImportError(str(abi_exc)) from abi_exc
            instance = cls()
            if not isinstance(instance, ModelAdapter):
                raise TypeError(f"Expected ModelAdapter, got {type(instance)}")
            return instance
        except Exception as e:
            raise ImportError(f"Failed to load adapter '{name}': {e}") from e

    def get_edit(self, name: str) -> ModelEdit:
        """Get an edit instance by name."""
        self._ensure_initialized()

        if name not in self._edits:
            available = list(self._edits.keys())
            raise KeyError(f"Unknown edit '{name}'. Available: {available}")

        info = self._edits[name]
        if not info.available:
            raise ImportError(f"Edit '{name}' unavailable: {info.status}")

        try:
            if info.entry_point:
                cls = info.entry_point.load()
            else:
                # Fallback loading
                module = importlib.import_module(info.module)
                cls = getattr(module, info.class_name)
            try:  # ABI check
                provider_mod = importlib.import_module(cls.__module__)
                plugin_abi = getattr(provider_mod, "INVARLOCK_CORE_ABI", None)
                if plugin_abi is not None and str(plugin_abi) != INVARLOCK_CORE_ABI:
                    raise ImportError(
                        f"ABI mismatch: plugin={plugin_abi} != core={INVARLOCK_CORE_ABI}"
                    )
            except Exception as abi_exc:
                raise ImportError(str(abi_exc)) from abi_exc
            instance = cls()
            if not isinstance(instance, ModelEdit):
                raise TypeError(f"Expected ModelEdit, got {type(instance)}")
            return instance
        except Exception as e:
            raise ImportError(f"Failed to load edit '{name}': {e}") from e

    def get_guard(self, name: str) -> Guard:
        """Get a guard instance by name."""
        self._ensure_initialized()

        if name not in self._guards:
            available = list(self._guards.keys())
            raise KeyError(f"Unknown guard '{name}'. Available: {available}")

        info = self._guards[name]
        if not info.available:
            raise ImportError(f"Guard '{name}' unavailable: {info.status}")

        try:
            if info.entry_point:
                cls = info.entry_point.load()
            else:
                # Fallback loading
                module = importlib.import_module(info.module)
                cls = getattr(module, info.class_name)
            try:  # ABI check
                provider_mod = importlib.import_module(cls.__module__)
                plugin_abi = getattr(provider_mod, "INVARLOCK_CORE_ABI", None)
                if plugin_abi is not None and str(plugin_abi) != INVARLOCK_CORE_ABI:
                    raise ImportError(
                        f"ABI mismatch: plugin={plugin_abi} != core={INVARLOCK_CORE_ABI}"
                    )
            except Exception as abi_exc:
                raise ImportError(str(abi_exc)) from abi_exc
            instance = cls()
            if not isinstance(instance, Guard):
                raise TypeError(f"Expected Guard, got {type(instance)}")
            return instance
        except Exception as e:
            raise ImportError(f"Failed to load guard '{name}': {e}") from e

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

    # Typed-error wrappers that preserve existing behavior for legacy methods
    def get_adapter_typed(self, name: str) -> ModelAdapter:
        try:
            return self.get_adapter(name)
        except Exception as e:  # pragma: no cover - exercised in tests
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
        except Exception as e:  # pragma: no cover - exercised in tests
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
        except Exception as e:  # pragma: no cover - exercised in tests
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
