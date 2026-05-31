"""Built-in edit implementations and edit plugin registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from invarlock.core import INVARLOCK_CORE_ABI
from invarlock.core.api import EditRuntime, ModelAdapter, ModelEdit

from .quant_rtn import RTNQuantEdit


class NoopEdit(ModelEdit):
    """A do-nothing edit that returns empty deltas."""

    name = "noop"

    def can_edit(self, model_desc: dict[str, Any]) -> bool:
        return True

    def preview(
        self, model: nn.Module, adapter: ModelAdapter, calib: Any
    ) -> dict[str, Any]:
        return {"name": self.name, "plan": {}}

    def apply(
        self,
        model: nn.Module,
        adapter: ModelAdapter,
        plan: dict[str, Any] | None = None,
        runtime: EditRuntime | None = None,
    ) -> dict[str, Any]:
        _ = plan, runtime
        return {
            "name": self.name,
            "plan_digest": "noop",
            "plan": {},
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
            "config": {},
            "model_desc": adapter.describe(model)
            if hasattr(adapter, "describe")
            else {},
        }


@dataclass
class EditPlugin:
    """Plugin metadata for a model edit."""

    name: str
    edit_class: type[Any]
    description: str
    is_available: bool = True
    dependencies: list[str] | None = None

    def __post_init__(self) -> None:
        if self.dependencies is None:
            self.dependencies = []


class EditRegistry:
    """Registry for model edit plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, EditPlugin] = {}
        self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Discover and register built-in edit plugins."""
        self.register_plugin(
            EditPlugin(
                name="quant_rtn",
                edit_class=RTNQuantEdit,
                description="RTN dequantized weight-edit simulation",
                is_available=True,
            )
        )
        self.register_plugin(
            EditPlugin(
                name="noop",
                edit_class=NoopEdit,
                description="No-op edit for baseline and calibration runs",
                is_available=True,
            )
        )

    def register_plugin(self, plugin: EditPlugin) -> None:
        """Register an edit plugin."""
        self._plugins[plugin.name] = plugin

    def get_plugin(self, name: str) -> EditPlugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_available(self) -> list[str]:
        """List all available edit names."""
        return [name for name, plugin in self._plugins.items() if plugin.is_available]

    def get_available_edits(self) -> dict[str, EditPlugin]:
        """Get all available edit plugins."""
        return {
            name: plugin
            for name, plugin in self._plugins.items()
            if plugin.is_available
        }


_registry: EditRegistry | None = None


def get_registry() -> EditRegistry:
    """Get the global edit registry."""
    global _registry
    if _registry is None:
        _registry = EditRegistry()
    return _registry


def register_edit(
    name: str,
    edit_class: type[Any],
    description: str = "",
    dependencies: list[str] | None = None,
) -> None:
    """Register an edit plugin."""
    registry = get_registry()
    plugin = EditPlugin(
        name=name,
        edit_class=edit_class,
        description=description,
        dependencies=dependencies or [],
    )
    registry.register_plugin(plugin)


def get_available_edits() -> dict[str, EditPlugin]:
    """Get all available edit plugins."""
    return get_registry().get_available_edits()


def validate_edit_availability(edit_name: str) -> bool:
    """Check if an edit is available."""
    registry = get_registry()
    plugin = registry.get_plugin(edit_name)
    return plugin is not None and plugin.is_available


def get_edit_guard_policy(edit_name: str) -> dict[str, Any]:
    """Get the default guard policy for an edit."""
    policies = {
        "quant_rtn": {"spectral": {"scope": "all"}, "rmt": {"enable": True}},
    }
    return policies.get(edit_name, {})


def list_available_edits() -> list[str]:
    """List available edit names."""
    return get_registry().list_available()


def check_edit_dependencies(edit_name: str) -> dict[str, bool]:
    """Check if all dependencies for an edit are satisfied."""
    registry = get_registry()
    plugin = registry.get_plugin(edit_name)
    if plugin is None:
        return {}

    result = {}
    for dep in plugin.dependencies or ():
        try:
            __import__(dep)
            result[dep] = True
        except ImportError:
            result[dep] = False

    return result


__all__ = [
    "EditPlugin",
    "EditRegistry",
    "INVARLOCK_CORE_ABI",
    "NoopEdit",
    "RTNQuantEdit",
    "check_edit_dependencies",
    "get_available_edits",
    "get_edit_guard_policy",
    "get_registry",
    "list_available_edits",
    "register_edit",
    "validate_edit_availability",
]
