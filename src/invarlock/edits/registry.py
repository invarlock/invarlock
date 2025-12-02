"""
InvarLock Edits Registry
===================

Plugin discovery and registration system for model edits.
Supports entry point-based discovery and manual registration.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class EditPlugin:
    """Plugin metadata for a model edit."""

    name: str
    edit_class: type
    description: str
    is_available: bool = True
    dependencies: list[str] | None = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class EditRegistry:
    """Registry for model edit plugins."""

    def __init__(self):
        self._plugins: dict[str, EditPlugin] = {}
        self._discover_plugins()

    def _discover_plugins(self):
        """Discover and register built-in edit plugins."""
        # Register built-in edits
        try:
            from .quant_rtn import RTNQuantEdit

            self.register_plugin(
                EditPlugin(
                    name="quant_rtn",
                    edit_class=RTNQuantEdit,
                    description="RTN (Random Truncation to N-bits) quantization",
                    is_available=True,
                )
            )
        except ImportError:
            pass

        # Register noop edit for baseline/calibration runs
        try:
            from .noop import NoopEdit

            self.register_plugin(
                EditPlugin(
                    name="noop",
                    edit_class=NoopEdit,
                    description="No-op edit for baseline and calibration runs",
                    is_available=True,
                )
            )
        except ImportError:
            pass

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


# Global registry instance
_registry = None


def get_registry() -> EditRegistry:
    """Get the global edit registry."""
    global _registry
    if _registry is None:
        _registry = EditRegistry()
    return _registry


def register_edit(
    name: str,
    edit_class: type,
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
    # Default policies by edit type
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
    for dep in plugin.dependencies:
        try:
            __import__(dep)
            result[dep] = True
        except ImportError:
            result[dep] = False

    return result


def print_edit_status() -> None:
    """Print status of all registered edits."""
    registry = get_registry()
    for name, plugin in registry._plugins.items():
        status = "✓" if plugin.is_available else "✗"
        print(f"{status} {name}: {plugin.description}")
