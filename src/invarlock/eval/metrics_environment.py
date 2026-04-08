"""Environment and capability helpers for eval metrics."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from .metrics_support import DependencyManager, MetricsConfig

logger = logging.getLogger(__name__)
_ENVIRONMENT_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True)
class MetricsEnvironmentReport:
    ok: bool
    available_dependencies: tuple[str, ...]
    missing_dependencies: tuple[tuple[str, str], ...]
    messages: tuple[str, ...] = ()


def get_metrics_info() -> dict[str, Any]:
    dep_manager = DependencyManager()
    return {
        "available_metrics": ["sigma_max", "head_energy", "mi_gini"],
        "available_dependencies": list(dep_manager.available_modules.keys()),
        "missing_dependencies": dep_manager.get_missing_dependencies(),
        "default_config": asdict(MetricsConfig(use_cache=False)),
    }


def validate_metrics_environment() -> MetricsEnvironmentReport:
    messages: list[str] = []
    try:
        dep_manager = DependencyManager()
        MetricsConfig(use_cache=False)

        messages.append("Basic dependencies available")
        logger.info("✓ Basic dependencies available")
        available_count = len(dep_manager.available_modules)
        total_count = available_count + len(dep_manager.missing_modules)
        messages.append(
            f"{available_count}/{total_count} optional dependencies available"
        )
        logger.info(
            f"✓ {available_count}/{total_count} optional dependencies available"
        )

        if dep_manager.missing_modules:
            missing_messages: list[str] = []
            logger.warning("Some optional dependencies are missing:")
            for name, error in dep_manager.missing_modules:
                line = f"{name}: {error}"
                missing_messages.append(line)
                logger.warning(f"  - {line}")

            messages.extend(missing_messages)

        return MetricsEnvironmentReport(
            ok=True,
            available_dependencies=tuple(dep_manager.available_modules.keys()),
            missing_dependencies=tuple(dep_manager.missing_modules),
            messages=tuple(messages),
        )
    except _ENVIRONMENT_ERRORS as e:
        logger.error(f"Environment validation failed: {e}")
        messages.append(str(e))
        return MetricsEnvironmentReport(
            ok=False,
            available_dependencies=(),
            missing_dependencies=(),
            messages=tuple(messages),
        )


__all__ = [
    "MetricsEnvironmentReport",
    "get_metrics_info",
    "validate_metrics_environment",
]
