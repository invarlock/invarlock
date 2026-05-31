from __future__ import annotations

from invarlock.reporting.report_types import AutoConfig


def make_auto_config(
    *,
    enabled: bool = False,
    tier: str = "balanced",
    probes_used: int = 0,
    target_pm_ratio: float | None = None,
) -> AutoConfig:
    """Return a fully typed AutoConfig for report fixture builders."""
    return AutoConfig(
        enabled=enabled,
        tier=tier,
        probes_used=probes_used,
        target_pm_ratio=target_pm_ratio,
    )
