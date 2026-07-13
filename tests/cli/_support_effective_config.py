from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from invarlock.cli.run_config import prepare_config_for_run


def preserve_effective_config(run_kwargs: Mapping[str, Any]) -> None:
    """Make a run stub honor evaluate's resolved-config side effect."""

    output = run_kwargs.get("resolved_config_out")
    if output is None:
        return
    prepare_config_for_run(
        config_path=str(run_kwargs["config"]),
        profile=(
            str(run_kwargs["profile"])
            if run_kwargs.get("profile") is not None
            else None
        ),
        edit=None,
        tier=(str(run_kwargs["tier"]) if run_kwargs.get("tier") is not None else None),
        probes=None,
        resolved_config_out=str(output),
    )


__all__ = ["preserve_effective_config"]
