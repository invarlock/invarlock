from __future__ import annotations

from io import StringIO

from rich.console import Console

from invarlock.cli import run_config as run_config_mod


class _BrokenConfig:
    def model_dump(self) -> dict:
        raise RuntimeError("broken dump")


def test_prepare_config_for_run_handles_model_dump_failure_without_auto_adapter() -> (
    None
):
    events: list[tuple[str, str]] = []

    def _event_fn(console, tag: str, message: str, **kwargs) -> None:  # noqa: ARG001
        events.append((tag, message))

    cfg = run_config_mod.prepare_config_for_run(
        config_path="config.yaml",
        profile="dev",
        edit=None,
        tier="balanced",
        probes=None,
        console=Console(file=StringIO(), force_terminal=False),
        event_fn=_event_fn,
        invarlock_config_cls=lambda payload: payload,
        load_config_fn=lambda path: _BrokenConfig(),  # noqa: ARG005
        apply_profile_fn=lambda cfg, profile: cfg,  # noqa: ARG005
    )

    assert cfg == {"auto": {"tier": "balanced"}}
    assert ("INIT", "Loading configuration: config.yaml") in events
    assert ("INIT", "Auto tier override: balanced") in events
