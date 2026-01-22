from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from invarlock.cli.config import apply_profile


def _patch_exists_block_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Path], bool]:
    """Patch Path.exists to pretend configs/profiles files are absent.

    Returns the original Path.exists so tests can call it if needed.
    """

    orig_exists = Path.exists

    def fake_exists(self: Path) -> bool:  # type: ignore[override]
        s = str(self)
        if "configs/profiles" in s:
            return False
        return orig_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists, raising=False)
    return orig_exists


@pytest.mark.unit
def test_apply_profile_uses_packaged_runtime_when_configs_absent(
    monkeypatch: pytest.MonkeyPatch,
):
    # Simulate missing repo configs/profiles to force packaged runtime path
    _patch_exists_block_configs(monkeypatch)

    # Should still load release profile from package data
    rel = apply_profile(
        # minimal base cfg
        cfg=__import__("invarlock.cli.config").cli.config.InvarLockConfig(
            model={"id": "gpt2", "adapter": "hf_causal"},
            edit={"name": "noop", "plan": {}},
        ),
        profile="release",
    )
    assert rel.dataset.preview_n >= 400 and rel.eval.bootstrap.replicates >= 3200

    # And CI CPU profile forces CPU device and stride
    ci_cpu = apply_profile(
        cfg=__import__("invarlock.cli.config").cli.config.InvarLockConfig(
            model={"id": "gpt2", "adapter": "hf_causal"},
            edit={"name": "noop", "plan": {}},
        ),
        profile="ci_cpu",
    )
    assert ci_cpu.model.device == "cpu" and ci_cpu.dataset.stride > 0
