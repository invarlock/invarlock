from __future__ import annotations

from scripts.evidence_packs.python import preset_generator


def test_preset_generator_default_guards_order() -> None:
    guards = preset_generator.get_default_guards_order()
    assert "spectral" in guards
    assert "rmt" in guards
