from __future__ import annotations

from pathlib import Path


def test_error_injection_set_includes_weight_tying_break() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    validation_suite = (
        repo_root / "scripts/proof_packs/lib/validation_suite.sh"
    ).read_text(encoding="utf-8")
    queue_manager = (repo_root / "scripts/proof_packs/lib/queue_manager.sh").read_text(
        encoding="utf-8"
    )
    internals_doc = (
        repo_root / "docs/user-guide/proof-packs-internals.md"
    ).read_text(encoding="utf-8")

    assert "weight_tying_break" in validation_suite
    assert "weight_tying_break" in queue_manager
    assert "weight_tying_break" in internals_doc

    assert "zero_layer" not in validation_suite
    assert "zero_layer" not in queue_manager
    assert "zero_layer" not in internals_doc

