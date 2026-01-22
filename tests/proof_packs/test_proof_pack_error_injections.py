from __future__ import annotations

import json
from pathlib import Path


def test_error_injection_set_includes_weight_tying_break() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    scenarios_path = repo_root / "scripts/proof_packs/scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenario_ids = {entry.get("id") for entry in scenarios.get("scenarios", [])}

    assert "weight_tying_break" in scenario_ids
    assert "zero_layer" not in scenario_ids

    # Ensure the harness is wired to the manifest (avoid drift between task graph and verdict).
    queue_manager = (repo_root / "scripts/proof_packs/lib/queue_manager.sh").read_text(
        encoding="utf-8"
    )
    assert "scenarios.json" in queue_manager
