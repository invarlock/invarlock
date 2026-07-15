from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_checkpoint_identity_benchmark_reports_measured_and_projected_costs(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"x" * (2 * 1024 * 1024))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/checks/benchmark_checkpoint_identity.py",
            str(checkpoint),
            "--repeat",
            "1",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["schema"] == "invarlock.checkpoint_identity_benchmark.v1"
    assert payload["checkpoint_bytes"] >= 2 * 1024 * 1024
    assert payload["repeat"] == 1
    assert payload["seconds_per_hash"] > 0
    assert payload["gib_per_second"] > 0
    assert payload["strict_local_full_reads"] == 3
    assert payload["projected_seconds"]["7b_bf16_13.04_gib"] > 0
    assert payload["projected_seconds"]["32b_bf16_59.60_gib"] > 0
