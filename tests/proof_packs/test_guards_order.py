from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.proof_packs.python import preset_generator


def _extract_guard_order_from_yaml(text: str) -> list[str]:
    order: list[str] = []
    in_guards = False
    in_order = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "guards:":
            in_guards = True
            continue
        if in_guards and stripped == "order:":
            in_order = True
            continue
        if in_order:
            if stripped.startswith("- "):
                order.append(stripped[2:].strip())
                continue
            if stripped:
                break
    return order


def test_config_generator_default_guards_order() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                "set -e\n"
                "source scripts/proof_packs/lib/config_generator.sh\n"
                'generate_invarlock_config "model" "/dev/stdout" "noop" 42 10 20 100 '
                "128 64 1\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    order = _extract_guard_order_from_yaml(result.stdout)
    assert "spectral" in order, "spectral missing from default guards_order"
    assert "rmt" in order, "rmt missing from default guards_order"


def test_preset_generator_default_guards_order() -> None:
    guards = preset_generator.get_default_guards_order()
    assert "spectral" in guards
    assert "rmt" in guards
