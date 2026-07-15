from __future__ import annotations

import re
from pathlib import Path


def _get_make_target_block(text: str, target: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(target)}\s*:\s*(?:##.*)?$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    start = m.end()
    # Collect subsequent tab-indented recipe lines until next target or EOF
    lines = []
    for line in text[start:].splitlines():
        if not line:
            # keep blank lines within recipe
            lines.append(line)
            continue
        if re.match(r"^[A-Za-z0-9_.-]+\s*:\s*", line):
            break
        lines.append(line)
    return "\n".join(lines)


def test_verify_target_runs_docs_api_refs_check() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "verify")
    assert block is not None, "verify target not found in Makefile"
    # Either always run or behind an env flag is acceptable; require presence
    # of the script path within the verify recipe body.
    assert "scripts/docs/docs_check.py --api-refs" in block, (
        "verify target should include docs API refs validation (optionally gated)"
    )


def test_verify_target_runs_repo_cruft_check() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "verify")
    assert block is not None, "verify target not found in Makefile"
    assert "$(MAKE) repo-cruft-check" in block, (
        "verify target should fail fast on macOS transport artifacts"
    )


def test_verify_target_runs_public_evidence_audit() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "verify")
    assert block is not None, "verify target not found in Makefile"
    assert "$(MAKE) public-evidence-audit" in block, (
        "verify target should fail fast on overclaimed public evidence"
    )


def test_verify_target_runs_scripts_inventory_check() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "verify")
    assert block is not None, "verify target not found in Makefile"
    assert "$(MAKE) scripts-inventory-check" in block, (
        "verify target should fail fast on unclassified scripts"
    )


def test_verify_target_runs_architecture_fragmentation_check() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "verify")
    assert block is not None, "verify target not found in Makefile"
    assert "$(MAKE) architecture-fragmentation-check" in block, (
        "verify target should track source fragmentation metrics"
    )


def test_makefile_exposes_scripts_audit_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "scripts-audit")
    assert block is not None, "scripts-audit target not found in Makefile"
    assert "scripts/check_scripts_inventory.py --json" in block


def test_empirical_and_negative_evidence_are_not_release_authority() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    makefile = repo_root / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    preflight_source = (repo_root / "scripts/release/release_preflight.py").read_text(
        encoding="utf-8"
    )

    inventory = _get_make_target_block(data, "empirical-guard-inventory-check")
    release_shape = _get_make_target_block(data, "release-evidence-check")
    release_preflight = _get_make_target_block(data, "release-preflight")

    assert inventory is not None
    assert "evidence_contracts.py empirical-inventory" in inventory
    assert "empirical-guard-evidence-check" not in data
    assert release_shape is not None
    assert "empirical" not in release_shape
    assert release_preflight is not None
    assert "empirical" not in release_preflight
    assert "--require-current-negative-evidence" not in preflight_source
    assert "_run_current_negative_evidence_audit" not in preflight_source


def test_contracts_check_runs_model_candidate_compatibility_audit() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "contracts-check")
    assert block is not None, "contracts-check target not found in Makefile"
    assert "scripts/checks/check_model_classification.py" in block
    assert "scripts/checks/check_model_candidate_compatibility.py" in block
