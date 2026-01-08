import os
from pathlib import Path

import pytest

from invarlock.cli.config import load_config

EXPECTED_CONFIGS = [
    ("presets/causal_lm", "wikitext2_512.yaml", None),
    ("overlays/edits/quant_rtn", "8bit_attn.yaml", None),
    ("overlays/edits/quant_rtn", "8bit_full.yaml", None),
]


def test_small_workflow_configs_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    configs_dir = repo_root / "configs"

    for subdir, primary, fallback in EXPECTED_CONFIGS:
        primary_path = configs_dir / subdir / primary
        cfg_path = primary_path
        if not primary_path.exists() and fallback:
            fb_path = configs_dir / subdir / fallback
            if fb_path.exists():
                cfg_path = fb_path
        assert cfg_path.exists(), f"Expected config {primary} (or fallback) to exist"

        config = load_config(str(cfg_path))
        model_section = getattr(config, "model", None)
        if model_section is not None:
            assert isinstance(model_section.id, str) and len(model_section.id) > 0
        dataset_section = getattr(config, "dataset", None)
        if dataset_section is not None:
            assert dataset_section.provider == "wikitext2"
        # For edit configs, verify an edit is specified; task presets may omit edit.
        if subdir.startswith("edits"):
            assert getattr(config.edit, "name", None)
        # Presets carry tier context via profile; auto tier may not be set at top-level


def test_cert_script_is_executable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "cert_once.sh"
    assert script_path.exists(), "Expected scripts/cert_once.sh to exist"
    assert os.access(script_path, os.X_OK), "cert_once.sh should be executable"


def test_agent_guidance_doc_contains_workflow() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    guide_path = repo_root / "AGENTS.md"
    if not guide_path.exists():
        pytest.skip("AGENTS.md not present in repository checkout")
    contents = guide_path.read_text()
    assert "Repository Guidelines" in contents
    assert "Agent Workflow (Required Process)" in contents
