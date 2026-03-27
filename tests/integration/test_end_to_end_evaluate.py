import os
from pathlib import Path

from invarlock.core.config_runtime import load_config

EXPECTED_CONFIGS = [
    ("presets/causal_lm", "gpt2_smoke_128.yaml", None),
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


def test_eval_script_is_executable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "eval_once.sh"
    assert script_path.exists(), "Expected scripts/eval_once.sh to exist"
    assert os.access(script_path, os.X_OK), "eval_once.sh should be executable"


def test_gpt2_smoke_campaign_script_is_executable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_gpt2_smoke_campaign.sh"
    assert script_path.exists(), "Expected scripts/run_gpt2_smoke_campaign.sh to exist"
    assert os.access(script_path, os.X_OK), (
        "run_gpt2_smoke_campaign.sh should be executable"
    )
    contents = script_path.read_text(encoding="utf-8")
    assert "ensure_writable_hf_cache" in contents
    assert "INVARLOCK_SMOKE_HOST_HF_CACHE_ROOT" in contents
    assert 'CLI=("$PYTHON_BIN" -m invarlock)' in contents
    assert "command -v invarlock" not in contents
    assert "INVARLOCK_SMOKE_CACHE_COMPLETE" in contents
    assert "evaluation report verification failed" in contents
    assert "proof-pack verification failed" in contents
