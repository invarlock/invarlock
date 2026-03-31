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
        if "model" in config:
            assert isinstance(config.model.id, str) and len(config.model.id) > 0
        if "dataset" in config:
            assert config.dataset.provider == "wikitext2"
        # For edit configs, verify an edit is specified; task presets may omit edit.
        if subdir.startswith("edits") and "edit" in config:
            assert config.edit.name
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
    assert "prefetch_hf_assets_on_host" in contents
    assert "prefetching GPT-2 + WikiText-2 into host HF cache" in contents
    assert "evaluation report verification failed" in contents
    assert "proof-pack verification failed" in contents


def test_tiny_attested_smoke_campaign_script_is_executable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_tiny_attested_smoke.sh"
    assert script_path.exists(), "Expected scripts/run_tiny_attested_smoke.sh to exist"
    assert os.access(script_path, os.X_OK), (
        "run_tiny_attested_smoke.sh should be executable"
    )
    contents = script_path.read_text(encoding="utf-8")
    assert "kind: local_jsonl" in contents
    assert "sshleifer/tiny-gpt2" in contents
    assert "prefetch_tiny_model_on_host" in contents
    assert "evaluation report verification failed" in contents
    assert "proof-pack verification failed" in contents
