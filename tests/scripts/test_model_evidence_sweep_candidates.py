from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.scripts._support_model_evidence_sweep import load_script_module


def test_promotion_candidate_text_suite_covers_prepared_deferred_lanes() -> None:
    mod = load_script_module("model_evidence_sweep")

    specs = {
        lane.slug: lane
        for lane in mod.select_specs(
            mod.PROMOTION_CANDIDATE_TEXT_LE_14B_SUITE,
            slugs=[],
            lane_ids=[],
            shard_index=0,
            shard_count=1,
        )
    }

    assert set(specs) == {
        "openllama_7b",
        "opt_1_3b",
        "falcon_7b",
        "glm4_9b_chat",
        "distilbert_base_uncased",
    }
    assert specs["openllama_7b"].preset_relpath == (
        "configs/presets/causal_lm/openllama_7b_512.yaml"
    )
    assert specs["opt_1_3b"].preset_relpath == (
        "configs/presets/causal_lm/opt_1_3b_512.yaml"
    )
    assert specs["falcon_7b"].adapter == "auto"
    assert specs["glm4_9b_chat"].adapter == "auto"
    assert specs["distilbert_base_uncased"].adapter == "hf_mlm"
    assert specs["distilbert_base_uncased"].preset_relpath == (
        "configs/presets/masked_lm/distilbert_base_uncased_128.yaml"
    )


def test_promotion_candidate_text_suite_glm_host_dry_run_uses_lane_preset(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence_sweep.py"
    output_root = tmp_path / "candidate-glm-host"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "promotion-candidates-text-le-14b",
            "--slug",
            "glm4_9b_chat",
            "--execution-mode",
            "host",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload) == 1
    assert payload[0]["slug"] == "glm4_9b_chat"
    assert payload[0]["prefetch"][-1] == "THUDM/glm-4-9b-chat"
    preset_idx = payload[0]["evaluate"].index("--preset") + 1
    assert payload[0]["evaluate"][preset_idx] == str(
        repo_root / "configs/presets/causal_lm/glm4_9b_chat_512.yaml"
    )
    assert "--runtime-provenance" in payload[0]["verify"]

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite"] == "promotion-candidates-text-le-14b"
    assert manifest["lanes"][0]["slug"] == "glm4_9b_chat"
