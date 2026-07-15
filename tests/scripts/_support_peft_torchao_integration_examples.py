from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATIONS_DIR = REPO_ROOT / "examples" / "integrations"
SOURCE_MATRIX = INTEGRATIONS_DIR / "source_matrix.json"
PEFT_DIR = REPO_ROOT / "examples" / "integrations" / "peft_lora"
FINE_TUNE_DIR = REPO_ROOT / "examples" / "integrations" / "fine_tune"
MAGNITUDE_PRUNE_DIR = REPO_ROOT / "examples" / "integrations" / "magnitude_prune"
TORCHAO_DIR = REPO_ROOT / "examples" / "integrations" / "torchao_int8_runtime"
EXAMPLE_RUNNERS = [
    INTEGRATIONS_DIR / "awq" / "run_tiny_awq.sh",
    INTEGRATIONS_DIR / "compressed_tensors" / "run_tiny_hf_ct.sh",
    INTEGRATIONS_DIR / "gptqmodel" / "run_tiny_gptqmodel.sh",
    INTEGRATIONS_DIR / "hf_bnb" / "run_tiny_hf_bnb_8bit.sh",
    INTEGRATIONS_DIR / "hqq" / "run_tiny_hf_hqq.sh",
    INTEGRATIONS_DIR / "lm_eval_harness" / "run_tiny_lm_eval_sidecar.sh",
    INTEGRATIONS_DIR / "peft_lora" / "run_tiny_peft_lora.sh",
    INTEGRATIONS_DIR / "fine_tune" / "run_tiny_fine_tune.sh",
    INTEGRATIONS_DIR / "magnitude_prune" / "run_tiny_magnitude_prune.sh",
    INTEGRATIONS_DIR / "quanto" / "run_tiny_hf_quanto.sh",
    INTEGRATIONS_DIR / "torchao_int8_runtime" / "run_tiny_hf_torchao_int8.sh",
]
README_EXAMPLES = [
    "awq",
    "compressed_tensors",
    "gptqmodel",
    "hf_bnb",
    "hqq",
    "lm_eval_harness",
    "peft_lora",
    "fine_tune",
    "magnitude_prune",
    "quanto",
    "torchao_int8_runtime",
]


def _load_source_matrix() -> dict[str, dict[str, object]]:
    payload = json.loads(SOURCE_MATRIX.read_text(encoding="utf-8"))
    assert payload["schema"] == "invarlock.integration_source_matrix.v1"
    entries = payload["entries"]
    assert isinstance(entries, list)
    return {entry["target"]: entry for entry in entries}
