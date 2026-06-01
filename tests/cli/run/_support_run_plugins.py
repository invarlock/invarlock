from __future__ import annotations

import json
from pathlib import Path

from tests.cli.run._support_run_common import common_ce_patches, write_base_run_config


def plugin_provenance_cfg(tmp_path: Path, preview: int = 4, final: int = 4) -> Path:
    return write_base_run_config(
        tmp_path,
        preview,
        final,
        edit_name="structured",
        eval_fields="  spike_threshold: 2.0\n",
    )


def plugin_provenance_common_ce():
    return common_ce_patches(
        include_registry=True,
        include_save_report=True,
        tokenizer_vocab_size=1000,
    )


def plugins_invariants_write_cfg(
    tmp_path: Path, preview: int = 2, final: int = 2, loss_type: str = "auto"
) -> Path:
    return write_base_run_config(
        tmp_path,
        preview,
        final,
        eval_fields="  spike_threshold: 2.0\n",
        loss_type=loss_type,
    )


def plugins_invariants_common_ce():
    return common_ce_patches(
        include_registry=True,
        include_save_report=True,
    )


def plugins_invariants_baseline_with_meta(
    tmp_path: Path, meta: dict, preview_ids, final_ids
) -> Path:
    p = tmp_path / "baseline.json"
    payload = {
        "meta": meta,
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 1.0,
            }
        },
        "edit": {
            "name": "structured",
            "plan_digest": "baseline",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {
            "preview": {"window_ids": [0], "input_ids": preview_ids},
            "final": {"window_ids": [1], "input_ids": final_ids},
        },
    }
    p.write_text(json.dumps(payload))
    return p
