from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.commands.explain_gates import explain_gates_command


def _mk_pairable_reports(ratio: float = 1.101) -> tuple[dict, dict]:
    # Baseline with finite ppl_final
    base = {
        "meta": {"model_id": "m", "adapter": "hf_gpt2", "device": "cpu", "seed": 42},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 50.0},
            "bootstrap": {"replicates": 400, "alpha": 0.05},
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [1.0, 1.1],
                "token_counts": [100, 200],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [1.0, 1.1],
                "token_counts": [100, 200],
            },
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    sub = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "seed": 42,
            "auto": {"tier": "balanced"},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 50.0 * ratio,
            },
            "preview_total_tokens": 30000,
            "final_total_tokens": 30000,
            "bootstrap": {"replicates": 400, "alpha": 0.05},
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [1.0, 1.1],
                "token_counts": [100, 200],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [1.0, 1.1],
                "token_counts": [100, 200],
            },
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    return sub, base


def test_explain_gates_hysteresis(tmp_path: Path, capsys):
    subject, baseline = _mk_pairable_reports(
        ratio=1.101
    )  # base limit 1.10 → needs hysteresis
    subj_path = tmp_path / "subject.json"
    base_path = tmp_path / "baseline.json"
    subj_path.write_text(json.dumps(subject))
    base_path.write_text(json.dumps(baseline))

    explain_gates_command(report=str(subj_path), baseline=str(base_path))
    out = capsys.readouterr().out.lower()
    assert "hysteresis applied" in out
    assert "effective" in out and ("threshold" in out or "limit" in out)
    # Mention token floors or min examples
    assert ("token floors" in out) or ("min examples" in out) or ("min_tokens" in out)
