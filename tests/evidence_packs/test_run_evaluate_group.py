from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.evidence_packs.python import run_evaluate_group as mod


def test_run_entry_scopes_shared_evaluate_tmp_dir(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, str | None] = {}
    group_tmp = tmp_path / "group" / "tmp" / "evaluate"
    report_out = tmp_path / "reports" / "run_1"

    def fake_evaluate_command(**kwargs: object) -> None:
        observed["evaluate_tmp_dir"] = os.environ.get("INVARLOCK_EVALUATE_TMP_DIR")
        out_dir = Path(str(kwargs["report_out"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evaluation.report.json").write_text(
            json.dumps({"ok": True}) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(mod, "evaluate_command", fake_evaluate_command)
    monkeypatch.delenv("INVARLOCK_EVALUATE_TMP_DIR", raising=False)

    result = mod._run_entry(
        {
            "baseline": str(tmp_path / "baseline"),
            "subject": str(tmp_path / "subject"),
            "profile": "ci",
            "tier": "balanced",
            "out": str(tmp_path / "runs"),
            "report_out": str(report_out),
            "preset": str(tmp_path / "preset.yaml"),
            "config_root": str(tmp_path / "config_root"),
            "work_dir": str(tmp_path / "work"),
            "evaluate_tmp_dir": str(group_tmp),
        }
    )

    assert result["ok"] is True
    assert observed["evaluate_tmp_dir"] == str(group_tmp)
    assert "INVARLOCK_EVALUATE_TMP_DIR" not in os.environ
