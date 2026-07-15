from __future__ import annotations

from pathlib import Path


def test_architecture_doc_tracks_shell_core_redesign() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    doc_path = repo_root / "docs" / "reference" / "architecture.md"
    text = doc_path.read_text(encoding="utf-8")

    required_snippets = (
        "cli/run_config.py",
        "cli/config_execution.py",
        "cli/run_pairing.py",
        "cli/run_metric_impact.py",
        "cli/run_execution.py",
        "`report_make.py` | Evaluation-report input normalization, build-section extraction, output shaping, and public report assembly",
        "`report_make_assembly.py` | Policy/provenance/guard assembly and report build-context composition",
        "`report_bundle.py` | Evaluation-bundle persistence, manifest writing, and evidence attachment",
        "`report_contract.py` | Input loading and report-generation planning",
        "`report_summary.py` | Console validation blocks and shared executive-summary/view-model derivation for reporting surfaces",
        "`run_policy.py`",
        "`retry.py`",
        "`run_snapshot_contract.py`",
        "`report_metric_impact.py`",
        "`run_report_contract.py` | Run provenance finalization, payload shaping, and run-report assembly contracts",
        "## Architecture Guardrails",
        "Package roots such as `adapters/__init__.py` and `guards/__init__.py` expose",
        "RMT ownership lives in `rmt.py`, `rmt_analysis.py`, and `rmt_detection.py`.",
        "Public command shells stay thin",
        "independently of `invarlock.cli`.",
    )

    missing = [snippet for snippet in required_snippets if snippet not in text]
    assert not missing, "\n".join(missing)
    assert "evaluate | run | verify" not in text
    assert "│  │ evaluate │ │   run    │" not in text
    assert "| `run` |" not in text
    assert "| `plugins` |" not in text
    assert "| `advanced` |" in text
    assert "| `version` |" in text
    assert "report_builder.py" not in text
    assert "report_make_support.py" not in text
