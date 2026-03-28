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
        "cli/run_overhead.py",
        "cli/run_artifacts.py",
        "cli/run_execution.py",
        "`report_make.py` | Canonical evaluation-report assembly owner",
        "`report_bundle.py` | Evaluation-bundle persistence, manifest writing, and evidence attachment",
        "`report_contract.py` | Input loading and report-generation planning",
        "`report_console.py`",
        "`report_summary.py` | Shared executive-summary/view-model derivation for reporting surfaces",
        "`run_policy.py`",
        "`run_retry_policy.py`",
        "`run_snapshot_contract.py` + `run_snapshot_policy.py`",
        "`run_guard_overhead_policy.py`",
        "`run_provenance_contract.py` + `run_report_contract.py`",
        "## Architecture Guardrails",
        "No lazy exports",
        "No `rmt_legacy` references in production source.",
        "No dependency-map orchestration for `run`.",
        "No CLI imports inside owner layers.",
    )

    missing = [snippet for snippet in required_snippets if snippet not in text]
    assert not missing, "\n".join(missing)
    assert "report_builder.py" not in text
    assert "report_make_support.py" not in text
