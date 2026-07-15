from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_root_workflow_visual_tracks_current_core_surfaces() -> None:
    readme = _read("README.md")
    svg = _read("docs/assets/evaluation-verification-flow.svg")

    assert 'src="docs/assets/evaluation-verification-flow.svg"' in readme
    assert (
        "raw.githubusercontent.com/invarlock/invarlock/main/docs/assets/"
        "evaluation-verification-flow.svg" not in readme
    )

    required = (
        "invarlock evaluate",
        "evaluation.report.json",
        "runtime.manifest.json",
        "Independent verifier inputs",
        "raw baseline report",
        "acceptance policy pack",
        "expected runtime-image digest",
        "invarlock verify",
        "invarlock report html",
        "exit 0: verified",
        "nonzero: rejected",
    )
    assert not [item for item in required if item not in svg]
    assert "promote" not in svg.lower()
    assert "optional evidence pack" not in svg.lower()


def test_reference_diagrams_track_guard_artifact_and_module_contracts() -> None:
    architecture = _read("docs/reference/architecture.md")
    guards = _read("docs/reference/guards.md")
    reports = _read("docs/reference/reports.md")

    assert (
        "invariants(pre).validate → edit/noop stage" in architecture
        and "invariants(post).validate → evaluate → finalize run report" in architecture
    )
    for module in (
        "retry.py",
        "run_snapshot_contract.py",
        "reporting/run_report_contract.py",
    ):
        assert module in architecture
    assert "run_retry" not in architecture
    assert "run_snapshot, run_report" not in architecture

    assert "prepare model + prepare all guards" in guards
    assert "spectral.validate → rmt.validate → variance.validate" in guards
    assert "invariants(post).validate → evaluate → finalize" in guards

    assert "independent verifier inputs" in reports
    assert "retained raw baseline report" in reports
    assert "acceptance policy pack" in reports
    assert "expected runtime-image digest" in reports
    assert "exit 0 + JSON verification result, or nonzero + diagnostics" in reports
