from __future__ import annotations

from invarlock.reporting import report_provenance


def test_build_provenance_block_skips_edit_digest_for_non_mapping_report() -> None:
    provenance = report_provenance.build_provenance_block(
        [],
        None,
        {"run_id": "baseline-1"},
        {"report_path": "/tmp/edited.report.json"},
        {"tier": "balanced"},
        None,
        {},
        "edited-1",
        compute_report_digest_fn=lambda payload: f"digest:{type(payload).__name__}",
        collect_backend_versions_fn=lambda: {"python": "3.12"},
        compute_edit_digest_fn=lambda payload: f"edit:{payload}",
    )

    assert provenance["baseline"]["run_id"] == "baseline-1"
    assert provenance["edited"]["run_id"] == "edited-1"
    assert provenance["env_flags"] == {"python": "3.12"}
    assert "edit_digest" not in provenance
