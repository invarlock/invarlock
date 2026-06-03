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


def test_compute_edit_digest_uses_provenance_edits_fallback() -> None:
    with_empty_config = report_provenance.compute_edit_digest(
        {"edit": {"name": "quant_rtn", "config": {}}}
    )
    with_provenance_edits = report_provenance.compute_edit_digest(
        {"provenance": {"edits": {"name": "quant_rtn", "config": ["bad"]}}}
    )

    assert with_provenance_edits["family"] == "quantization"
    assert with_provenance_edits["impl_hash"] == with_empty_config["impl_hash"]


def test_compute_edit_digest_ignores_non_mapping_provenance_edits() -> None:
    digest = report_provenance.compute_edit_digest({"provenance": {"edits": "legacy"}})

    assert digest["family"] == "report_only"


def test_compute_report_digest_handles_non_mapping_metric_sections() -> None:
    digest = report_provenance.compute_report_digest(
        {
            "meta": {"model_id": "model-a"},
            "edit": {"name": "noop", "plan_digest": "deadbeef"},
            "metrics": {"spectral": ["bad"], "rmt": "not-a-mapping"},
        }
    )

    assert isinstance(digest, str)
    assert len(digest) == 16


def test_build_provenance_block_ignores_noncanonical_optional_fields() -> None:
    provenance = report_provenance.build_provenance_block(
        {
            "provenance": {
                "provider_digest": "legacy-provider",
                "dataset_split": "",
                "split_fallback": "yes",
            }
        },
        {"artifacts": {}},
        {"run_id": "baseline-2"},
        {"report_path": "/tmp/edited.report.json"},
        {"tier": "strict"},
        "feedface",
        [],
        "edited-2",
        compute_report_digest_fn=lambda payload: f"digest:{type(payload).__name__}",
        collect_backend_versions_fn=lambda: {"python": "3.12"},
        compute_edit_digest_fn=lambda payload: f"edit:{payload}",
    )

    assert provenance["provider_digest"] == {"ids_sha256": "feedface"}
    assert "dataset_split" not in provenance
    assert "split_fallback" not in provenance
    assert "window_plan" not in provenance
