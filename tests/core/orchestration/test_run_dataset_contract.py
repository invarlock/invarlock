from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core.exceptions import InvarlockError
from invarlock.core.run_provider_dataset_plan import materialize_run_dataset
from invarlock.eval.data_support import DatasetDiagnostic


def test_materialize_run_dataset_uses_provider_plan_diagnostics() -> None:
    diagnostic = DatasetDiagnostic(
        kind="provider.loaded",
        severity="info",
        message="loaded",
        metadata={"provider": "synthetic"},
    )
    result = materialize_run_dataset(
        pairing_schedule=None,
        cfg=SimpleNamespace(dataset=SimpleNamespace(provider="synthetic")),
        baseline_report_data=None,
        tokenizer_hash="tok",
        resolved_loss_type="ppl_causal",
        profile="dev",
        model_profile=SimpleNamespace(),
        tokenizer=None,
        use_mlm=False,
        mask_prob=0.0,
        mask_seed=1,
        random_token_prob=0.0,
        original_token_prob=0.0,
        tier=None,
        requested_preview=2,
        requested_final=2,
        effective_preview=2,
        effective_final=2,
        resolved_device="cpu",
        profile_normalized="dev",
        resolved_split=None,
        validate_and_harvest_baseline_schedule_fn=lambda *args, **kwargs: None,
        materialize_baseline_pairing_schedule_fn=lambda *args, **kwargs: None,
        resolve_tokenizer_fn=lambda *args, **kwargs: (None, "tok"),
        build_provider_dataset_plan_fn=lambda **kwargs: SimpleNamespace(
            resolved_split="validation",
            used_fallback_split=False,
            tokenizer="tok",
            tokenizer_hash="tok-hash",
            calibration_data=[{"input_ids": [1]}],
            dataset_meta={"provider": "synthetic"},
            window_plan={"preview": 2, "final": 2},
            preview_count=2,
            final_count=2,
            effective_preview=2,
            effective_final=2,
            preview_mask_counts=[],
            final_mask_counts=[],
            preview_records=[{"window_id": 0}],
            final_records=[{"window_id": 1}],
            diagnostics=[diagnostic],
        ),
    )

    assert result.resolved_split == "validation"
    assert result.tokenizer_hash == "tok-hash"
    assert result.preview_count == 2
    assert result.diagnostics == (diagnostic,)


def test_materialize_run_dataset_harvests_pairing_schedule() -> None:
    harvested = {
        "dataset_meta": {"provider": "baseline"},
        "window_plan": {"preview": 1, "final": 1},
        "calibration_data": [{"input_ids": [1, 2, 3]}],
    }
    materialized = SimpleNamespace(
        calibration_data=[{"input_ids": [1, 2, 3]}],
        dataset_meta={"provider": "baseline"},
        window_plan={"preview": 1, "final": 1},
        preview_count=1,
        final_count=1,
        effective_preview=1,
        effective_final=1,
        preview_mask_counts=[0],
        final_mask_counts=[0],
    )

    result = materialize_run_dataset(
        pairing_schedule={"preview": {}, "final": {}},
        cfg=SimpleNamespace(dataset=SimpleNamespace(provider=None)),
        baseline_report_data={"evaluation_windows": {}},
        tokenizer_hash="tok",
        resolved_loss_type="ppl_mlm",
        profile="dev",
        model_profile=SimpleNamespace(),
        tokenizer=None,
        use_mlm=True,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        tier="balanced",
        requested_preview=1,
        requested_final=1,
        effective_preview=1,
        effective_final=1,
        resolved_device="cpu",
        profile_normalized="dev",
        resolved_split="validation",
        validate_and_harvest_baseline_schedule_fn=lambda *args, **kwargs: harvested,
        materialize_baseline_pairing_schedule_fn=lambda **kwargs: materialized,
        resolve_tokenizer_fn=lambda profile: ("tok", "tok-hash"),
        build_provider_dataset_plan_fn=lambda **kwargs: None,
    )

    assert result.resolved_split == "validation"
    assert result.dataset_meta == {"provider": "baseline"}
    assert result.preview_count == 1
    assert result.final_count == 1
    assert result.preview_records == []
    assert result.final_records == []


def test_materialize_run_dataset_preserves_materialized_pairing_records() -> None:
    harvested = {
        "dataset_meta": {"provider": "baseline"},
        "window_plan": {"preview": 1, "final": 1},
        "calibration_data": [{"example_id": "ex-1"}],
    }
    materialized = SimpleNamespace(
        calibration_data=[{"example_id": "ex-1"}],
        dataset_meta={"provider": "baseline"},
        window_plan={"preview": 1, "final": 1},
        preview_count=1,
        final_count=1,
        effective_preview=1,
        effective_final=1,
        preview_mask_counts=[0],
        final_mask_counts=[0],
        preview_records=[{"example_id": "ex-1", "window_id": "preview::0"}],
        final_records=[{"example_id": "ex-2", "window_id": "final::0"}],
    )

    result = materialize_run_dataset(
        pairing_schedule={"preview": {}, "final": {}},
        cfg=SimpleNamespace(dataset=SimpleNamespace(provider=None)),
        baseline_report_data={"evaluation_windows": {}},
        tokenizer_hash="tok",
        resolved_loss_type="classification",
        profile="dev",
        model_profile=SimpleNamespace(),
        tokenizer=None,
        use_mlm=False,
        mask_prob=0.0,
        mask_seed=1,
        random_token_prob=0.0,
        original_token_prob=0.0,
        tier="balanced",
        requested_preview=1,
        requested_final=1,
        effective_preview=1,
        effective_final=1,
        resolved_device="cpu",
        profile_normalized="dev",
        resolved_split=None,
        validate_and_harvest_baseline_schedule_fn=lambda *args, **kwargs: harvested,
        materialize_baseline_pairing_schedule_fn=lambda **kwargs: materialized,
        resolve_tokenizer_fn=lambda *args, **kwargs: (None, "tok"),
        build_provider_dataset_plan_fn=lambda **kwargs: None,
    )

    assert result.preview_records == [{"example_id": "ex-1", "window_id": "preview::0"}]
    assert result.final_records == [{"example_id": "ex-2", "window_id": "final::0"}]


def test_materialize_run_dataset_rehydrates_multimodal_pairing_from_provider() -> None:
    observed: dict[str, object] = {}
    runtime_record = {
        "id": "ex-1",
        "example_id": "ex-1",
        "image_path": "/runtime/ex-1.ppm",
        "image_sha256": "a" * 64,
    }

    def materialize(**kwargs):
        observed["calibration_data"] = kwargs["calibration_data"]
        return SimpleNamespace(
            calibration_data=list(kwargs["calibration_data"]),
            dataset_meta=kwargs["dataset_meta"],
            window_plan=kwargs["window_plan"],
            preview_count=1,
            final_count=0,
            effective_preview=1,
            effective_final=0,
            preview_mask_counts=[0],
            final_mask_counts=[],
            preview_records=[runtime_record],
            final_records=[],
        )

    result = materialize_run_dataset(
        pairing_schedule={
            "preview": {
                "example_ids": ["ex-1"],
                "records": [{"id": "ex-1", "image_sha256": "a" * 64}],
            },
            "final": {"example_ids": [], "records": []},
        },
        cfg=SimpleNamespace(dataset=SimpleNamespace(provider="vision_text")),
        baseline_report_data={"evaluation_windows": {}},
        tokenizer_hash=None,
        resolved_loss_type="classification",
        profile="release",
        model_profile=SimpleNamespace(),
        tokenizer=None,
        use_mlm=False,
        mask_prob=0.0,
        mask_seed=1,
        random_token_prob=0.0,
        original_token_prob=0.0,
        tier="balanced",
        requested_preview=1,
        requested_final=0,
        effective_preview=1,
        effective_final=0,
        resolved_device="cuda",
        profile_normalized="release",
        resolved_split=None,
        validate_and_harvest_baseline_schedule_fn=lambda *args, **kwargs: {
            "dataset_meta": {"provider_kind": "vision_text"},
            "window_plan": {"actual_preview": 1, "actual_final": 0},
            "calibration_data": [],
        },
        materialize_baseline_pairing_schedule_fn=materialize,
        resolve_tokenizer_fn=lambda *args, **kwargs: (None, None),
        build_provider_dataset_plan_fn=lambda **kwargs: SimpleNamespace(
            resolved_split="validation",
            used_fallback_split=False,
            tokenizer=None,
            tokenizer_hash=None,
            calibration_data=[runtime_record],
            dataset_meta={"provider_kind": "vision_text"},
            window_plan={"actual_preview": 1, "actual_final": 0},
            diagnostics=(),
        ),
    )

    assert observed["calibration_data"] == [runtime_record]
    assert result.resolved_split == "validation"
    assert result.calibration_data[0]["image_path"] == "/runtime/ex-1.ppm"


def test_materialize_run_dataset_returns_passthrough_defaults_without_provider() -> (
    None
):
    result = materialize_run_dataset(
        pairing_schedule=None,
        cfg=SimpleNamespace(dataset=SimpleNamespace(provider=None)),
        baseline_report_data=None,
        tokenizer_hash="tok-hash",
        resolved_loss_type="ppl_causal",
        profile="dev",
        model_profile=SimpleNamespace(),
        tokenizer="tok",
        use_mlm=False,
        mask_prob=0.0,
        mask_seed=1,
        random_token_prob=0.0,
        original_token_prob=0.0,
        tier=None,
        requested_preview=3,
        requested_final=4,
        effective_preview=3,
        effective_final=4,
        resolved_device="cpu",
        profile_normalized="dev",
        resolved_split="validation",
        validate_and_harvest_baseline_schedule_fn=lambda *args, **kwargs: None,
        materialize_baseline_pairing_schedule_fn=lambda *args, **kwargs: None,
        resolve_tokenizer_fn=lambda *args, **kwargs: ("tok", "tok-hash"),
        build_provider_dataset_plan_fn=lambda **kwargs: None,
    )

    assert result.resolved_split == "validation"
    assert result.preview_count == 3
    assert result.final_count == 4
    assert result.dataset_meta == {}
    assert result.preview_records == []
    assert result.final_records == []


def test_materialize_run_dataset_requests_typed_baseline_schedule_failures() -> None:
    observed: dict[str, object] = {}

    def _raise_typed_failure(*args, **kwargs):
        observed["typed_failures"] = kwargs.get("typed_failures")
        raise InvarlockError(code="E001", message="PAIRING-EVIDENCE-MISSING")

    with pytest.raises(InvarlockError):
        materialize_run_dataset(
            pairing_schedule={"preview": {}, "final": {}},
            cfg=SimpleNamespace(dataset=SimpleNamespace(provider=None)),
            baseline_report_data={"evaluation_windows": {}},
            tokenizer_hash="tok",
            resolved_loss_type="ppl_mlm",
            profile="dev",
            model_profile=SimpleNamespace(),
            tokenizer=None,
            use_mlm=True,
            mask_prob=0.15,
            mask_seed=43,
            random_token_prob=0.1,
            original_token_prob=0.1,
            tier="balanced",
            requested_preview=1,
            requested_final=1,
            effective_preview=1,
            effective_final=1,
            resolved_device="cpu",
            profile_normalized="dev",
            resolved_split=None,
            validate_and_harvest_baseline_schedule_fn=_raise_typed_failure,
            materialize_baseline_pairing_schedule_fn=lambda **kwargs: None,
            resolve_tokenizer_fn=lambda profile: ("tok", "tok-hash"),
            build_provider_dataset_plan_fn=lambda **kwargs: None,
        )

    assert observed["typed_failures"] is True
