from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core.run_provider_dataset_plan import (
    ProviderDatasetPlanDiagnostic,
    _build_provider_kwargs,
    _build_signature_transform,
    _cfg_auto_tier,
    _hash_texts,
    _optional_text,
    _resolve_release_window_plan,
    _section_dict,
    _section_value,
)
from invarlock.eval.window_planning import resolve_effective_windows
from tests.core._support_run_provider_dataset_plan import (
    _DummyTokenizer,
    _ProviderConfig,
)


def test_provider_dataset_plan_section_helpers_cover_fallback_paths() -> None:
    class _Section:
        value = "fallback"

        def get(self, _key: str) -> object:
            raise TypeError("mapping disabled")

    assert _section_value(_Section(), "value") == "fallback"
    assert _section_value(_Section(), "missing") is None

    class _Config:
        dataset = {"name": "demo"}
        text = SimpleNamespace(value="ns", _private="drop")

        def section(self, _name: str) -> object:
            raise KeyError("boom")

    assert _section_dict(_Config(), "dataset") == {"name": "demo"}
    assert _section_dict(_Config(), "text") == {"value": "ns"}


def test_cfg_auto_tier_falls_back_when_section_lookup_fails() -> None:
    class _Config:
        auto = SimpleNamespace(tier="release")

        def section(self, _name: str) -> object:
            raise KeyError("missing")

    assert _cfg_auto_tier(_Config()) == "release"


def test_provider_dataset_plan_small_helpers_cover_blank_text_and_provider_overrides() -> (
    None
):
    cfg_dataset = SimpleNamespace(
        dataset_name="demo",
        config_name="",
        text_field="text",
        cache_dir=None,
        provider=_ProviderConfig("local_jsonl", path="demo.jsonl", data_files=""),
    )

    assert _build_provider_kwargs(cfg_dataset) == {
        "dataset_name": "demo",
        "text_field": "text",
        "path": "demo.jsonl",
    }
    assert _optional_text("  keep  ") == "keep"
    assert _optional_text("   ") is None


def test_provider_dataset_plan_helper_variants_cover_section_and_hash_edges() -> None:
    class _Config:
        as_map = {"name": "demo"}
        as_ns = SimpleNamespace(value="ns", _private="drop")
        as_scalar = 7

        def section(self, name: str) -> object:
            if name == "section_map":
                return {"kind": "direct"}
            if name == "as_map":
                return SimpleNamespace(kind="not-a-dict")
            if name == "missing":
                return None
            raise KeyError("fallback")

    cfg = _Config()

    assert _section_dict(cfg, "section_map") == {"kind": "direct"}
    assert _section_dict(cfg, "as_map") == {"name": "demo"}
    assert _section_dict(cfg, "as_ns") == {"value": "ns"}
    assert _section_dict(cfg, "as_scalar") == {}
    assert _section_dict(cfg, "missing") == {}
    assert _optional_text(None) is None
    assert _hash_texts(["a", "b"]) == _hash_texts(["a", "b"])
    assert _hash_texts(["a", "b"]) != _hash_texts(["b", "a"])


def test_resolve_release_window_plan_helper_variants() -> None:
    diagnostics: list[ProviderDatasetPlanDiagnostic] = []
    tokenizer = _DummyTokenizer()

    window_plan, preview_n, final_n = _resolve_release_window_plan(
        data_provider=SimpleNamespace(),
        eval_section={},
        guards_section={},
        cfg_dataset=SimpleNamespace(seq_len=8, stride=4),
        resolved_split="validation",
        tokenizer=tokenizer,
        requested_preview=2,
        requested_final=3,
        profile="dev",
        pairing_schedule_present=False,
        maybe_plan_release_windows_fn=lambda *args, **kwargs: pytest.fail(
            "release planner should not run"
        ),
        diagnostics=diagnostics,
    )

    assert window_plan is None
    assert preview_n == 2
    assert final_n == 3

    captured: dict[str, object] = {}

    def _estimate_capacity(**kwargs: object) -> dict[str, object]:
        captured["capacity_kwargs"] = kwargs
        return {"available_examples": 12}

    def _plan_windows(
        capacity_meta: dict[str, object], **kwargs: object
    ) -> dict[str, object]:
        captured["plan_kwargs"] = kwargs
        return {"actual_preview": 4, "capacity": capacity_meta}

    release_plan, preview_n, final_n = _resolve_release_window_plan(
        data_provider=SimpleNamespace(estimate_capacity=_estimate_capacity),
        eval_section={"capacity_fast": True},
        guards_section={"variance": {"max_calib": 7}},
        cfg_dataset=SimpleNamespace(seq_len=8, stride=3),
        resolved_split="validation",
        tokenizer=tokenizer,
        requested_preview=2,
        requested_final=5,
        profile="release",
        pairing_schedule_present=False,
        maybe_plan_release_windows_fn=_plan_windows,
        diagnostics=[],
    )

    assert captured["capacity_kwargs"] == {
        "tokenizer": tokenizer,
        "seq_len": 8,
        "stride": 3,
        "split": "validation",
        "target_total": 7,
        "fast_mode": True,
    }
    assert captured["plan_kwargs"] == {
        "requested_preview": 2,
        "requested_final": 5,
        "max_calibration": 7,
    }
    assert release_plan == {"actual_preview": 4, "capacity": {"available_examples": 12}}
    assert preview_n == 4
    assert final_n == 4


def test_build_signature_transform_handles_disabled_and_clone_paths() -> None:
    assert (
        _build_signature_transform(
            use_mlm=False,
            tokenizer=_DummyTokenizer(),
            mask_prob=0.15,
            mask_seed=43,
            random_token_prob=0.1,
            original_token_prob=0.1,
            apply_mlm_masks_fn=lambda *args, **kwargs: (0, []),
        )
        is None
    )

    calls: list[str] = []

    def _apply_masks(
        records: list[dict[str, object]],
        *,
        prefix: str,
        **_: object,
    ) -> tuple[int, list[int]]:
        calls.append(prefix)
        records[0]["labels"] = [99]
        records[0]["input_ids"][0] = -1
        return 1, [1]

    transform = _build_signature_transform(
        use_mlm=True,
        tokenizer=_DummyTokenizer(),
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        apply_mlm_masks_fn=_apply_masks,
    )
    assert transform is not None

    preview_records = [
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "dataset_index": 1,
            "window_id": "preview::0",
        }
    ]
    final_records = [
        {
            "input_ids": [3, 4],
            "attention_mask": [1, 1],
            "dataset_index": 2,
            "window_id": "final::0",
        }
    ]

    transformed = transform(preview_records, final_records)

    assert calls == ["preview", "final"]
    assert preview_records[0]["input_ids"] == [1, 2]
    assert final_records[0]["input_ids"] == [3, 4]
    assert transformed[0]["input_ids"] == [1, 2]
    assert transformed[1]["input_ids"] == [3, 4]
    assert transformed[0]["labels"] == [99]
    assert transformed[1]["labels"] == [99]
    assert transformed[0]["window_id"] == "preview::0"
    assert transformed[1]["window_id"] == "final::0"


def test_build_signature_transform_preserves_raw_window_identity_for_mlm_dedupe() -> (
    None
):
    class _Provider:
        def windows(
            self, *, preview_n: int, final_n: int, **_: object
        ) -> tuple[SimpleNamespace, SimpleNamespace]:
            preview_ids = [
                [100 + idx, 101 + idx, 102 + idx] for idx in range(preview_n)
            ]
            final_ids = [[200 + idx, 201 + idx, 202 + idx] for idx in range(final_n)]
            if preview_n >= 20 and final_n >= 20:
                final_ids[-1] = list(preview_ids[0])
            masks = [[1, 1, 1] for _ in range(preview_n)]
            return (
                SimpleNamespace(input_ids=preview_ids, attention_masks=masks),
                SimpleNamespace(
                    input_ids=final_ids,
                    attention_masks=[[1, 1, 1] for _ in range(final_n)],
                ),
            )

    def _apply_masks(
        records: list[dict[str, object]],
        *,
        prefix: str,
        **_: object,
    ) -> tuple[int, list[int]]:
        marker = -1 if prefix == "preview" else -2
        records[-1]["input_ids"][0] = marker
        records[-1]["labels"] = [7, -100, -100]
        return 1, [1] + [0] * (len(records) - 1)

    transform = _build_signature_transform(
        use_mlm=True,
        tokenizer=_DummyTokenizer(),
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        apply_mlm_masks_fn=_apply_masks,
    )

    result = resolve_effective_windows(
        data_provider=_Provider(),
        tokenizer=_DummyTokenizer(),
        seq_len=8,
        stride=8,
        preview_n=20,
        final_n=20,
        seed=43,
        split="validation",
        requested_preview=20,
        requested_final=20,
        profile="dev",
        signature_transform=transform,
    )

    assert result["actual_preview"] == 15
    assert result["actual_final"] == 15
    assert result["dedupe_adjustments"] == [{"deficit": 1, "proposed_per_arm": 15}]
