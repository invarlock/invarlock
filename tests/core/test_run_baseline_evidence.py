from __future__ import annotations

import json
from pathlib import Path

from invarlock.core.run_baseline_evidence import (
    load_baseline_pairing_evidence,
    materialize_baseline_pairing_schedule,
)


def _extract_pairing_schedule(report: dict | None) -> dict | None:
    if not isinstance(report, dict):
        return None
    return report.get("pairing_schedule")


def test_load_baseline_pairing_evidence_missing_path(tmp_path: Path) -> None:
    result = load_baseline_pairing_evidence(
        baseline_path=tmp_path / "missing.json",
        tokenizer_hash=None,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "missing_path"
    assert result.report_data is None
    assert result.pairing_schedule is None
    assert "PAIRING-EVIDENCE-MISSING" in str(result.message)


def test_load_baseline_pairing_evidence_parse_failure(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{not-json", encoding="utf-8")

    result = load_baseline_pairing_evidence(
        baseline_path=baseline,
        tokenizer_hash=None,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "parse_failed"
    assert result.report_data is None
    assert result.pairing_schedule is None
    assert "JSON parse failed" in str(result.message)


def test_load_baseline_pairing_evidence_invalid_schedule(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"meta": {"tokenizer_hash": "tok"}}), encoding="utf-8"
    )

    result = load_baseline_pairing_evidence(
        baseline_path=baseline,
        tokenizer_hash=None,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "missing_schedule"
    assert result.report_data is None
    assert result.pairing_schedule is None
    assert "missing or invalid evaluation_windows" in str(result.message)


def test_load_baseline_pairing_evidence_invalid_report_preserves_explicit_hash(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(["not-a-dict"]), encoding="utf-8")

    result = load_baseline_pairing_evidence(
        baseline_path=baseline,
        tokenizer_hash="explicit-hash",
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "invalid_report"
    assert result.tokenizer_hash == "explicit-hash"
    assert result.report_data is None
    assert result.pairing_schedule is None


def test_load_baseline_pairing_evidence_merges_schedule_and_harvests_hash(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {"logloss": [0.1]},
                    "final": {"token_counts": [3]},
                },
                "pairing_schedule": {
                    "preview": {"window_ids": [1], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [2], "input_ids": [[4, 5, 6]]},
                },
            }
        ),
        encoding="utf-8",
    )

    result = load_baseline_pairing_evidence(
        baseline_path=baseline,
        tokenizer_hash=None,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "loaded"
    assert result.tokenizer_hash == "tokhash123"
    assert result.pairing_schedule == {
        "preview": {"window_ids": [1], "input_ids": [[1, 2, 3]]},
        "final": {"window_ids": [2], "input_ids": [[4, 5, 6]]},
    }
    assert result.report_data is not None
    assert result.report_data["evaluation_windows"]["preview"]["window_ids"] == [1]
    assert result.report_data["evaluation_windows"]["preview"]["logloss"] == [0.1]
    assert result.report_data["evaluation_windows"]["final"]["token_counts"] == [3]


def test_load_baseline_pairing_evidence_builds_missing_windows_and_uses_data_hash(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "data": {"tokenizer_hash": "tokhash-from-data"},
                "pairing_schedule": {
                    "preview": {"window_ids": [7], "input_ids": [[1, 2, 3]]},
                    "final": "skip-me",
                },
            }
        ),
        encoding="utf-8",
    )

    result = load_baseline_pairing_evidence(
        baseline_path=baseline,
        tokenizer_hash=None,
        extract_pairing_schedule_fn=_extract_pairing_schedule,
    )

    assert result.status == "loaded"
    assert result.tokenizer_hash == "tokhash-from-data"
    assert result.report_data is not None
    assert result.report_data["evaluation_windows"] == {
        "preview": {"window_ids": [7], "input_ids": [[1, 2, 3]]}
    }


def test_materialize_baseline_pairing_schedule_preserves_mask_counts_and_hashes() -> (
    None
):
    result = materialize_baseline_pairing_schedule(
        pairing_schedule={
            "preview": {
                "window_ids": [10],
                "input_ids": [[1, 2]],
                "attention_masks": [[1, 1]],
                "labels": [[-100, 9]],
                "masked_token_counts": [1],
            },
            "final": {
                "window_ids": [20],
                "input_ids": [[3, 4, 5]],
                "attention_masks": [[1, 1, 1]],
                "labels": [[8, -100, -100]],
                "masked_token_counts": [1],
            },
        },
        calibration_data=[],
        dataset_meta={},
        window_plan=None,
        tokenizer=object(),
        use_mlm=True,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        resolved_tier="balanced",
        profile="ci",
        apply_mlm_masks_fn=lambda *args, **kwargs: (0, []),
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 4,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tensor_or_list_to_ints_fn=lambda values: list(values),
    )

    assert result.preview_count == 1
    assert result.final_count == 1
    assert result.calibration_data[0]["window_id"] == "preview::10"
    assert result.calibration_data[1]["window_id"] == "final::20"
    assert result.preview_mask_counts == [1]
    assert result.final_mask_counts == [1]
    assert result.dataset_meta["dataset_hash"] == "a538ec9757cc6936907c9089cfa0209a"
    assert result.window_plan is not None
    assert result.window_plan["tokens_floor_met"] is True
    assert result.dataset_meta["masked_tokens_total"] == 2


def test_materialize_baseline_pairing_schedule_fails_when_masks_missing() -> None:
    try:
        materialize_baseline_pairing_schedule(
            pairing_schedule={
                "preview": {
                    "input_ids": [[1, 2]],
                    "attention_masks": [[1, 1]],
                },
                "final": {
                    "input_ids": [[3, 4]],
                    "attention_masks": [[1, 1]],
                },
            },
            calibration_data=[],
            dataset_meta={},
            window_plan=None,
            tokenizer=object(),
            use_mlm=True,
            mask_prob=0.15,
            mask_seed=43,
            random_token_prob=0.1,
            original_token_prob=0.1,
            resolved_tier="balanced",
            profile="dev",
            apply_mlm_masks_fn=lambda *args, **kwargs: (0, [0]),
            resolve_pm_min_tokens_target_fn=lambda **kwargs: 4,
            hash_sequences_fn=lambda seqs: "hash",
            tensor_or_list_to_ints_fn=lambda values: list(values),
        )
    except ValueError as exc:
        assert "provided no masked tokens for preview windows" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")


def test_materialize_baseline_pairing_schedule_applies_missing_preview_masks_and_preserves_existing_meta() -> (
    None
):
    applied_prefixes: list[str] = []
    dataset_meta = {
        "preview_hash": "existing-preview",
        "final_hash": "existing-final",
        "dataset_hash": "existing-dataset",
        "preview_total_tokens": 11,
        "final_total_tokens": 13,
    }
    window_plan = {"capacity": {"estimated": 4}}

    result = materialize_baseline_pairing_schedule(
        pairing_schedule={
            "preview": {
                "input_ids": [[1, 2]],
                "attention_masks": [[0, 0]],
                "labels": [[-100, -100]],
                "masked_token_counts": ["bad-count"],
            },
            "final": {
                "window_ids": [9],
                "input_ids": [[3, 4]],
                "attention_masks": [[1, 1]],
                "labels": [[7, -100]],
                "masked_token_counts": [1],
            },
        },
        calibration_data=[],
        dataset_meta=dataset_meta,
        window_plan=window_plan,
        tokenizer=object(),
        use_mlm=True,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        resolved_tier="balanced",
        profile="release",
        apply_mlm_masks_fn=lambda records, **kwargs: (
            applied_prefixes.append(str(kwargs["prefix"])) or 2,
            [2],
        ),
        resolve_pm_min_tokens_target_fn=lambda **kwargs: 10,
        hash_sequences_fn=lambda seqs: f"hash-{len(list(seqs))}",
        tensor_or_list_to_ints_fn=lambda values: list(values),
    )

    assert applied_prefixes == ["preview"]
    assert result.preview_mask_counts == [2]
    assert result.final_mask_counts == [1]
    assert result.preview_mask_total == 2
    assert result.final_mask_total == 1
    assert result.calibration_data[1]["window_id"] == "final::9"
    assert result.dataset_meta["preview_hash"] == "existing-preview"
    assert result.dataset_meta["final_hash"] == "existing-final"
    assert result.dataset_meta["dataset_hash"] == "existing-dataset"
    assert result.dataset_meta["window_capacity"] == {"estimated": 4}
    assert result.window_plan is window_plan
    assert result.window_plan["preview_total_tokens"] == 11
    assert result.window_plan["final_total_tokens"] == 2


def test_materialize_baseline_pairing_schedule_fails_when_final_masks_missing() -> None:
    try:
        materialize_baseline_pairing_schedule(
            pairing_schedule={
                "preview": {
                    "input_ids": [[1, 2]],
                    "attention_masks": [[1, 1]],
                    "labels": [[9, -100]],
                    "masked_token_counts": [1],
                },
                "final": {
                    "input_ids": [[3, 4]],
                    "attention_masks": [[1, 1]],
                },
            },
            calibration_data=[],
            dataset_meta={},
            window_plan=None,
            tokenizer=object(),
            use_mlm=True,
            mask_prob=0.15,
            mask_seed=43,
            random_token_prob=0.1,
            original_token_prob=0.1,
            resolved_tier="balanced",
            profile="dev",
            apply_mlm_masks_fn=lambda *args, **kwargs: (0, [0]),
            resolve_pm_min_tokens_target_fn=lambda **kwargs: 4,
            hash_sequences_fn=lambda seqs: "hash",
            tensor_or_list_to_ints_fn=lambda values: list(values),
        )
    except ValueError as exc:
        assert "provided no masked tokens for final windows" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")
