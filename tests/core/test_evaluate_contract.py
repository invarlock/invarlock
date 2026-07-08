from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import invarlock.core.evaluate_contract as evaluate_contract_mod
from invarlock.core.evaluate_contract import (
    apply_edited_primary_metric_policy,
    load_validated_baseline_report,
    require_run_report_artifact,
)
from invarlock.core.exceptions import ConfigError, ValidationError
from invarlock.core.report_inputs import ReportInputError


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _baseline_payload(
    *,
    model_id: str = "baseline-model",
    adapter: str = "hf_causal",
    profile: str = "dev",
    tier: str = "balanced",
    assurance_mode: str = "off",
    data: dict[str, object] | None = None,
    edit_name: str = "noop",
    evaluation_windows: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "edit": {"name": edit_name},
        "meta": {"model_id": model_id, "adapter": adapter},
        "context": {
            "profile": profile,
            "auto": {"tier": tier},
            "assurance": {"mode": assurance_mode},
        },
        "data": data
        or {
            "provider": "wikitext2",
            "split": "validation",
            "seq_len": 512,
            "stride": 512,
            "preview_n": 64,
            "final_n": 64,
            "seed": 43,
        },
        "evaluation_windows": evaluation_windows
        or {
            "preview": {"window_ids": ["preview-0"], "input_ids": [[1, 2, 3]]},
            "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
        },
    }


def _baseline_validation_kwargs(adapter: str = "hf_causal") -> dict[str, object]:
    return {
        "expected_model_id": "baseline-model",
        "expected_profile": "dev",
        "expected_tier": "balanced",
        "expected_adapter": adapter,
        "expected_assurance_mode": "off",
        "expected_dataset": {
            "provider": "wikitext2",
            "split": "validation",
            "seq_len": 512,
            "stride": 512,
            "preview_n": 64,
            "final_n": 64,
            "seed": 43,
        },
    }


def test_require_run_report_artifact_accepts_explicit_file(tmp_path: Path) -> None:
    report = _write_json(tmp_path / "report.json", {"ok": True})

    resolved = require_run_report_artifact(str(report), stage="Baseline")

    assert resolved == report.resolve()


def test_require_run_report_artifact_rejects_missing_or_non_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="did not return a report path"):
        require_run_report_artifact(None, stage="Baseline")

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    with pytest.raises(ConfigError, match="returned a non-file report path"):
        require_run_report_artifact(report_dir, stage="Edited")


def test_load_validated_baseline_report_accepts_valid_explicit_file(
    tmp_path: Path,
) -> None:
    report = _write_json(tmp_path / "report.json", _baseline_payload())

    resolved, payload = load_validated_baseline_report(
        report,
        **_baseline_validation_kwargs(),
    )

    assert resolved == report.resolve()
    assert payload["edit"] == {"name": "noop"}


def test_expected_baseline_value_matches_provider_kind_edge_cases() -> None:
    matcher = evaluate_contract_mod._baseline_dataset_value_matches  # noqa: SLF001

    assert matcher("provider", {"kind": "hf"}, {"provider": "hf"}) is True
    assert matcher("provider", {"provider": "hf"}, {"dataset": "hf"}) is True
    assert matcher("provider", {"kind": ""}, {"provider": "hf"}) is False
    assert matcher("provider", {"unexpected": "hf"}, {"provider": "hf"}) is False
    assert matcher("provider", object(), {"provider": "hf"}) is False


def test_load_validated_baseline_report_accepts_multimodal_baseline_windows(
    tmp_path: Path,
) -> None:
    report = _write_json(
        tmp_path / "vision-report.json",
        _baseline_payload(
            adapter="hf_multimodal",
            evaluation_windows={
                "preview": {
                    "example_ids": ["red-square"],
                    "records": [{"id": "red-square", "correct": True}],
                },
                "final": {
                    "example_ids": ["green-square"],
                    "records": [{"id": "green-square", "correct": True}],
                },
            },
        ),
    )

    resolved, payload = load_validated_baseline_report(
        report,
        **_baseline_validation_kwargs("hf_multimodal"),
    )

    assert resolved == report.resolve()
    assert payload["meta"] == {
        "model_id": "baseline-model",
        "adapter": "hf_multimodal",
    }


def test_load_validated_baseline_report_rejects_context_without_tier(
    tmp_path: Path,
) -> None:
    report = _write_json(
        tmp_path / "baseline.json",
        {
            **_baseline_payload(),
            "context": {
                "profile": "dev",
                "auto": "skip-tier-check",
                "assurance": {"mode": "off"},
            },
        },
    )

    with pytest.raises(ValidationError, match="tier mismatch"):
        load_validated_baseline_report(report, **_baseline_validation_kwargs())


def test_load_validated_baseline_report_accepts_legacy_identity_locations(
    tmp_path: Path,
) -> None:
    kwargs = _baseline_validation_kwargs()

    context_tier = _write_json(
        tmp_path / "context-tier.json",
        {
            **_baseline_payload(),
            "context": {
                "profile": "dev",
                "tier": "balanced",
                "assurance": {"mode": "off"},
            },
        },
    )
    resolved, payload = load_validated_baseline_report(context_tier, **kwargs)
    assert resolved == context_tier.resolve()
    assert payload["context"]["tier"] == "balanced"

    top_level_auto = _write_json(
        tmp_path / "top-level-auto.json",
        {
            **_baseline_payload(),
            "context": {"profile": "dev", "assurance": {"mode": "off"}},
            "auto": {"tier": "balanced"},
        },
    )
    resolved, payload = load_validated_baseline_report(top_level_auto, **kwargs)
    assert resolved == top_level_auto.resolve()
    assert payload["auto"]["tier"] == "balanced"

    meta_auto = _write_json(
        tmp_path / "meta-auto.json",
        {
            **_baseline_payload(),
            "meta": {
                "model_id": "baseline-model",
                "adapter": "hf_causal",
                "auto": {"tier": "balanced"},
            },
            "context": {"profile": "dev", "assurance": {"mode": "off"}},
        },
    )
    resolved, payload = load_validated_baseline_report(meta_auto, **kwargs)
    assert resolved == meta_auto.resolve()
    assert payload["meta"]["auto"]["tier"] == "balanced"

    top_level_assurance = _write_json(
        tmp_path / "top-level-assurance.json",
        {
            **_baseline_payload(),
            "context": {"profile": "dev", "auto": {"tier": "balanced"}},
            "assurance": {"mode": "off"},
        },
    )
    resolved, payload = load_validated_baseline_report(top_level_assurance, **kwargs)
    assert resolved == top_level_assurance.resolve()
    assert payload["assurance"]["mode"] == "off"


def test_load_validated_baseline_report_accepts_path_equivalent_model_ids(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    report = _write_json(
        tmp_path / "baseline.json",
        _baseline_payload(model_id=str(model_dir)),
    )
    kwargs = {
        **_baseline_validation_kwargs(),
        "expected_model_id": str(model_dir.resolve()),
    }

    resolved, payload = load_validated_baseline_report(report, **kwargs)
    assert resolved == report.resolve()
    assert payload["meta"]["model_id"] == str(model_dir)


def test_load_validated_baseline_report_rejects_identity_mismatches(
    tmp_path: Path,
) -> None:
    wrong_model = _write_json(
        tmp_path / "wrong-model.json",
        _baseline_payload(model_id="other-model"),
    )
    with pytest.raises(ValidationError, match="model mismatch"):
        load_validated_baseline_report(wrong_model, **_baseline_validation_kwargs())

    wrong_assurance = _write_json(
        tmp_path / "wrong-assurance.json",
        _baseline_payload(assurance_mode="strict"),
    )
    with pytest.raises(ValidationError, match="assurance mode mismatch"):
        load_validated_baseline_report(
            wrong_assurance,
            **_baseline_validation_kwargs(),
        )

    missing_assurance = _write_json(
        tmp_path / "missing-assurance.json",
        {
            **_baseline_payload(),
            "context": {"profile": "dev", "auto": {"tier": "balanced"}},
        },
    )
    with pytest.raises(ValidationError, match="assurance mode mismatch"):
        load_validated_baseline_report(
            missing_assurance,
            **_baseline_validation_kwargs(),
        )

    wrong_dataset_payload = _baseline_payload()
    data = wrong_dataset_payload["data"]
    assert isinstance(data, dict)
    wrong_dataset_payload["data"] = {**data, "seed": 999}
    wrong_dataset = _write_json(tmp_path / "wrong-dataset.json", wrong_dataset_payload)
    with pytest.raises(ValidationError, match="dataset/window-plan mismatch"):
        load_validated_baseline_report(wrong_dataset, **_baseline_validation_kwargs())


def test_load_validated_baseline_report_rejects_missing_context_and_data(
    tmp_path: Path,
) -> None:
    missing_context = _write_json(
        tmp_path / "missing-context.json",
        {**_baseline_payload(), "context": None},
    )
    with pytest.raises(ValidationError, match="missing context object"):
        load_validated_baseline_report(missing_context, **_baseline_validation_kwargs())

    missing_data = _write_json(
        tmp_path / "missing-data.json",
        {**_baseline_payload(), "data": None},
    )
    with pytest.raises(ValidationError, match="missing data object"):
        load_validated_baseline_report(missing_data, **_baseline_validation_kwargs())


def test_load_validated_baseline_report_accepts_partial_expected_dataset(
    tmp_path: Path,
) -> None:
    report = _write_json(tmp_path / "baseline.json", _baseline_payload())
    kwargs = {
        **_baseline_validation_kwargs(),
        "expected_dataset": {"provider": "wikitext2"},
    }

    resolved, payload = load_validated_baseline_report(report, **kwargs)
    assert resolved == report.resolve()
    assert payload["data"]["provider"] == "wikitext2"


def test_load_validated_baseline_report_accepts_missing_seed_with_windows(
    tmp_path: Path,
) -> None:
    payload = _baseline_payload()
    data = payload["data"]
    assert isinstance(data, dict)
    payload["data"] = {key: value for key, value in data.items() if key != "seed"}
    report = _write_json(tmp_path / "baseline.json", payload)

    load_validated_baseline_report(report, **_baseline_validation_kwargs())


def test_load_validated_baseline_report_accepts_hf_provider_kind_report(
    tmp_path: Path,
) -> None:
    report = _write_json(
        tmp_path / "baseline.json",
        _baseline_payload(
            data={
                "dataset": "hf_text",
                "split": "train",
                "seq_len": 512,
                "stride": 512,
                "preview_n": 400,
                "final_n": 400,
                "seed": 42,
            }
        ),
    )
    kwargs = {
        **_baseline_validation_kwargs(),
        "expected_dataset": {
            "provider": {
                "kind": "hf_text",
                "dataset_name": "Salesforce/wikitext",
                "config_name": "wikitext-103-v1",
                "text_field": "text",
                "max_samples": 10000,
            },
            "split": "train",
            "seq_len": 512,
            "stride": 512,
            "preview_n": 400,
            "final_n": 400,
            "seed": 42,
        },
    }

    resolved, payload = load_validated_baseline_report(report, **kwargs)
    assert resolved == report.resolve()
    assert payload["data"]["dataset"] == "hf_text"
    assert payload["data"]["split"] == "train"


def test_load_validated_baseline_report_rejects_hf_provider_kind_mismatch(
    tmp_path: Path,
) -> None:
    report = _write_json(
        tmp_path / "baseline.json",
        _baseline_payload(data={**_baseline_payload()["data"], "dataset": "hf_text"}),
    )
    kwargs = {
        **_baseline_validation_kwargs(),
        "expected_dataset": {
            **_baseline_validation_kwargs()["expected_dataset"],
            "provider": {"kind": "hf_image"},
        },
    }

    with pytest.raises(ValidationError, match="dataset/window-plan mismatch"):
        load_validated_baseline_report(report, **kwargs)


def test_model_id_equivalence_returns_false_when_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    original_resolve = Path.resolve

    def _raise_for_model_dir(self: Path, *args: object, **kwargs: object) -> Path:
        if self == model_dir:
            raise OSError("cannot resolve")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _raise_for_model_dir)

    assert (
        evaluate_contract_mod._model_ids_equivalent(str(model_dir), "other-model")
        is False
    )


def test_load_validated_baseline_report_rejects_directory_and_bad_edit(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    with pytest.raises(
        ValidationError, match="must be an explicit report.json file path"
    ):
        load_validated_baseline_report(
            report_dir,
            **_baseline_validation_kwargs(),
        )

    bad_edit = _write_json(
        tmp_path / "baseline.json",
        _baseline_payload(edit_name="quant_rtn"),
    )
    with pytest.raises(ValidationError, match="must be a no-op run"):
        load_validated_baseline_report(
            bad_edit,
            **_baseline_validation_kwargs(),
        )


def test_load_validated_baseline_report_rejects_policy_and_window_mismatches(
    tmp_path: Path,
) -> None:
    mismatched = _write_json(
        tmp_path / "baseline.json",
        _baseline_payload(profile="release"),
    )
    with pytest.raises(ValidationError, match="profile mismatch"):
        load_validated_baseline_report(
            mismatched,
            **_baseline_validation_kwargs(),
        )

    bad_windows = _write_json(
        tmp_path / "bad-windows.json",
        _baseline_payload(
            evaluation_windows={
                "preview": {"window_ids": ["preview-0"], "input_ids": [[1, 2, 3]]},
                "final": {"window_ids": ["final-0", "final-1"], "input_ids": [[4]]},
            }
        ),
    )
    with pytest.raises(ValidationError, match="inconsistent evaluation window"):
        load_validated_baseline_report(
            bad_windows,
            **_baseline_validation_kwargs(),
        )

    missing_records = _write_json(
        tmp_path / "missing-records.json",
        _baseline_payload(
            adapter="hf_multimodal",
            evaluation_windows={
                "preview": {"example_ids": ["ex-1"]},
                "final": {
                    "example_ids": ["ex-2"],
                    "records": [{"id": "ex-2", "correct": True}],
                },
            },
        ),
    )
    with pytest.raises(
        ValidationError, match="missing evaluation_windows.preview.records"
    ):
        load_validated_baseline_report(
            missing_records,
            **_baseline_validation_kwargs("hf_multimodal"),
        )

    inconsistent_records = _write_json(
        tmp_path / "inconsistent-records.json",
        _baseline_payload(
            adapter="hf_multimodal",
            evaluation_windows={
                "preview": {
                    "example_ids": ["ex-1"],
                    "records": [{"id": "ex-1", "correct": True}],
                },
                "final": {
                    "example_ids": ["ex-2", "ex-3"],
                    "records": [{"id": "ex-2", "correct": True}],
                },
            },
        ),
    )
    with pytest.raises(
        ValidationError, match="inconsistent multimodal evaluation payloads"
    ):
        load_validated_baseline_report(
            inconsistent_records,
            **_baseline_validation_kwargs("hf_multimodal"),
        )


def test_load_validated_baseline_report_missing_windows_has_no_env_remediation(
    tmp_path: Path,
) -> None:
    missing_windows = _write_json(
        tmp_path / "missing-windows.json",
        _baseline_payload(evaluation_windows={"preview": {}, "final": {}}),
    )
    with pytest.raises(ValidationError) as excinfo:
        load_validated_baseline_report(
            missing_windows,
            **_baseline_validation_kwargs(),
        )
    assert "INVARLOCK_STORE_EVAL_WINDOWS" not in str(excinfo.value)


def test_load_validated_baseline_report_rejects_non_regular_file(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo unavailable on this platform")

    fifo = tmp_path / "baseline.pipe"
    os.mkfifo(fifo)

    with pytest.raises(ValidationError, match="Baseline report not found"):
        load_validated_baseline_report(
            fifo,
            **_baseline_validation_kwargs(),
        )


def test_apply_edited_primary_metric_policy_does_not_carry_exit_code() -> None:
    outcome = apply_edited_primary_metric_policy(
        {
            "meta": {"device": "cpu", "adapter": "hf_causal"},
            "edit": {"name": "quant_rtn"},
            "metrics": {"primary_metric": {"final": {"bad": "value"}}},
        },
        profile="ci",
    )

    assert outcome.error is not None
    assert outcome.error.code == "E111"
    assert outcome.diagnostic is not None
    assert outcome.diagnostic.code == "evaluate.primary_metric_degraded"
    assert outcome.diagnostic.details["reason"] == "non_finite_pm"
    assert outcome.payload["metrics"]["primary_metric"]["final"] == {"bad": "value"}
    assert "ratio_vs_baseline" not in outcome.payload["metrics"]["primary_metric"]
    assert not hasattr(outcome, "warning")
    assert not hasattr(outcome, "exit_code")


def test_apply_edited_primary_metric_policy_reraises_unexpected_profile_errors() -> (
    None
):
    class _BadProfile:
        def __str__(self) -> str:
            raise AssertionError("explode")

    with pytest.raises(AssertionError, match="explode"):
        apply_edited_primary_metric_policy(
            {
                "meta": {"device": "cpu", "adapter": "hf_causal"},
                "edit": {"name": "quant_rtn"},
                "metrics": {"primary_metric": {"final": 1.0}},
            },
            profile=_BadProfile(),
        )


def test_apply_edited_primary_metric_policy_treats_typeerror_profile_as_non_enforcing() -> (
    None
):
    class _TypeErrorProfile:
        def __str__(self) -> str:
            raise TypeError("bad profile")

    payload = {
        "meta": {"device": "cpu", "adapter": "hf_causal"},
        "edit": {"name": "quant_rtn"},
        "metrics": {"primary_metric": {"final": 1.0}},
    }

    outcome = apply_edited_primary_metric_policy(payload, profile=_TypeErrorProfile())

    assert outcome.error is None
    assert outcome.diagnostic is None
    assert outcome.payload == payload


def test_baseline_input_error_message_formats_unreadable_reports(
    tmp_path: Path,
) -> None:
    path = tmp_path / "baseline.json"
    exc = ReportInputError("unreadable", path, detail="permission denied")

    assert evaluate_contract_mod._baseline_input_error_message(exc) == (
        f"Baseline report is not readable: {path} (permission denied)"
    )
