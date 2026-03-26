from __future__ import annotations

from invarlock.core.evaluate_plan import (
    DEFAULT_EVALUATE_GUARDS_ORDER,
    build_baseline_run_config,
    build_evaluation_report_kwargs,
    build_subject_edit_run_config,
    default_preset_data_for_adapter,
    determine_subject_label,
    normalize_model_id,
    resolve_guards_order,
    sanitize_preset_data_for_evaluate,
)


def test_default_preset_data_for_adapter_uses_expected_sequence_lengths() -> None:
    assert default_preset_data_for_adapter("hf_causal")["dataset"]["seq_len"] == 512
    assert default_preset_data_for_adapter("hf_mlm")["dataset"]["seq_len"] == 128


def test_normalize_model_id_strips_hf_prefix_for_hf_adapters() -> None:
    assert normalize_model_id("hf:org/model", "hf_causal") == "org/model"
    assert normalize_model_id("hf:org/model", "custom") == "hf:org/model"


def test_sanitize_preset_data_for_evaluate_removes_pinned_device() -> None:
    payload = {"model": {"device": "cuda", "id": "demo"}}

    sanitized = sanitize_preset_data_for_evaluate(payload)

    assert sanitized == {"model": {"id": "demo"}}
    assert payload == {"model": {"device": "cuda", "id": "demo"}}


def test_resolve_guards_order_prefers_preset_then_default() -> None:
    assert resolve_guards_order({}) == DEFAULT_EVALUATE_GUARDS_ORDER
    assert resolve_guards_order({"guards": {"order": ["g1", "g2"]}}) == ["g1", "g2"]


def test_determine_subject_label_matches_compare_and_edit_modes() -> None:
    assert (
        determine_subject_label(
            edit_label="quant_rtn",
            edit_config=None,
            source_model_id="a",
            subject_model_id="b",
        )
        == "quant_rtn"
    )
    assert (
        determine_subject_label(
            edit_label=None,
            edit_config=None,
            source_model_id="a",
            subject_model_id="b",
        )
        == "custom"
    )
    assert (
        determine_subject_label(
            edit_label=None,
            edit_config=None,
            source_model_id="a",
            subject_model_id="a",
        )
        == "noop"
    )
    assert (
        determine_subject_label(
            edit_label=None,
            edit_config="edit.yaml",
            source_model_id="a",
            subject_model_id="b",
        )
        is None
    )


def test_build_baseline_run_config_injects_evaluate_context() -> None:
    cfg = build_baseline_run_config(
        {"dataset": {"provider": "wikitext2"}},
        model_id="org/model",
        adapter_name="hf_causal",
        output_dir="runs/source",
        profile="dev",
        tier="balanced",
        guards_order=["g1", "g2"],
    )

    assert cfg["model"] == {"id": "org/model", "adapter": "hf_causal"}
    assert cfg["edit"] == {"name": "noop", "plan": {}}
    assert cfg["output"] == {"dir": "runs/source"}
    assert cfg["guards"] == {"order": ["g1", "g2"]}
    assert cfg["context"] == {"profile": "dev", "tier": "balanced"}


def test_build_subject_edit_run_config_normalizes_placeholders_and_guards() -> None:
    cfg = build_subject_edit_run_config(
        {"dataset": {"provider": "wikitext2"}},
        {"model": {"id": "<MODEL_ID>"}, "edit": {"name": "quant_rtn"}},
        subject_model_id="hf:org/model",
        adapter_name="hf_causal",
        output_dir="runs/edited",
        profile="ci",
        tier="balanced",
        guards_order=["invariants", "variance"],
    )

    assert cfg["model"] == {"id": "hf:org/model", "adapter": "hf_causal"}
    assert cfg["output"] == {"dir": "runs/edited"}
    assert cfg["context"] == {"profile": "ci", "tier": "balanced"}
    assert cfg["guards"] == {"order": ["invariants", "variance"]}


def test_build_evaluation_report_kwargs_are_explicit() -> None:
    kwargs = build_evaluation_report_kwargs(
        edited_report="runs/edited/report.json",
        baseline_report="runs/source/report.json",
        report_out="reports/eval",
        style="audit",
        no_color=True,
        baseline_seconds=1.0,
        subject_seconds=2.0,
        report_start=3.0,
    )

    assert kwargs["run"] == "runs/edited/report.json"
    assert kwargs["baseline"] == "runs/source/report.json"
    assert kwargs["output"] == "reports/eval"
    assert kwargs["summary_baseline_seconds"] == 1.0
    assert kwargs["summary_subject_seconds"] == 2.0
    assert kwargs["summary_report_start"] == 3.0
