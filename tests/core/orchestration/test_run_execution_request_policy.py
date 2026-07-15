from __future__ import annotations

from dataclasses import dataclass

from invarlock.core.run_policy import (
    build_run_execution_request,
    env_flag,
    env_text,
)


@dataclass
class _Request:
    config: str = "config.yaml"
    device: str | None = "cpu"
    profile: str | None = "ci"
    out: str | None = "runs"
    edit: str | None = None
    edit_label: str | None = None
    tier: str | None = "balanced"
    metric_kind: str | None = None
    probes: int | None = None
    until_pass: bool = False
    max_attempts: int = 1
    timeout: int | None = None
    baseline: str | None = None
    no_cleanup: bool = False
    timing: bool = False
    progress: bool = True
    telemetry: bool = True
    prefer_local_files_only: bool = False
    resolved_config_out: str | None = None


def test_env_text_and_flag_normalize_environ_values() -> None:
    environ = {
        "INVARLOCK_FLAG": " yes ",
        "INVARLOCK_TEXT": " value ",
    }

    assert env_flag("INVARLOCK_FLAG", environ=environ) is True
    assert env_text("INVARLOCK_TEXT", environ=environ) == "value"
    assert env_text("MISSING", environ=environ) is None


def test_env_text_returns_none_for_whitespace_only_value() -> None:
    assert env_text("INVARLOCK_TEXT", environ={"INVARLOCK_TEXT": "   "}) is None


def test_env_text_ignores_non_string_values_and_env_flag_treats_none_as_false() -> None:
    environ = {
        "INVARLOCK_EVAL_DEVICE": 7,
        "INVARLOCK_TINY_RELAX": None,
    }

    assert env_text("INVARLOCK_EVAL_DEVICE", environ=environ) is None
    assert env_flag("INVARLOCK_TINY_RELAX", environ=environ) is False


def test_env_flag_returns_false_for_falsey_values_and_missing() -> None:
    environ = {
        "INVARLOCK_FLAG_FALSE": "off",
        "INVARLOCK_FLAG_ZERO": " 0 ",
        "INVARLOCK_FLAG_OTHER": "maybe",
    }

    assert env_flag("INVARLOCK_FLAG_FALSE", environ=environ) is False
    assert env_flag("INVARLOCK_FLAG_ZERO", environ=environ) is False
    assert env_flag("INVARLOCK_FLAG_OTHER", environ=environ) is False
    assert env_flag("INVARLOCK_FLAG_MISSING", environ=environ) is False


def test_build_run_execution_request_reads_policy_from_environ() -> None:
    request = _Request()
    core_request = build_run_execution_request(
        request,
        environ={
            "INVARLOCK_EVAL_DEVICE": "cuda:0",
            "PACK_DETERMINISM": "strict",
            "INVARLOCK_TINY_RELAX": "true",
            "INVARLOCK_EXPORT_MODEL": "yes",
            "INVARLOCK_EXPORT_DIR": "exports",
        },
    )

    assert core_request.capture_timings is True
    assert core_request.eval_device_override == "cuda:0"
    assert core_request.determinism_mode == "strict"
    assert core_request.determinism_warn_only is False
    assert core_request.tiny_relax_enabled is True
    assert core_request.export_model_requested is True
    assert core_request.export_dir == "exports"


def test_build_run_execution_request_uses_pack_determinism_and_can_disable_timings() -> (
    None
):
    request = _Request(progress=False, timing=False)

    core_request = build_run_execution_request(
        request,
        environ={
            "INVARLOCK_EVAL_DEVICE": "   ",
            "PACK_DETERMINISM": "strict",
            "INVARLOCK_EXPORT_DIR": "   ",
        },
    )

    assert core_request.capture_timings is False
    assert core_request.eval_device_override is None
    assert core_request.determinism_mode == "strict"
    assert core_request.determinism_warn_only is False
    assert core_request.export_dir is None


def test_build_run_execution_request_treats_blank_pack_determinism_as_missing() -> None:
    request = _Request(progress=False, timing=False)

    core_request = build_run_execution_request(
        request,
        environ={
            "PACK_DETERMINISM": "   ",
        },
    )

    assert core_request.capture_timings is False
    assert core_request.determinism_mode is None


def test_build_run_execution_request_propagates_request_fields_and_timing_flag() -> (
    None
):
    request = _Request(
        device="cuda:1",
        profile="prod",
        out="custom-runs",
        edit="quant_rtn",
        edit_label="wave2",
        tier="strict",
        metric_kind="ppl_causal",
        probes=7,
        until_pass=True,
        max_attempts=4,
        timeout=90,
        baseline="baseline.json",
        no_cleanup=True,
        timing=True,
        progress=False,
        telemetry=False,
        prefer_local_files_only=True,
        resolved_config_out="resolved-config.yaml",
    )

    core_request = build_run_execution_request(request, environ={})

    assert core_request.device == "cuda:1"
    assert core_request.profile == "prod"
    assert core_request.out == "custom-runs"
    assert core_request.edit == "quant_rtn"
    assert core_request.edit_label == "wave2"
    assert core_request.tier == "strict"
    assert core_request.metric_kind == "ppl_causal"
    assert core_request.probes == 7
    assert core_request.until_pass is True
    assert core_request.max_attempts == 4
    assert core_request.timeout == 90
    assert core_request.baseline == "baseline.json"
    assert core_request.no_cleanup is True
    assert core_request.capture_timings is True
    assert core_request.telemetry is False
    assert core_request.prefer_local_files_only is True
    assert core_request.resolved_config_out == "resolved-config.yaml"
