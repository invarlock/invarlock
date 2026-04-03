from __future__ import annotations

from dataclasses import dataclass

from invarlock.core.run_execution_request_policy import (
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


def test_build_run_execution_request_reads_policy_from_environ() -> None:
    request = _Request()
    core_request = build_run_execution_request(
        request,
        environ={
            "INVARLOCK_EVAL_DEVICE": "cuda:0",
            "PACK_DETERMINISM": "strict",
            "INVARLOCK_DETERMINISM_WARN_ONLY": "1",
            "INVARLOCK_TINY_RELAX": "true",
            "INVARLOCK_EXPORT_MODEL": "yes",
            "INVARLOCK_EXPORT_DIR": "exports",
        },
    )

    assert core_request.capture_timings is True
    assert core_request.eval_device_override == "cuda:0"
    assert core_request.determinism_mode == "strict"
    assert core_request.determinism_warn_only is True
    assert core_request.tiny_relax_enabled is True
    assert core_request.export_model_requested is True
    assert core_request.export_dir == "exports"


def test_build_run_execution_request_prefers_pack_determinism_over_legacy_env() -> None:
    request = _Request()

    core_request = build_run_execution_request(
        request,
        environ={
            "PACK_DETERMINISM": "strict",
            "INVARLOCK_DETERMINISM": "legacy",
        },
    )

    assert core_request.determinism_mode == "strict"
