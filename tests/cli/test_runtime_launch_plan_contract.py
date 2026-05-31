from __future__ import annotations

from pathlib import Path

import pytest

import invarlock.runtime_security as runtime_launch_plan
from invarlock.cli.config_execution import ConfigExecutionRequest
from invarlock.runtime_security import ContainerLaunchPlan


def test_runtime_launch_plan_helper_functions_cover_device_and_flag_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runtime_launch_plan._command_tokens(
        ["evaluate", "--config", "cfg.yaml", "--device", "cuda"]
    ) == ["evaluate", "cfg.yaml", "cuda"]
    assert runtime_launch_plan._requested_device(["evaluate"]) == "auto"
    assert runtime_launch_plan._requested_device(["run"]) == "auto"
    assert runtime_launch_plan._requested_device(["calibrate"]) == "auto"
    assert runtime_launch_plan._requested_device(["advanced", "calibrate"]) == "auto"
    assert runtime_launch_plan._requested_device(["verify"]) is None
    assert runtime_launch_plan._requested_device(["--help"]) is None
    assert (
        runtime_launch_plan._requested_device(["evaluate", "--device", "CUDA"])
        == "cuda"
    )
    assert runtime_launch_plan._requested_device(["evaluate", "--device"]) is None

    occurrences = runtime_launch_plan._iter_flag_occurrences(
        [
            "evaluate",
            "--config",
            "cfg.yaml",
            "--out=reports",
            "--baseline",
            "/tmp/base",
            "--device",
        ],
        flags={"--config", "--out", "--baseline", "--device"},
    )
    assert occurrences == [
        (1, "--config", "cfg.yaml", 2),
        (3, "--out", "reports", None),
        (4, "--baseline", "/tmp/base", 5),
    ]

    argv = ["evaluate", "--config", "cfg.yaml", "--out=reports"]
    runtime_launch_plan._replace_flag_value(
        argv,
        token_index=1,
        flag="--config",
        value_index=2,
        new_value="/workspace/cfg.yaml",
    )
    runtime_launch_plan._replace_flag_value(
        argv,
        token_index=3,
        flag="--out",
        value_index=None,
        new_value="/workspace/reports",
    )
    assert argv == [
        "evaluate",
        "--config",
        "/workspace/cfg.yaml",
        "--out=/workspace/reports",
    ]

    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )
    assert runtime_launch_plan._needs_gpu_passthrough(["evaluate"]) is True
    assert (
        runtime_launch_plan._needs_gpu_passthrough(["evaluate", "--device", "cpu"])
        is False
    )

    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )
    assert (
        runtime_launch_plan._needs_gpu_passthrough(["evaluate", "--device", "cuda"])
        is False
    )


def test_normalize_delegated_argv_rejects_explicit_cuda_without_nvidia_visibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()

    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: False,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="Requested --device cuda"):
        runtime_launch_plan.normalize_delegated_argv(
            ["evaluate", "--device", "cuda"],
            cwd=cwd,
        )


def test_normalize_delegated_argv_rewrites_paths_and_builders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    external_mount = tmp_path / "external"
    external_mount.mkdir()

    monkeypatch.setattr(
        runtime_launch_plan,
        "_normalize_config_path_for_container",
        lambda value, *, cwd, scan_dependencies: (
            f"/workspace/{Path(value).name}",
            [cwd / "config-dependency"] if scan_dependencies else [],
            scan_dependencies,
        ),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_launch_plan,
        "_normalize_output_path_for_container",
        lambda value, *, cwd: (f"/workspace/{Path(value).name}", [cwd / "reports"]),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_launch_plan,
        "_normalize_local_model_path_for_container",
        lambda value, *, cwd: (
            f"/workspace/{Path(value).name}",
            [external_mount],
            value.endswith("subject"),
        ),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_launch_plan,
        "_minimize_mounts",
        lambda mounts: sorted(mounts, key=str),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_launch_plan,
        "_host_nvidia_visible",
        lambda: True,
        raising=True,
    )

    plan = runtime_launch_plan.normalize_delegated_argv(
        [
            "evaluate",
            "--config",
            "config.yaml",
            "--baseline-report=baseline.json",
            "--out=reports",
            "--baseline",
            "baseline-model",
            "--subject",
            "subject",
        ],
        cwd=cwd,
    )

    assert isinstance(plan, ContainerLaunchPlan)
    assert list(plan.argv) == [
        "evaluate",
        "--config",
        "/workspace/config.yaml",
        "--baseline-report=/workspace/baseline.json",
        "--out=/workspace/reports",
        "--baseline",
        "baseline-model",
        "--subject",
        "/workspace/subject",
    ]
    assert list(plan.argv_mounts) == sorted(
        [cwd / "config-dependency", cwd / "reports", external_mount],
        key=str,
    )
    assert plan.needs_cwd_host_mirror is True
    assert plan.gpu_passthrough is True

    recorded: dict[str, object] = {}

    def _normalize(argv: list[str], *, cwd: Path) -> ContainerLaunchPlan:
        recorded["argv"] = list(argv)
        recorded["cwd"] = cwd
        return ContainerLaunchPlan(
            argv=tuple(argv),
            argv_mounts=(),
            needs_cwd_host_mirror=False,
            gpu_passthrough=False,
        )

    monkeypatch.setattr(
        runtime_launch_plan,
        "normalize_delegated_argv",
        _normalize,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_launch_plan.sys,
        "argv",
        ["invarlock", "evaluate", "--help"],
        raising=True,
    )

    current_process_plan = (
        runtime_launch_plan.build_current_process_container_launch_plan()
    )
    assert list(current_process_plan.argv) == ["evaluate", "--help"]
    assert recorded["argv"] == ["evaluate", "--help"]

    request = ConfigExecutionRequest(
        config="cfg.yaml",
        device="cuda",
        profile="ci",
        out="report-dir",
        edit="quant",
        edit_label="nightly",
        tier="strict",
        metric_kind="ppl_causal",
        probes=4,
        until_pass=True,
        max_attempts=5,
        timeout=120,
        baseline="baseline",
        no_cleanup=True,
        style="audit",
        progress=True,
        timing=True,
        telemetry=True,
        no_color=True,
        prefer_local_files_only=True,
    )
    request_plan = runtime_launch_plan.build_request_container_launch_plan(
        "evaluate",
        request,
    )
    assert list(request_plan.argv) == [
        "--invoked-command",
        "evaluate",
        "--config",
        "cfg.yaml",
        "--device",
        "cuda",
        "--profile",
        "ci",
        "--out",
        "report-dir",
        "--edit",
        "quant",
        "--edit-label",
        "nightly",
        "--tier",
        "strict",
        "--metric-kind",
        "ppl_causal",
        "--probes",
        "4",
        "--max-attempts",
        "5",
        "--timeout",
        "120",
        "--baseline",
        "baseline",
        "--style",
        "audit",
        "--until-pass",
        "--no-cleanup",
        "--progress",
        "--timing",
        "--telemetry",
        "--no-color",
        "--prefer-local-files-only",
    ]

    request = ConfigExecutionRequest(
        config="cfg.yaml",
        profile="ci",
        out="report-dir",
        edit="quant",
        edit_label="nightly",
        tier="strict",
        metric_kind="ppl_causal",
        probes=4,
        timeout=120,
        baseline="baseline",
        style="audit",
    )
    request_plan_default = runtime_launch_plan.build_request_container_launch_plan(
        "run",
        request,
    )
    assert list(request_plan_default.argv[:4]) == [
        "--invoked-command",
        "run",
        "--config",
        "cfg.yaml",
    ]
    assert "--device" in request_plan_default.argv
    assert "auto" in request_plan_default.argv
    assert "--max-attempts" not in request_plan_default.argv
    assert "--until-pass" not in request_plan_default.argv
    assert "--no-cleanup" not in request_plan_default.argv

    calibrate_request_plan = runtime_launch_plan.build_request_container_launch_plan(
        ("advanced", "calibrate", "null-sweep"),
        request,
    )
    assert list(calibrate_request_plan.argv[:2]) == [
        "--invoked-command",
        "advanced calibrate null-sweep",
    ]
