from __future__ import annotations

import invarlock.cli.config_execution as config_execution
from invarlock.cli.config_execution import ConfigExecutionRequest


def test_config_execution_entrypoint_forwards_args_without_redelegating(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        config_execution,
        "run_request",
        lambda request, **kwargs: seen.update({"request": request, **kwargs}),
        raising=True,
    )

    exit_code = config_execution.main(
        [
            "--invoked-command",
            "advanced calibrate null-sweep",
            "--config",
            "configs/demo.yaml",
            "--device",
            "auto",
            "--profile",
            "ci",
            "--out",
            "runs/demo",
            "--edit",
            "quant_rtn",
            "--edit-label",
            "nightly",
            "--tier",
            "balanced",
            "--metric-kind",
            "ppl_causal",
            "--probes",
            "4",
            "--until-pass",
            "--max-attempts",
            "5",
            "--timeout",
            "120",
            "--baseline",
            "baseline.json",
            "--no-cleanup",
            "--style",
            "audit",
            "--progress",
            "--timing",
            "--telemetry",
            "--no-color",
            "--prefer-local-files-only",
        ]
    )

    assert exit_code == 0
    assert seen == {
        "request": ConfigExecutionRequest(
            config="configs/demo.yaml",
            device="auto",
            profile="ci",
            out="runs/demo",
            edit="quant_rtn",
            edit_label="nightly",
            tier="balanced",
            metric_kind="ppl_causal",
            probes=4,
            until_pass=True,
            max_attempts=5,
            timeout=120,
            baseline="baseline.json",
            no_cleanup=True,
            style="audit",
            progress=True,
            timing=True,
            telemetry=True,
            no_color=True,
            prefer_local_files_only=True,
        ),
        "command_name": "advanced calibrate null-sweep",
        "delegate": False,
    }
