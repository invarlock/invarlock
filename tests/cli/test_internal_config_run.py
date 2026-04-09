from __future__ import annotations

import invarlock.cli.internal_config_run as internal_config_run


def test_internal_config_run_forwards_args_without_redelegating(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        internal_config_run,
        "run_from_config",
        lambda **kwargs: seen.update(kwargs),
        raising=True,
    )

    exit_code = internal_config_run.main(
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
        "config": "configs/demo.yaml",
        "device": "auto",
        "profile": "ci",
        "out": "runs/demo",
        "edit": "quant_rtn",
        "edit_label": "nightly",
        "tier": "balanced",
        "metric_kind": "ppl_causal",
        "probes": 4,
        "until_pass": True,
        "max_attempts": 5,
        "timeout": 120,
        "baseline": "baseline.json",
        "no_cleanup": True,
        "style": "audit",
        "progress": True,
        "timing": True,
        "telemetry": True,
        "no_color": True,
        "prefer_local_files_only": True,
        "command_name": "advanced calibrate null-sweep",
        "delegate": False,
    }
