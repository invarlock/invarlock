# ruff: noqa: I001,E402,F811
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml

from invarlock.cli.commands import calibrate as calibrate_mod


def _write_base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"adapter": "noop", "id": "x", "device": "cpu"},
                "edit": {"name": "noop", "plan": {}},
                "dataset": {"provider": "synthetic", "seed": 0},
                "guards": {"order": ["spectral", "variance"]},
                "output": {"dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_ve_sweep_covers_guard_search_and_ci_width_exceptions(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "dataset": {"seed": 0},
                "edit": {"name": "noop", "plan": {"seed": 0}},
                "guards": {"variance": {"calibration": "oops"}},
                "output": {"dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    def _fake_run_command(*, out: Path, tier: str, config: Path, **_kwargs) -> str:  # noqa: ARG001
        report_path = Path(out) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "guards": [
                {
                    "name": "variance",
                    "metrics": {
                        "predictive_gate": {
                            "evaluated": True,
                            "delta_ci": ["a", "b"],
                        }
                    },
                },
                {"name": "other", "metrics": {}},
            ],
            "meta": {"tier": tier, "config": str(config)},
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with (
        patch.object(calibrate_mod, "_run_calibration_config", _fake_run_command),
        patch(
            "invarlock.cli.commands.calibrate.get_tier_guard_config",
            lambda *_a, **_k: {"predictive_one_sided": True},
        ),
    ):
        calibrate_mod.ve_sweep(
            config=base,
            out=out,
            tiers=["balanced"],
            seed=[42, 43],
            n_seeds=2,
            seed_start=42,
            window=[6],
            target_enable_rate=0.05,
            profile="ci",
            device=None,
            safety_margin=0.0,
        )


def test_ve_sweep_covers_non_dict_variance_config_and_guard_loop_continue(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "dataset": {"seed": 0},
                "edit": {"name": "noop", "plan": {"seed": 0}},
                "guards": {"variance": "oops"},
                "output": {"dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    def _fake_run_command(*, out: Path, tier: str, config: Path, **_kwargs) -> str:  # noqa: ARG001
        report_path = Path(out) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "guards": [
                {"name": "other", "metrics": {}},
                {
                    "name": "variance",
                    "metrics": {
                        "predictive_gate": {
                            "evaluated": True,
                            "mean_delta": -0.1,
                            "delta_ci": [-0.2, -0.1],
                        }
                    },
                },
            ],
            "meta": {"tier": tier, "config": str(config)},
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with (
        patch.object(calibrate_mod, "_run_calibration_config", _fake_run_command),
        patch(
            "invarlock.cli.commands.calibrate.get_tier_guard_config",
            lambda *_a, **_k: {"predictive_one_sided": True},
        ),
    ):
        calibrate_mod.ve_sweep(
            config=base,
            out=out,
            tiers=["balanced"],
            seed=[42],
            n_seeds=1,
            seed_start=42,
            window=[6],
            target_enable_rate=0.05,
            profile="ci",
            device=None,
            safety_margin=0.0,
        )

    assert (out / "ve_sweep_report.json").exists()
