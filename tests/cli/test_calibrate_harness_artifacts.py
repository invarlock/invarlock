# ruff: noqa: I001,E402,F811
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
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


def test_calibrate_helpers_cover_defaults_and_errors(tmp_path: Path) -> None:
    cfg = tmp_path / "not_a_mapping.yaml"
    cfg.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter):
        calibrate_mod._load_yaml(cfg)

    specs = calibrate_mod._materialize_sweep_specs(
        tiers=None, seeds=None, n_seeds=2, seed_start=100
    )
    assert len(specs) == 3 * 2

    specs_w = calibrate_mod._materialize_sweep_specs(
        tiers=["balanced"], seeds=[1], n_seeds=1, seed_start=1, windows=[6]
    )
    assert specs_w[0].windows == 6

    csv_path = tmp_path / "empty.csv"
    calibrate_mod._dump_csv(csv_path, [])
    assert csv_path.read_text(encoding="utf-8") == ""


def test_null_sweep_emits_json_csv_md_and_tier_patch(tmp_path: Path) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"

    def _fake_run_command(*, out: str, tier: str, config: str, **_kwargs) -> str | None:  # noqa: ARG001
        # Exercise both branches: one run produces a report; one is skipped.
        if "seed_43" in out:
            return None
        report_path = Path(out) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "guards": [
                {
                    "name": "spectral",
                    "metrics": {
                        "caps_applied": 1,
                        "caps_exceeded": False,
                        "family_z_summary": {
                            "ffn": {"max": 3.0},
                            "attn": {"max": 2.0},
                        },
                        "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                        "multiple_testing_selection": {
                            "family_pvalues": {"ffn": 0.04, "attn": 0.2},
                            "families_selected": ["ffn"],
                            "family_violation_counts": {"ffn": 1, "attn": 0},
                        },
                    },
                    "violations": [{"family": "ffn"}],
                }
            ],
            "meta": {"tier": tier, "seed": 0, "config": config},
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with patch("invarlock.cli.commands.run.run_command", _fake_run_command):
        calibrate_mod.null_sweep(
            config=cfg,
            out=out,
            tiers=["balanced"],
            seed=[42, 43],
            n_seeds=2,
            seed_start=42,
            profile="ci",
            device=None,
            safety_margin=0.05,
            target_any_warning_rate=0.01,
        )

    assert (out / "null_sweep_report.json").exists()
    assert (out / "null_sweep_runs.csv").exists()
    assert (out / "null_sweep_summary.md").exists()
    assert (out / "tiers_patch_spectral_null.yaml").exists()

    report = json.loads((out / "null_sweep_report.json").read_text(encoding="utf-8"))
    assert report["kind"] == "spectral_null_sweep"
    assert "balanced" in report["summaries"]

    tiers_patch = yaml.safe_load(
        (out / "tiers_patch_spectral_null.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(tiers_patch, dict)
    assert "balanced" in tiers_patch
    assert "spectral_guard" in tiers_patch["balanced"]


def test_null_sweep_handles_missing_spectral_and_bad_metrics(tmp_path: Path) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"

    def _fake_run_command(*, out: str, tier: str, config: str, **_kwargs) -> str:  # noqa: ARG001
        report_path = Path(out) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "guards": [
                {
                    "name": "spectral",
                    "metrics": {
                        "caps_applied": "nope",
                        "caps_exceeded": False,
                        "family_z_summary": {"ffn": {"max": "bad"}, "attn": "oops"},
                        "multiple_testing_selection": {
                            "family_violation_counts": {"ffn": "bad"},
                            "families_selected": "nope",
                        },
                    },
                    "violations": "nope",
                },
                {"name": "other", "metrics": {}},
            ],
            "meta": {"tier": tier, "config": config},
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with patch("invarlock.cli.commands.run.run_command", _fake_run_command):
        calibrate_mod.null_sweep(
            config=cfg,
            out=out,
            tiers=["balanced"],
            seed=[42],
            n_seeds=1,
            seed_start=42,
            profile="ci",
            device=None,
            safety_margin=0.05,
            target_any_warning_rate=0.01,
        )


def test_null_sweep_covers_guard_search_empty_and_non_dict_guards(tmp_path: Path) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"

    def _fake_run_command(*, out: str, tier: str, config: str, **_kwargs) -> str:  # noqa: ARG001
        report_path = Path(out) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        guards = [] if "seed_42" in out else ["not-a-dict"]
        payload = {"guards": guards, "meta": {"tier": tier, "config": config}}
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with patch("invarlock.cli.commands.run.run_command", _fake_run_command):
        calibrate_mod.null_sweep(
            config=cfg,
            out=out,
            tiers=["balanced"],
            seed=[42, 43],
            n_seeds=2,
            seed_start=42,
            profile="ci",
            device=None,
            safety_margin=0.05,
            target_any_warning_rate=0.01,
        )


def test_ve_sweep_emits_json_csv_power_curve_and_tier_patch(tmp_path: Path) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"

    def _fake_run_command(*, out: str, tier: str, config: str, **_kwargs) -> str:  # noqa: ARG001
        report_path = Path(out) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # Vary CI width by window size to exercise branches and power curve.
        if "windows_6" in out:
            delta_ci = [-0.002, -0.001]
        else:
            delta_ci = None
        payload = {
            "guards": [
                {
                    "name": "variance",
                    "metrics": {
                        "predictive_gate": {
                            "evaluated": True,
                            "mean_delta": -0.001,
                            "delta_ci": delta_ci,
                        }
                    },
                }
            ],
            "meta": {"tier": tier, "config": config},
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with (
        patch("invarlock.cli.commands.run.run_command", _fake_run_command),
        patch(
            "invarlock.cli.commands.calibrate.get_tier_guard_config",
            lambda *_a, **_k: {"predictive_one_sided": True},
        ),
    ):
        calibrate_mod.ve_sweep(
            config=cfg,
            out=out,
            tiers=["balanced"],
            seed=[42],
            n_seeds=1,
            seed_start=42,
            window=[6, 8],
            target_enable_rate=0.05,
            profile="ci",
            device=None,
            safety_margin=0.0,
        )

    assert (out / "ve_sweep_report.json").exists()
    assert (out / "ve_sweep_runs.csv").exists()
    assert (out / "ve_power_curve.csv").exists()
    assert (out / "ve_sweep_summary.md").exists()
    assert (out / "tiers_patch_variance_ve.yaml").exists()

    tiers_patch = yaml.safe_load(
        (out / "tiers_patch_variance_ve.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(tiers_patch, dict)
    assert tiers_patch["balanced"]["variance_guard"]["min_effect_lognll"] is not None


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

    def _fake_run_command(*, out: str, tier: str, config: str, **_kwargs) -> str | None:  # noqa: ARG001
        if "seed_43" in out:
            return None
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
            "meta": {"tier": tier, "config": config},
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with (
        patch("invarlock.cli.commands.run.run_command", _fake_run_command),
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
