# ruff: noqa: I001,E402,F811
from __future__ import annotations

import builtins
import json
import sys
import types
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


def test_mark_calibration_context_repairs_non_mapping_state() -> None:
    cfg = {"context": "bad"}
    calibrate_mod._mark_calibration_context(cfg)
    assert cfg["context"]["run"]["skip_overhead_check"] is True

    cfg2 = {"context": {"run": "bad"}}
    calibrate_mod._mark_calibration_context(cfg2)
    assert cfg2["context"]["run"]["skip_overhead_check"] is True


def test_get_tier_guard_config_missing_optional_deps_and_reraise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _missing_torch(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "invarlock.guards.tier_config":
            exc = ModuleNotFoundError("missing torch")
            exc.name = "torch"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_torch)
    with pytest.raises(typer.Exit):
        calibrate_mod.get_tier_guard_config("balanced", "variance_guard")

    def _missing_other(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "invarlock.guards.tier_config":
            exc = ModuleNotFoundError("missing other")
            exc.name = "not_torch"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_other)
    with pytest.raises(ModuleNotFoundError):
        calibrate_mod.get_tier_guard_config("balanced", "variance_guard")


def test_calibrate_commands_exit_on_missing_optional_deps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"
    real_import = builtins.__import__

    def _missing_spectral(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "invarlock.calibration.spectral_null":
            exc = ModuleNotFoundError("missing torch")
            exc.name = "torch"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_spectral)
    with pytest.raises(typer.Exit):
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

    def _missing_variance(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "invarlock.calibration.variance_ve":
            exc = ModuleNotFoundError("missing transformers")
            exc.name = "transformers"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_variance)
    with pytest.raises(typer.Exit):
        calibrate_mod.ve_sweep(
            config=cfg,
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


def test_null_sweep_runtime_flags_do_not_block_missing_dep_error(
    tmp_path: Path,
) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"
    real_import = builtins.__import__

    def _missing_spectral(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "invarlock.calibration.spectral_null":
            exc = ModuleNotFoundError("missing torch")
            exc.name = "torch"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch.object(calibrate_mod, "_run_calibration_config") as run_calibration,
        patch("builtins.__import__", _missing_spectral),
        pytest.raises(typer.Exit),
    ):
        calibrate_mod.null_sweep(
            config=cfg,
            out=out,
            tiers=["balanced"],
            seed=[42],
            n_seeds=1,
            seed_start=42,
            profile="ci",
            device=None,
            allow_network=True,
            allow_host_execution=True,
            allow_third_party_plugins=True,
            allow_remote_code=True,
            safety_margin=0.05,
            target_any_warning_rate=0.01,
        )

    run_calibration.assert_not_called()


def test_ve_sweep_runtime_flags_do_not_block_missing_dep_error(
    tmp_path: Path,
) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"
    real_import = builtins.__import__

    def _missing_variance(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "invarlock.calibration.variance_ve":
            exc = ModuleNotFoundError("missing transformers")
            exc.name = "transformers"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch.object(calibrate_mod, "_run_calibration_config") as run_calibration,
        patch("builtins.__import__", _missing_variance),
        pytest.raises(typer.Exit),
    ):
        calibrate_mod.ve_sweep(
            config=cfg,
            out=out,
            tiers=["balanced"],
            seed=[42],
            n_seeds=1,
            seed_start=42,
            window=[6],
            target_enable_rate=0.05,
            profile="ci",
            device=None,
            allow_network=True,
            allow_host_execution=True,
            allow_third_party_plugins=True,
            allow_remote_code=True,
            safety_margin=0.0,
        )

    run_calibration.assert_not_called()


def test_null_sweep_emits_json_csv_md_and_tier_patch(tmp_path: Path) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"

    def _fake_run_command(*, out: Path, tier: str, config: Path, **_kwargs) -> str:  # noqa: ARG001
        loaded = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        assert loaded["context"]["run"]["skip_overhead_check"] is True
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
                            "other": {"max": None},
                        },
                        "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                        "multiple_testing_selection": {
                            "family_pvalues": {"ffn": 0.04, "attn": 0.2},
                            "families_selected": ["ffn"],
                            "family_violation_counts": {"ffn": 1, "attn": 0},
                        },
                    },
                    "violations": [{"family": None}, {"family": "ffn"}, "not-a-dict"],
                }
            ],
            "meta": {"tier": tier, "seed": 0, "config": str(config)},
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with patch.object(calibrate_mod, "_run_calibration_config", _fake_run_command):
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


def test_ve_sweep_handles_reports_without_variance_guard(tmp_path: Path) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"
    fake_module = types.SimpleNamespace(
        summarize_ve_sweep_reports=lambda reports, **kwargs: {
            "n_runs": len(reports),
            "recommendations": {"min_effect_lognll": 0.12},
        }
    )

    def _fake_run_command(*, out: Path, tier: str, config: Path, **_kwargs) -> str:  # noqa: ARG001
        report_path = Path(out) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "guards": [{"name": "other", "metrics": {}}],
                    "meta": {"tier": tier, "config": str(config)},
                }
            ),
            encoding="utf-8",
        )
        return str(report_path)

    with (
        patch.object(calibrate_mod, "_run_calibration_config", _fake_run_command),
        patch.object(
            calibrate_mod,
            "get_tier_guard_config",
            return_value={"predictive_one_sided": True},
        ),
        patch.dict(sys.modules, {"invarlock.calibration.variance_ve": fake_module}),
    ):
        calibrate_mod.ve_sweep(
            config=cfg,
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

    runs_csv = (out / "ve_sweep_runs.csv").read_text(encoding="utf-8")
    power_csv = (out / "ve_power_curve.csv").read_text(encoding="utf-8")
    assert "predictive_evaluated" in runs_csv
    assert "False" in runs_csv
    assert "mean_ci_width" in power_csv


def test_null_sweep_handles_missing_spectral_and_bad_metrics(tmp_path: Path) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"

    def _fake_run_command(*, out: Path, tier: str, config: Path, **_kwargs) -> str:  # noqa: ARG001
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
            "meta": {"tier": tier, "config": str(config)},
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with patch.object(calibrate_mod, "_run_calibration_config", _fake_run_command):
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


def test_null_sweep_covers_guard_search_empty_and_non_dict_guards(
    tmp_path: Path,
) -> None:
    cfg = _write_base_config(tmp_path)
    out = tmp_path / "out"

    def _fake_run_command(*, out: Path, tier: str, config: Path, **_kwargs) -> str:  # noqa: ARG001
        report_path = Path(out) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        guards = [] if "seed_42" in str(out) else ["not-a-dict"]
        payload = {"guards": guards, "meta": {"tier": tier, "config": str(config)}}
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(report_path)

    with patch.object(calibrate_mod, "_run_calibration_config", _fake_run_command):
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

    def _fake_run_command(*, out: Path, tier: str, config: Path, **_kwargs) -> str:  # noqa: ARG001
        loaded = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        assert loaded["context"]["run"]["skip_overhead_check"] is True
        report_path = Path(out) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # Vary CI width by window size to exercise branches and power curve.
        if "windows_6" in str(out):
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
