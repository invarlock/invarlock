from __future__ import annotations

import json
from pathlib import Path

from scripts.evidence_packs.python import preset_generator


def _write_calibration_run(
    cal_dir: Path,
    *,
    run_name: str,
    attn_growth: float,
    ratio: float = 1.0,
) -> None:
    run_dir = cal_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "guards": [
            {
                "name": "rmt",
                "policy": {},
                "metrics": {
                    "epsilon_by_family": {
                        "attn": 0.01,
                        "ffn": 0.01,
                        "embed": 0.01,
                        "other": 0.01,
                    },
                    "edge_risk_by_family_base": {
                        "attn": 16.0,
                        "ffn": 17.0,
                        "embed": 8.0,
                        "other": 1.0,
                    },
                    "edge_risk_by_family": {
                        "attn": 16.0 * (1.0 + attn_growth),
                        "ffn": 17.0,
                        "embed": 8.0,
                        "other": 1.0,
                    },
                },
            }
        ],
        "metrics": {
            "primary_metric": {
                "preview_final_ratio": ratio,
            }
        },
    }
    (run_dir / "baseline_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_generate_preset_raises_rmt_epsilon_from_observed_clean_growth(
    tmp_path: Path,
) -> None:
    cal_dir = tmp_path / "calibration"
    growths = [0.0019, 0.0219, 0.0, 0.0, 0.0019]
    for idx, growth in enumerate(growths, start=1):
        _write_calibration_run(
            cal_dir,
            run_name=f"run_{idx}",
            attn_growth=growth,
            ratio=1.0 + (idx * 0.001),
        )

    preset_file, _stats_path, derived_files = preset_generator.generate_preset(
        cal_dir=cal_dir,
        preset_file=tmp_path / "calibrated_preset_demo.json",
        model_name="demo-model",
        model_path="/models/demo",
        tier="balanced",
        dataset_provider="wikitext2",
        seq_len=512,
        stride=512,
        preview_n=64,
        final_n=64,
        edit_types=["magnitude_prune"],
    )

    preset = json.loads(preset_file.read_text(encoding="utf-8"))
    derived = json.loads(derived_files[0].read_text(encoding="utf-8"))

    attn_eps = preset["guards"]["rmt"]["epsilon_by_family"]["attn"]
    assert attn_eps > 0.01
    assert attn_eps == derived["guards"]["rmt"]["epsilon_by_family"]["attn"]
    assert derived["_calibration_meta"]["edit_type"] == "magnitude_prune"


def test_calibrate_drift_single_ratio_stays_truthful_about_compatibility() -> None:
    drift = preset_generator.calibrate_drift(
        [{"primary_metric": {"preview_final_ratio": 1.08}}]
    )

    assert drift == {
        "mean": 1.08,
        "std": 0.0,
        "min": 1.08,
        "max": 1.08,
        "suggested_band": [1.07, 1.09],
        "band_compatible": False,
    }


def test_generate_preset_single_run_derives_narrow_observed_drift_band(
    tmp_path: Path,
) -> None:
    cal_dir = tmp_path / "calibration"
    _write_calibration_run(
        cal_dir,
        run_name="run_1",
        attn_growth=0.0,
        ratio=1.08,
    )

    preset_file, _stats_path, _derived_files = preset_generator.generate_preset(
        cal_dir=cal_dir,
        preset_file=tmp_path / "calibrated_preset_demo.json",
        model_name="demo-model",
        model_path="/models/demo",
        tier="balanced",
        dataset_provider="wikitext2",
        seq_len=512,
        stride=512,
        preview_n=64,
        final_n=64,
        edit_types=["magnitude_prune"],
    )

    preset = json.loads(preset_file.read_text(encoding="utf-8"))

    assert preset["_calibration_meta"]["drift_mean"] == 1.08
    assert preset["_calibration_meta"]["drift_band_compatible"] is False
    assert preset["_calibration_meta"]["suggested_drift_band"] == [1.07, 1.09]
    assert preset["primary_metric"]["drift_band"] == {"min": 1.07, "max": 1.09}
