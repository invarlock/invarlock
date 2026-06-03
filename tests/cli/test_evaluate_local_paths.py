from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.cli.commands.evaluate import evaluate_command


def _write_run_report(
    dir_: Path,
    *,
    pm_kind: str,
    pm_final: float,
    pm_preview: float,
    ratio_vs_baseline: float,
    latency_ms_per_tok: float,
    provider_ids_digest: str,
    guard_overhead: dict | None = None,
) -> Path:
    ts_dir = dir_ / "20250101_000000"
    ts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": {
            "model_id": "stub",
            "adapter": "hf_causal",
            "seed": 7,
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "x",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": pm_kind,
                "preview": pm_preview,
                "final": pm_final,
                "ratio_vs_baseline": ratio_vs_baseline,
                "display_ci": (pm_final, pm_final),
            },
            # Legacy fields kept in run reports for CLI printing paths; evaluation reports are PM-only
            "ppl_preview": pm_preview,
            "ppl_final": pm_final,
            "ppl_ratio": ratio_vs_baseline,
            "latency_ms_per_tok": latency_ms_per_tok,
        },
        "provenance": {"provider_digest": {"ids_sha256": provider_ids_digest}},
        "evaluation_windows": {
            "final": {
                "window_ids": [1],
                "logloss": [1.0],
                "input_ids": [[1, 2]],
                "attention_masks": [[1, 1]],
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    if guard_overhead:
        report["guard_overhead"] = guard_overhead
    (ts_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return ts_dir / "report.json"


@pytest.mark.unit
def test_evaluate_local_paths_pm_and_digests(monkeypatch, tmp_path: Path):
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2"}), encoding="utf-8"
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2"}), encoding="utf-8"
    )

    calls: dict[str, list] = {"runs": []}

    def fake_run(**kwargs):
        out_dir = Path(kwargs.get("out"))
        calls["runs"].append(kwargs)
        if Path(kwargs.get("baseline") or "").exists():
            # Edited run
            return str(
                _write_run_report(
                    out_dir,
                    pm_kind="ppl_causal",
                    pm_final=10.0,
                    pm_preview=10.0,
                    ratio_vs_baseline=1.0,
                    latency_ms_per_tok=0.80,
                    provider_ids_digest="edited1234",
                )
            )
        else:
            # Baseline run
            return str(
                _write_run_report(
                    out_dir,
                    pm_kind="ppl_causal",
                    pm_final=10.0,
                    pm_preview=10.0,
                    ratio_vs_baseline=1.0,
                    latency_ms_per_tok=0.80,
                    provider_ids_digest="baseline1234",
                )
            )

    # Patch evaluate workflow to use our fake run via the run module; bind _report to the programmatic wrapper
    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as mod

    monkeypatch.setattr(
        run_mod, "run_command", lambda **kwargs: fake_run(**kwargs), raising=False
    )
    import json as _json

    def _report_wrapper(
        *,
        run: str,
        format: str,
        baseline: str | None,
        output: str,
        compare: str | None = None,
    ):
        from invarlock.reporting.report_bundle import (
            save_evaluation_bundle as _save_evaluation_bundle,
        )
        from invarlock.reporting.report_make import make_report as _make_report

        with open(run, encoding="utf-8") as fh:
            primary = _json.load(fh)
        base = None
        if baseline:
            with open(baseline, encoding="utf-8") as fh:
                base = _json.load(fh)
        return _save_evaluation_bundle(
            run_report=primary,
            output_dir=output,
            evaluation_report=_make_report(primary, base) if base is not None else {},
        )

    monkeypatch.setattr(mod, "generate_reports", _report_wrapper, raising=False)

    report_dir = tmp_path / "reports"
    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(report_dir),
    )

    report_path = Path(report_dir) / "evaluation.report.json"
    assert report_path.exists(), "evaluation report JSON not written"
    report = json.loads(report_path.read_text())

    # PM-only evaluation report v1
    assert report.get("schema_version") == "v1"
    pm = report.get("primary_metric", {})
    assert isinstance(pm, dict)
    assert pm.get("kind") in {
        "ppl_causal",
        "ppl_mlm",
        "ppl_seq2seq",
        "accuracy",
    }
    # ratio should be deterministic at 1.0
    assert abs(float(pm.get("ratio_vs_baseline", 0.0)) - 1.0) < 1e-6
    # provider digest copied through
    prov = report.get("provenance", {})
    pd = prov.get("provider_digest", {})
    assert isinstance(pd, dict)
    # edited provider digest is reflected
    # Note: evaluation report copies provider_digest from run report; edited side dominates
    assert "ids_sha256" in pd


@pytest.mark.unit
def test_evaluate_local_paths_quantized_subject_overheads(monkeypatch, tmp_path: Path):
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2"}), encoding="utf-8"
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2"}), encoding="utf-8"
    )

    def fake_run(**kwargs):
        out_dir = Path(kwargs.get("out"))
        if Path(kwargs.get("baseline") or "").exists():
            # Edited run: slightly worse ppl; slower
            guard = {
                "bare_report": {
                    "evaluation_windows": {
                        "final": {"logloss": [1.0], "token_counts": [100]}
                    },
                },
                "guarded_report": {
                    "evaluation_windows": {
                        "final": {"logloss": [1.05], "token_counts": [100]}
                    },
                },
                "overhead_threshold": 0.10,
            }
            path = _write_run_report(
                out_dir,
                pm_kind="ppl_causal",
                pm_final=10.5,
                pm_preview=10.0,
                ratio_vs_baseline=1.05,
                latency_ms_per_tok=0.90,
                provider_ids_digest="editedq123",
                guard_overhead=guard,
            )
            # Explicit latency p50 to stabilize System Overhead ratio
            data = json.loads(path.read_text())
            data.setdefault("metrics", {})["latency_ms_p50"] = 0.90
            path.write_text(json.dumps(data), encoding="utf-8")
            return str(path)
        else:
            # Baseline run
            path = _write_run_report(
                out_dir,
                pm_kind="ppl_causal",
                pm_final=10.0,
                pm_preview=10.0,
                ratio_vs_baseline=1.0,
                latency_ms_per_tok=0.80,
                provider_ids_digest="baselineq123",
            )
            data = json.loads(path.read_text())
            data.setdefault("metrics", {})["latency_ms_p50"] = 0.80
            path.write_text(json.dumps(data), encoding="utf-8")
            return str(path)

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as mod

    monkeypatch.setattr(
        run_mod, "run_command", lambda **kwargs: fake_run(**kwargs), raising=False
    )
    import json as _json

    def _report_wrapper(
        *,
        run: str,
        format: str,
        baseline: str | None,
        output: str,
        compare: str | None = None,
    ):
        from invarlock.reporting.report_bundle import (
            save_evaluation_bundle as _save_evaluation_bundle,
        )
        from invarlock.reporting.report_make import make_report as _make_report

        with open(run, encoding="utf-8") as fh:
            primary = _json.load(fh)
        base = None
        if baseline:
            with open(baseline, encoding="utf-8") as fh:
                base = _json.load(fh)
        return _save_evaluation_bundle(
            run_report=primary,
            output_dir=output,
            evaluation_report=_make_report(primary, base) if base is not None else {},
        )

    monkeypatch.setattr(mod, "generate_reports", _report_wrapper, raising=False)

    report_dir = tmp_path / "reports"
    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(report_dir),
    )

    report_path = Path(report_dir) / "evaluation.report.json"
    assert report_path.exists(), "evaluation report JSON not written"
    report = json.loads(report_path.read_text())

    # Quality Overhead (Primary Metric) should be present with ratio basis > 1.0
    qo = report.get("quality_overhead", {})
    assert isinstance(qo, dict) and qo.get("basis") == "ratio"
    assert float(qo["value"]) > 1.0
    # System Overhead exists and carries latency entry
    sys = report.get("system_overhead", {})
    lat = sys.get("latency_ms_p50", {})
    assert isinstance(lat, dict)
    assert isinstance(lat.get("edited"), int | float)
    assert isinstance(lat.get("baseline"), int | float)
