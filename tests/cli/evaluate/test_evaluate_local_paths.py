from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from invarlock.cli.commands.evaluate import evaluate_command
from invarlock.eval.guard_metric_impact import (
    REQUIRED_GUARD_METRIC_IMPACT_CHECKS,
    build_guard_metric_bare_report,
    compute_guard_metric_impact,
    extract_guard_metric_arm_facts,
    guard_metric_schedule_digest,
)
from tests.cli._support_effective_config import preserve_effective_config
from tests.cli._support_runtime_policy import bind_runtime_policy


def _write_run_report(
    dir_: Path,
    *,
    pm_kind: str,
    pm_final: float,
    pm_preview: float,
    ratio_vs_baseline: float,
    latency_ms_per_tok: float,
    provider_ids_digest: str,
    guard_metric_impact: dict | None = None,
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
            # Direct run-report metric fields used by CLI printing paths.
            "ppl_preview": pm_preview,
            "ppl_final": pm_final,
            "ppl_ratio": ratio_vs_baseline,
            "latency_ms_per_tok": latency_ms_per_tok,
        },
        "provenance": {"provider_digest": {"ids_sha256": provider_ids_digest}},
        "evaluation_windows": {
            "final": {
                "window_ids": [1],
                # Serialized run reports retain the unused ID family as an empty
                # list; PPL binding must still use the non-empty window IDs.
                "example_ids": [],
                "logloss": [math.log(pm_final)],
                "token_counts": [1],
                "input_ids": [[1, 2]],
                "attention_masks": [[1, 1]],
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    if guard_metric_impact:
        report["guard_metric_impact"] = guard_metric_impact
    bind_runtime_policy(report, profile="ci")
    (ts_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return ts_dir / "report.json"


def _attach_canonical_metric_impact(
    report: dict, *, bare_value: float, degradation_limit: float
) -> None:
    metric_kind = report["metrics"]["primary_metric"]["kind"]
    guarded_value = report["metrics"]["primary_metric"]["final"]
    final_ids = report["evaluation_windows"]["final"]["window_ids"]
    bare_arm = {
        "primary_metric": {"kind": metric_kind, "final": bare_value},
        "final": {
            "window_ids": list(final_ids),
            "logloss": [math.log(bare_value)],
            "token_counts": [1],
        },
        "status": "success",
    }
    bare_report = build_guard_metric_bare_report(bare_arm, metric_kind)
    bare_facts = extract_guard_metric_arm_facts(bare_arm, metric_kind)
    guarded_facts = extract_guard_metric_arm_facts(report, metric_kind)
    measurement = compute_guard_metric_impact(
        metric_kind,
        bare_value,
        guarded_value,
    )
    schedule_digest = guard_metric_schedule_digest(report, metric_kind)
    assert bare_report is not None
    assert bare_facts is not None
    assert guarded_facts is not None
    assert measurement is not None
    assert schedule_digest is not None
    report["guard_metric_impact"] = {
        "evaluated": True,
        "passed": measurement.degradation <= degradation_limit,
        **measurement.to_metrics(),
        "degradation_limit": degradation_limit,
        "bare_facts": bare_facts,
        "guarded_facts": guarded_facts,
        "bare_report": bare_report,
        "checks": dict.fromkeys(REQUIRED_GUARD_METRIC_IMPACT_CHECKS, True),
        "diagnostics": [],
        "source": "strict_fixture",
        "schedule_digest": schedule_digest,
    }


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
        preserve_effective_config(kwargs)
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
        render_optional: bool,
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
def test_evaluate_local_paths_quantized_subject_impact_and_system_overhead(
    monkeypatch, tmp_path: Path
):
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
        preserve_effective_config(kwargs)
        out_dir = Path(kwargs.get("out"))
        if Path(kwargs.get("baseline") or "").exists():
            # Edited run: slightly worse ppl; slower
            path = _write_run_report(
                out_dir,
                pm_kind="ppl_causal",
                pm_final=10.5,
                pm_preview=10.0,
                ratio_vs_baseline=1.05,
                latency_ms_per_tok=0.90,
                provider_ids_digest="editedq123",
            )
            # Explicit latency p50 to stabilize System Overhead ratio
            data = json.loads(path.read_text())
            data.setdefault("metrics", {})["latency_ms_p50"] = 0.90
            _attach_canonical_metric_impact(
                data,
                bare_value=10.0,
                degradation_limit=0.10,
            )
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
        render_optional: bool,
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

    # Guard metric impact should record a positive relative increase.
    qo = report.get("guard_metric_impact", {})
    assert isinstance(qo, dict)
    assert qo.get("degradation_basis") == "relative_increase"
    assert float(qo["degradation"]) == pytest.approx(0.05)
    # System Overhead exists and carries latency entry
    sys = report.get("system_overhead", {})
    lat = sys.get("latency_ms_p50", {})
    assert isinstance(lat, dict)
    assert isinstance(lat.get("edited"), int | float)
    assert isinstance(lat.get("baseline"), int | float)
