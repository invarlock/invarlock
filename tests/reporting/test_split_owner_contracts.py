from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core.exceptions import MetricsError
from invarlock.reporting import report_builder_support, run_report_contract
from invarlock.reporting.report_types import create_empty_report


def test_report_builder_support_covers_meta_and_baseline_fallback_paths() -> None:
    class FlickerFloat(float):
        def __new__(cls, *values: float):
            obj = float.__new__(cls, values[0])
            obj._values = iter(values)
            return obj

        def __float__(self) -> float:
            return next(self._values)

    diagnostics: list[dict[str, object]] = []
    meta = report_builder_support.extract_report_meta({"meta": {}}, diagnostics)
    assert meta["seed"] == 0
    assert [entry["code"] for entry in diagnostics] == [
        "meta.model_id_unavailable",
        "meta.adapter_unavailable",
        "meta.device_unavailable",
    ]

    assert report_builder_support.optional_text("  hello  ") == "hello"
    assert report_builder_support.optional_text("   ") is None
    extra_diagnostics: list[dict[str, object]] = []
    report_builder_support.append_build_diagnostic(
        extra_diagnostics,
        code="extra",
        message="details present",
        details={"path": "report.json"},
        severity="error",
    )
    assert extra_diagnostics == [
        {
            "code": "extra",
            "message": "details present",
            "severity": "error",
            "details": {"path": "report.json"},
        }
    ]
    assert report_builder_support.extract_report_meta({"meta": None}, None)["seed"] == 0
    assert report_builder_support.generate_run_id(
        {"meta": {"ts": "2026-03-30", "model_id": "gpt2", "commit": "abc1234567890"}}
    )
    assert (
        report_builder_support.generate_run_id({"meta": {"run_id": "existing-run-id"}})
        == "existing-run-id"
    )
    assert report_builder_support.generate_run_id(
        SimpleNamespace(
            meta={
                "ts": "2026-03-30",
                "model_id": "object-report",
                "commit_sha": "deadbeefcafebabe",
            }
        )
    )

    complete_diagnostics: list[dict[str, object]] = []
    complete_meta = report_builder_support.extract_report_meta(
        {
            "meta": {
                "model_id": "gpt2",
                "adapter": "hf_causal",
                "device": "cpu",
                "seed": 11,
                "seeds": {"python": 11},
            }
        },
        complete_diagnostics,
    )
    assert complete_meta["model_id"] == "gpt2"
    assert complete_meta["adapter"] == "hf_causal"
    assert complete_meta["device"] == "cpu"
    assert complete_meta["seed"] == 11
    assert complete_diagnostics == []

    report = {
        "metrics": {"primary_metric": {"kind": "acc"}},
    }
    assert report_builder_support._direct_baseline_metric(report, None) is None
    assert report_builder_support._direct_baseline_metric(
        report,
        {"primary_metric": {"kind": "acc", "final": 9.0}},
    ) == {"kind": "acc", "final": 9.0}
    assert report_builder_support._direct_baseline_metric(
        report,
        {"ppl_final": 5.0, "ppl_preview": "bad"},
    ) == {"kind": "acc", "preview": 5.0, "final": 5.0}
    assert report_builder_support._direct_baseline_metric(
        report,
        {"ppl_final": 6.0, "ppl_preview": 5.5},
    ) == {"kind": "acc", "preview": 5.5, "final": 6.0}
    assert report_builder_support._direct_baseline_metric(
        report,
        {"metrics": {"primary_metric": {"kind": "acc", "final": 7.0}}},
    ) == {"kind": "acc", "final": 7.0}
    assert report_builder_support._direct_baseline_metric(
        report,
        {"metrics": {"ppl_final": 8.0}},
    ) == {"kind": "acc", "preview": 8.0, "final": 8.0}
    assert report_builder_support._direct_baseline_metric(
        report,
        {"metrics": {"ppl_final": 8.0, "ppl_preview": float("nan")}},
    ) == {"kind": "acc", "preview": 8.0, "final": 8.0}
    assert (
        report_builder_support._direct_baseline_metric(
            report,
            {"ppl_final": 2.0, "ppl_preview": FlickerFloat(1.0, float("nan"))},
        )
        is None
    )
    assert (
        report_builder_support._direct_baseline_metric(
            report,
            {
                "metrics": {
                    "ppl_final": 2.0,
                    "ppl_preview": FlickerFloat(1.0, float("nan")),
                }
            },
        )
        is None
    )
    assert (
        report_builder_support._direct_baseline_metric(
            report,
            {
                "primary_metric": {"kind": "acc", "final": "bad"},
                "metrics": {"primary_metric": {"kind": "acc", "final": "bad"}},
            },
        )
        is None
    )

    baseline_ref = report_builder_support.build_baseline_reference(
        report,
        {
            "evaluation_windows": {"present": True},
            "metrics": {"classification": {"f1": 0.9}},
        },
        {"run_id": "base", "model_id": "baseline", "tokenizer_hash": "tokhash"},
        compute_primary_metric_from_report_fn=lambda payload: {
            "kind": "acc",
            "final": 0.8,
        }
        if "evaluation_windows" in payload
        else {"kind": "acc", "final": 0.7},
    )
    assert baseline_ref["primary_metric"]["final"] == 0.8
    assert baseline_ref["metrics"]["classification"] == {"f1": 0.9}
    assert baseline_ref["tokenizer_hash"] == "tokhash"
    direct_baseline_ref = report_builder_support.build_baseline_reference(
        report,
        {"metrics": {"primary_metric": {"kind": "acc", "final": 0.95}}},
        {"run_id": "direct", "model_id": "baseline"},
    )
    assert direct_baseline_ref["primary_metric"]["final"] == 0.95
    assert "metrics" not in direct_baseline_ref
    assert "tokenizer_hash" not in direct_baseline_ref

    class ExplodingMetrics(dict):
        def get(self, key, default=None):  # type: ignore[override]
            raise ValueError("boom")

    with pytest.raises(MetricsError, match="E233"):
        report_builder_support.build_baseline_reference(
            report,
            {"metrics": ExplodingMetrics()},
            {},
        )

    with pytest.raises(MetricsError, match="E234"):
        report_builder_support.build_baseline_reference(
            report,
            {},
            {},
            compute_primary_metric_from_report_fn=lambda _payload: (
                _ for _ in ()
            ).throw(RuntimeError("no metric")),
        )

    with pytest.raises(MetricsError, match="E235"):
        report_builder_support.build_baseline_reference(
            report,
            {},
            {},
            compute_primary_metric_from_report_fn=lambda _payload: {
                "kind": "acc",
                "final": float("nan"),
            },
        )


def test_run_report_contract_covers_env_collection_and_persistence_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch_mod = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: True,
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(
                allow_tf32=False,
                deterministic=True,
                benchmark=False,
            ),
            mps=SimpleNamespace(is_available=lambda: True),
        ),
    )
    env_flags = run_report_contract._collect_env_flags(
        lambda: torch_mod,
        {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
    )
    assert env_flags == {
        "torch_deterministic_algorithms": True,
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "mps_available": True,
        "CUBLAS_WORKSPACE_CONFIG": ":16:8",
    }
    assert (
        run_report_contract._collect_env_flags(
            lambda: (_ for _ in ()).throw(ValueError("boom")),
            {},
        )
        == {}
    )
    assert run_report_contract._collect_env_flags(
        lambda: SimpleNamespace(
            are_deterministic_algorithms_enabled=False,
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=object()),
                cudnn=None,
                mps=SimpleNamespace(
                    is_available=lambda: (_ for _ in ()).throw(TypeError("mps"))
                ),
            ),
        ),
        {},
    ) == {"CUBLAS_WORKSPACE_CONFIG": None}

    class ExplodingBackends:
        @property
        def cuda(self):
            raise RuntimeError("cuda unavailable")

        @property
        def cudnn(self):
            raise TypeError("cudnn unavailable")

        @property
        def mps(self):
            raise AttributeError("mps unavailable")

    assert run_report_contract._collect_env_flags(
        lambda: SimpleNamespace(
            are_deterministic_algorithms_enabled=lambda: (_ for _ in ()).throw(
                RuntimeError("determinism unavailable")
            ),
            backends=ExplodingBackends(),
        ),
        {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    ) == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "mps_available": False,
    }

    original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "invarlock" and "__version__" in fromlist:
            raise ImportError("version unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert run_report_contract._resolve_version() is None

    core_report = SimpleNamespace(context=None, edit=None, guards=None, metrics=None)
    cfg = SimpleNamespace(
        model=SimpleNamespace(id="gpt2", adapter="hf_causal"),
        dataset=SimpleNamespace(provider=None, dataset="fallback-dataset", seq_len=16),
        meta=SimpleNamespace(commit="abc123"),
    )
    assembled = run_report_contract.assemble_run_report(
        core_report=core_report,
        cfg=cfg,
        run_context=None,
        profile_normalized=None,
        auto_config=None,
        resolved_device="cpu",
        seed_bundle={"python": 7},
        guard_overhead_threshold=0.01,
        model_profile=SimpleNamespace(name="causal"),
        determinism_meta={},
        pm_acceptance_range=None,
        pm_drift_band=None,
        tokenizer_hash=None,
        resolved_split=None,
        preview_count=4,
        final_count=5,
        snapshot_provenance={},
        edit_op=SimpleNamespace(name="noop"),
        edit_label=None,
        run_dir=tmp_path,
        run_config=SimpleNamespace(event_path=tmp_path / "events.jsonl"),
        resolved_loss_type="causal",
        timings={"load": 1.0},
        guard_overhead_payload=None,
        baseline=None,
        preview_records=[],
        final_records=[],
        use_mlm=False,
        preview_mask_counts=None,
        final_mask_counts=None,
        profile=None,
        used_fallback_split=False,
        baseline_report_data=None,
        effective_preview=4,
        effective_final=5,
        metric_kind=None,
        window_plan=None,
        debug_metric_diffs_enabled=False,
        create_empty_report_fn=create_empty_report,
        build_run_report_context_fn=lambda **_kwargs: (_ for _ in ()).throw(
            TypeError("skip context")
        ),
        build_run_report_meta_fn=lambda **kwargs: {"model_id": kwargs["model_id"]},
        canonical_dataset_id_fn=lambda provider: provider,
        safe_int_fn=int,
        build_run_report_data_fn=lambda **kwargs: (
            {
                "dataset": kwargs["canonical_dataset_id"],
                "preview_n": kwargs["preview_count"],
            },
            kwargs["tokenizer_hash"],
        ),
        build_snapshot_provenance_fn=lambda _payload: (_ for _ in ()).throw(
            KeyError("skip provenance")
        ),
        build_edit_payload_fn=lambda **_kwargs: ({}, {}),
        persist_ref_masks_fn=lambda _core_report, run_dir: run_dir / "ref_masks.json",
        build_artifacts_payload_fn=lambda **kwargs: {
            "event_path": str(kwargs["event_path"]),
            "mask_artifact_path": str(kwargs["mask_artifact_path"]),
        },
        merge_core_timing_metrics_fn=lambda timings, metrics: {**timings, **metrics},
        build_metrics_payload_fn=lambda **_kwargs: {"primary_metric": {"final": 1.0}},
        prepare_guard_overhead_report_fn=lambda payload, **_kwargs: payload,
        finalize_run_provenance_fn=lambda **_kwargs: {"ok": True},
        build_guard_entries_fn=lambda guards: [] if guards is None else [{"name": "x"}],
        build_flags_payload_fn=lambda guards: {}
        if guards is None
        else {"all_passed": True},
        enrich_run_report_metrics_fn=lambda **kwargs: kwargs["report"],
        optional_torch_fn=lambda: None,
        environ={},
    )
    assert assembled.report["data"]["dataset"] == "fallback-dataset"
    assert assembled.report["artifacts"]["event_path"].endswith("events.jsonl")
    assert assembled.provenance_result == {"ok": True}

    def _save_report(_report, out_dir, formats=None, filename_prefix="report"):
        _ = formats, filename_prefix
        out = out_dir / "report.json"
        out.write_text("{}", encoding="utf-8")
        return {"json": out}

    monkeypatch.setattr(
        "invarlock.reporting.report_files.save_report",
        _save_report,
    )

    result = run_report_contract.persist_run_report_outputs(
        report=create_empty_report(),
        run_dir=tmp_path,
        run_config=SimpleNamespace(event_path=tmp_path / "events.jsonl"),
        telemetry=True,
        save_telemetry_report_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("telemetry unavailable")
        ),
    )
    assert result.report_path_out == str(tmp_path / "report.json")
    assert result.telemetry_saved_path is None
    assert result.telemetry_error == "telemetry unavailable"

    monkeypatch.setattr(
        "invarlock.reporting.report_files.save_report",
        lambda *_args, **_kwargs: {},
    )
    with pytest.raises(RuntimeError, match="json artifact path"):
        run_report_contract.persist_run_report_outputs(
            report=create_empty_report(),
            run_dir=tmp_path,
            run_config=SimpleNamespace(event_path=tmp_path / "events.jsonl"),
            telemetry=False,
            save_telemetry_report_fn=lambda *_args, **_kwargs: None,
        )
