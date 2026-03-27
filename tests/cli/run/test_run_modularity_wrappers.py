from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import invarlock.cli.commands.run as run_mod


class _DummyError(Exception):
    pass


def test_policy_wrappers_delegate(monkeypatch):
    monkeypatch.setattr(run_mod, "_coerce_mapping_impl", lambda obj: {"k": "v"})
    assert run_mod._coerce_mapping(object()) == {"k": "v"}

    monkeypatch.setattr(
        run_mod,
        "_resolve_pm_acceptance_range_impl",
        lambda cfg, coerce_mapping_fn: {"min": 0.95, "max": 1.10},
    )
    assert run_mod._resolve_pm_acceptance_range({}) == {"min": 0.95, "max": 1.10}

    monkeypatch.setattr(
        run_mod,
        "_resolve_pm_drift_band_impl",
        lambda cfg, coerce_mapping_fn: {"min": 0.9, "max": 1.2},
    )
    assert run_mod._resolve_pm_drift_band({}) == {"min": 0.9, "max": 1.2}

    seen: dict[str, object] = {}

    def _threshold_stub(cfg, *, default_threshold, coerce_mapping_fn):
        seen["default_threshold"] = default_threshold
        return 0.02

    monkeypatch.setattr(
        run_mod, "_resolve_guard_overhead_threshold_impl", _threshold_stub
    )
    assert run_mod._resolve_guard_overhead_threshold({}) == 0.02
    assert seen["default_threshold"] == run_mod.GUARD_OVERHEAD_THRESHOLD

    monkeypatch.setattr(run_mod, "_coerce_bool_like_impl", lambda value: True)
    assert run_mod._coerce_bool_like("yes") is True

    monkeypatch.setattr(
        run_mod,
        "_resolve_skip_overhead_policy_impl",
        lambda cfg, coerce_mapping_fn: (True, "config:context.run.skip_overhead_check"),
    )
    assert run_mod._resolve_skip_overhead_policy({}) == (
        True,
        "config:context.run.skip_overhead_check",
    )

    monkeypatch.setattr(
        run_mod,
        "_should_measure_overhead_impl",
        lambda profile_normalized, cfg, coerce_mapping_fn: (
            False,
            True,
            "config:context.run.skip_overhead_check",
        ),
    )
    assert run_mod._should_measure_overhead("ci", {}) == (
        False,
        True,
        "config:context.run.skip_overhead_check",
    )

    monkeypatch.setattr(
        run_mod,
        "_resolve_snapshot_config_impl",
        lambda context, to_serialisable_dict_fn: {"mode": "bytes"},
    )
    assert run_mod._resolve_snapshot_config({"snapshot": {"mode": "bytes"}}) == {
        "mode": "bytes"
    }

    monkeypatch.setattr(run_mod, "_estimate_model_bytes_impl", lambda model: 123)
    assert run_mod._estimate_model_bytes(object()) == 123

    monkeypatch.setattr(
        run_mod,
        "_choose_snapshot_mode_impl",
        lambda **kwargs: "chunked",
    )
    assert (
        run_mod._choose_snapshot_mode(
            snapshot_config={},
            env_mode="auto",
            supports_bytes=True,
            supports_chunked=True,
            estimated_model_mb=128.0,
            available_ram_mb=256.0,
            disk_free_mb=1024.0,
        )
        == "chunked"
    )

    monkeypatch.setattr(
        run_mod,
        "_build_timing_summary_payload_impl",
        lambda **kwargs: "summary",
    )
    assert (
        run_mod._build_timing_summary_payload(
            timings={"load_model": 1.0},
            total_duration=2.0,
            report={"metrics": {}},
        )
        == "summary"
    )

    monkeypatch.setattr(
        run_mod,
        "_serialize_evaluation_windows_impl",
        lambda evaluation_windows: {"preview": {}, "final": {}},
    )
    assert run_mod._serialize_evaluation_windows({"preview": {}}) == {
        "preview": {},
        "final": {},
    }

    monkeypatch.setattr(
        run_mod,
        "_build_fallback_evaluation_windows_impl",
        lambda preview_records, final_records, **kwargs: {
            "preview": {"window_ids": [0]},
            "final": {"window_ids": [1]},
        },
    )
    assert run_mod._build_fallback_evaluation_windows([], [], use_mlm=False) == {
        "preview": {"window_ids": [0]},
        "final": {"window_ids": [1]},
    }

    monkeypatch.setattr(
        run_mod,
        "_finalize_guard_overhead_payload_impl",
        lambda payload, result: {"passed": True},
    )
    assert run_mod._finalize_guard_overhead_payload({}, object()) == {"passed": True}

    split_seen: dict[str, object] = {}

    def _split_stub(*, requested, available, split_aliases):
        split_seen["split_aliases"] = tuple(split_aliases)
        return "validation", True

    monkeypatch.setattr(run_mod, "_choose_dataset_split_impl", _split_stub)
    assert run_mod._choose_dataset_split(requested=None, available=["test"]) == (
        "validation",
        True,
    )
    assert split_seen["split_aliases"] == run_mod.SPLIT_ALIASES


def test_artifact_wrappers_delegate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_mod,
        "_persist_ref_masks_impl",
        lambda core_report, run_dir: (
            run_dir / "artifacts" / "edit_masks" / "masks.json"
        ),
    )
    assert run_mod._persist_ref_masks({}, tmp_path).name == "masks.json"

    seen: dict[str, object] = {}

    def _exit_stub(exc, *, profile):
        seen["profile"] = profile
        return 3

    monkeypatch.setattr(run_mod, "_resolve_command_exit_code", _exit_stub)
    assert run_mod._resolve_exit_code(_DummyError("x"), profile="release") == 3
    assert seen["profile"] == "release"

    monkeypatch.setattr(
        run_mod,
        "_build_retry_result_summary_impl",
        lambda validation: {"passed": True, "failures": [], "validation": validation},
    )
    assert run_mod._build_retry_result_summary({"ok": True}) == {
        "passed": True,
        "failures": [],
        "validation": {"ok": True},
    }

    monkeypatch.setattr(
        run_mod,
        "_apply_mask_only_head_autotune_impl",
        lambda edit_config, validation: ({"heads": {"global_k": 6}}, {"global_k": 6}),
    )
    assert run_mod._apply_mask_only_head_autotune({}, {"ok": False}) == (
        {"heads": {"global_k": 6}},
        {"global_k": 6},
    )


def test_pairing_wrappers_delegate(monkeypatch):
    monkeypatch.setattr(
        run_mod,
        "_extract_pairing_schedule_impl",
        lambda report, tensor_or_list_to_ints_fn: {"preview": {}, "final": {}},
    )
    assert run_mod._extract_pairing_schedule({}) == {"preview": {}, "final": {}}

    monkeypatch.setattr(
        run_mod,
        "_load_baseline_pairing_evidence_impl",
        lambda **kwargs: {"status": "loaded"},
    )
    assert run_mod._load_baseline_pairing_evidence(
        baseline_path=Path("baseline.json"),
        tokenizer_hash="tok",
    ) == {"status": "loaded"}

    monkeypatch.setattr(
        run_mod,
        "_materialize_baseline_pairing_schedule_impl",
        lambda **kwargs: {"preview_count": 1, "final_count": 1},
    )
    assert run_mod._materialize_baseline_pairing_schedule(
        pairing_schedule={"preview": {}, "final": {}},
        calibration_data=[],
        dataset_meta={},
        window_plan=None,
        tokenizer=None,
        use_mlm=False,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        resolved_tier="balanced",
        profile="dev",
    ) == {"preview_count": 1, "final_count": 1}

    monkeypatch.setattr(
        run_mod,
        "_compute_provider_digest_impl",
        lambda report, compute_mask_positions_digest_fn: {"ids_sha256": "abc"},
    )
    assert run_mod._compute_provider_digest({}) == {"ids_sha256": "abc"}

    monkeypatch.setattr(
        run_mod,
        "_validate_and_harvest_baseline_schedule_impl",
        lambda *args, **kwargs: {"dataset_id": "d", "tokenizer_hash": "h"},
    )
    out = run_mod._validate_and_harvest_baseline_schedule(
        {},
        {},
        {},
        tokenizer_hash="tok",
        resolved_loss_type="ppl_causal",
    )
    assert out == {"dataset_id": "d", "tokenizer_hash": "h"}

    seen: dict[str, object] = {}

    def _parity_stub(subject_digest, baseline_digest, *, profile, invarlock_error_cls):
        seen["profile"] = profile

    monkeypatch.setattr(run_mod, "_enforce_provider_parity_impl", _parity_stub)
    run_mod._enforce_provider_parity(
        {"ids_sha256": "a"}, {"ids_sha256": "a"}, profile="ci"
    )
    assert seen["profile"] == "ci"

    monkeypatch.setattr(
        run_mod,
        "_resolve_metric_and_provider_impl",
        lambda cfg, model_profile, resolved_loss_type, metric_kind_override: (
            "ppl_causal",
            "wikitext2",
            {"threshold": 1.1},
        ),
    )
    assert run_mod._resolve_metric_and_provider({}, SimpleNamespace()) == (
        "ppl_causal",
        "wikitext2",
        {"threshold": 1.1},
    )


def test_config_wrappers_delegate(monkeypatch):
    monkeypatch.setattr(run_mod, "_prepare_config_for_run_impl", lambda **kwargs: "cfg")
    assert (
        run_mod._prepare_config_for_run(
            config_path="cfg.yaml",
            profile="ci",
            edit=None,
            tier=None,
            probes=None,
            console=run_mod.console,
        )
        == "cfg"
    )

    monkeypatch.setattr(
        run_mod,
        "_resolve_device_and_output_impl",
        lambda *args, **kwargs: ("cpu", kwargs["out"]),
    )
    device, out = run_mod._resolve_device_and_output(
        SimpleNamespace(model=SimpleNamespace(device="cpu")),
        device=None,
        out="runs",
        console=run_mod.console,
    )
    assert device == "cpu"
    assert str(out) == "runs"

    monkeypatch.setattr(
        run_mod,
        "_resolve_provider_and_split_impl",
        lambda *args, **kwargs: ("provider", "validation", True),
    )
    assert run_mod._resolve_provider_and_split(
        {},
        SimpleNamespace(default_provider="wikitext2"),
        get_provider_fn=lambda *a, **k: None,
        console=run_mod.console,
    ) == ("provider", "validation", True)

    seen: dict[str, object] = {}

    def _load_kwargs_stub(cfg, *, invarlock_error_cls):
        seen["err_cls"] = invarlock_error_cls
        return {"dtype": "float16"}

    monkeypatch.setattr(run_mod, "_extract_model_load_kwargs_impl", _load_kwargs_stub)
    assert run_mod._extract_model_load_kwargs(SimpleNamespace()) == {"dtype": "float16"}
    assert seen["err_cls"] is run_mod.InvarlockError


def test_analysis_and_overhead_wrappers_delegate(monkeypatch):
    monkeypatch.setattr(
        run_mod,
        "_plan_release_windows_impl",
        lambda *args, **kwargs: {
            "actual_preview": 100,
            "actual_final": 100,
            "coverage_ok": True,
        },
    )
    out = run_mod._plan_release_windows(
        {"available_unique": 1000},
        requested_preview=100,
        requested_final=100,
        max_calibration=20,
        console=run_mod.console,
    )
    assert out["coverage_ok"] is True

    monkeypatch.setattr(
        run_mod,
        "_merge_primary_metric_health_impl",
        lambda primary_metric, core_primary_metric: {"degraded": True},
    )
    assert run_mod._merge_primary_metric_health({}, {}) == {"degraded": True}

    monkeypatch.setattr(
        run_mod,
        "_format_debug_metric_diffs_impl",
        lambda pm, metrics, baseline_report_data: "diffs",
    )
    assert run_mod._format_debug_metric_diffs({}, {}, {}) == "diffs"

    monkeypatch.setattr(
        run_mod,
        "_normalize_overhead_result_impl",
        lambda payload: {"evaluated": False, "passed": True},
    )
    assert run_mod._normalize_overhead_result({}) == {
        "evaluated": False,
        "passed": True,
    }

    monkeypatch.setattr(
        run_mod,
        "_build_provider_dataset_plan_impl",
        lambda **kwargs: {"resolved_split": "validation", "preview_count": 2},
    )
    assert run_mod._build_provider_dataset_plan(
        cfg={},
        model_profile=SimpleNamespace(),
        console=run_mod.console,
        resolved_device="cpu",
        profile="dev",
        profile_normalized="dev",
        requested_preview=2,
        requested_final=2,
        effective_preview=2,
        effective_final=2,
        pairing_schedule_present=False,
        use_mlm=False,
        mask_prob=0.15,
        mask_seed=43,
        random_token_prob=0.1,
        original_token_prob=0.1,
        resolved_loss_type="ppl_causal",
        tier="balanced",
        get_provider_fn=lambda *args, **kwargs: None,
    ) == {"resolved_split": "validation", "preview_count": 2}

    monkeypatch.setattr(
        run_mod,
        "_validate_retry_evaluation_report_impl",
        lambda **kwargs: {"status": "passed"},
    )
    assert run_mod._validate_retry_evaluation_report(
        report={},
        baseline_report_data=None,
        baseline_path=Path("baseline.json"),
    ) == {"status": "passed"}

    monkeypatch.setattr(
        run_mod,
        "_build_run_context_payload_impl",
        lambda **kwargs: {"profile": "ci", "run_id": "run-1"},
    )
    assert run_mod._build_run_context_payload(
        cfg={},
        profile="ci",
        pairing_schedule=None,
        seed_bundle={"python": 43},
        plugin_provenance={},
        run_id="run-1",
        baseline_report_data=None,
        pm_acceptance_range=(0.9, 1.1),
        pm_drift_band=None,
        guard_overhead_threshold=0.02,
        model_profile=SimpleNamespace(),
        resolved_loss_type="ppl_causal",
        tiny_relax_enabled=False,
    ) == {"profile": "ci", "run_id": "run-1"}

    monkeypatch.setattr(
        run_mod,
        "_build_run_execution_config_payloads_impl",
        lambda **kwargs: SimpleNamespace(auto_config={"enabled": True}, edit_config={}),
    )
    payloads = run_mod._build_run_execution_config_payloads(
        cfg={},
        model_profile=SimpleNamespace(),
    )
    assert payloads.auto_config == {"enabled": True}

    monkeypatch.setattr(
        run_mod,
        "_enrich_run_report_metrics_impl",
        lambda **kwargs: SimpleNamespace(
            report=kwargs["report"],
            pairing_violations=(),
            debug_diffs_line="diffs",
            match_fraction=1.0,
            overlap_fraction=0.0,
        ),
    )
    enriched = run_mod._enrich_run_report_metrics(
        report={"metrics": {}},
        core_report=SimpleNamespace(),
        run_config=SimpleNamespace(context={}),
        cfg=SimpleNamespace(dataset=SimpleNamespace()),
        model_profile=SimpleNamespace(),
        baseline_requested=False,
        baseline_report_data=None,
        metric_kind="ppl_causal",
        resolved_loss_type="ppl_causal",
        effective_preview=1,
        effective_final=1,
        profile_normalized="dev",
        window_plan=None,
        debug_metric_diffs_enabled=True,
    )
    assert enriched.debug_diffs_line == "diffs"


def test_run_command_injects_explicit_deps(monkeypatch, tmp_path: Path):
    sentinel = object()
    captured: dict[str, object] = {}
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        run_mod, "_resolve_pm_acceptance_range", sentinel, raising=False
    )

    def _fake_run_command_impl(**kwargs):
        captured.update(kwargs)
        return str(report_path)

    monkeypatch.setattr(run_mod, "_run_command_impl", _fake_run_command_impl)
    out = run_mod.run_command(config="configs/example.yml")

    assert out == report_path.resolve()
    deps = captured.get("deps")
    assert isinstance(deps, dict)
    assert deps["_resolve_pm_acceptance_range"] is sentinel
    assert deps["_build_provider_dataset_plan"] is run_mod._build_provider_dataset_plan
    assert (
        deps["_materialize_baseline_pairing_schedule"]
        is run_mod._materialize_baseline_pairing_schedule
    )
    assert deps["_build_run_context_payload"] is run_mod._build_run_context_payload
    assert (
        deps["_build_run_execution_config_payloads"]
        is run_mod._build_run_execution_config_payloads
    )
    assert deps["_enrich_run_report_metrics"] is run_mod._enrich_run_report_metrics
    assert (
        deps["_validate_retry_evaluation_report"]
        is run_mod._validate_retry_evaluation_report
    )
    assert deps["_choose_snapshot_mode"] is run_mod._choose_snapshot_mode
    assert (
        deps["_build_timing_summary_payload"] is run_mod._build_timing_summary_payload
    )
    assert (
        deps["_apply_mask_only_head_autotune"] is run_mod._apply_mask_only_head_autotune
    )
    assert deps["console"] is run_mod.console
    assert callable(deps["get_torch"])
    assert callable(deps["get_psutil"])


def test_build_run_command_deps_keeps_optional_modules_lazy(monkeypatch):
    calls = {"torch": 0, "psutil": 0}

    def _fake_get_torch():
        calls["torch"] += 1
        return object()

    def _fake_get_psutil():
        calls["psutil"] += 1
        return object()

    monkeypatch.setattr(run_mod, "_get_torch", _fake_get_torch)
    monkeypatch.setattr(run_mod, "_get_psutil", _fake_get_psutil)

    deps = run_mod._build_run_command_deps()

    assert callable(deps["get_torch"])
    assert callable(deps["get_psutil"])
    assert calls == {"torch": 0, "psutil": 0}
    assert deps["get_torch"]() is not None
    assert deps["get_psutil"]() is not None
    assert calls == {"torch": 1, "psutil": 1}
