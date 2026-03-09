from __future__ import annotations

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

    def _exit_stub(
        exc,
        *,
        profile,
        config_error_cls,
        validation_error_cls,
        data_error_cls,
        invarlock_error_cls,
    ):
        seen["profile"] = profile
        seen["config_error_cls"] = config_error_cls
        seen["validation_error_cls"] = validation_error_cls
        seen["data_error_cls"] = data_error_cls
        seen["invarlock_error_cls"] = invarlock_error_cls
        return 3

    monkeypatch.setattr(run_mod, "_resolve_exit_code_impl", _exit_stub)
    assert run_mod._resolve_exit_code(_DummyError("x"), profile="release") == 3
    assert seen["profile"] == "release"


def test_pairing_wrappers_delegate(monkeypatch):
    monkeypatch.setattr(
        run_mod,
        "_extract_pairing_schedule_impl",
        lambda report, tensor_or_list_to_ints_fn: {"preview": {}, "final": {}},
    )
    assert run_mod._extract_pairing_schedule({}) == {"preview": {}, "final": {}}

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


def test_run_command_injects_explicit_deps(monkeypatch):
    sentinel = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        run_mod, "_resolve_pm_acceptance_range", sentinel, raising=False
    )

    def _fake_run_command_impl(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(run_mod, "_run_command_impl", _fake_run_command_impl)
    out = run_mod.run_command(config="configs/example.yml")

    assert out == "ok"
    deps = captured.get("deps")
    assert isinstance(deps, dict)
    assert deps["_resolve_pm_acceptance_range"] is sentinel
    assert deps["console"] is run_mod.console
