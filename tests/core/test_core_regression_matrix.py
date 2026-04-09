from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import invarlock.core.adapter_auto as adapter_auto_mod
import invarlock.core.auto_tuning as auto_tuning_mod
import invarlock.core.bootstrap as bootstrap_mod
import invarlock.core.config_runtime as config_mod
import invarlock.core.metric_provider_resolution as metric_provider_mod
import invarlock.core.run_guard_overhead_policy as run_guard_overhead_mod
import invarlock.core.run_report_payload_policy as run_report_payload_mod
import invarlock.core.run_snapshot_contract as run_snapshot_contract_mod
import invarlock.core.run_timing_policy as run_timing_policy_mod
import invarlock.core.runner_context as runner_context_mod
import invarlock.core.runner_eval_metrics_stats as runner_eval_stats
import invarlock.core.runtime_manifest_verify as runtime_manifest_verify_mod
import invarlock.core.types as core_types_mod
from invarlock.core.api import RunConfig, RunReport
from invarlock.core.doctor_preflight import run_doctor_config_preflight
from invarlock.core.events import EventLogger
from invarlock.core.plugins_inventory import (
    gather_adapter_inventory_rows,
    gather_generic_inventory_rows,
)
from invarlock.core.runner import CoreRunner
from invarlock.core.runner_guards import (
    _coerce_diagnostics,
    _normalize_guard_result,
    resolve_guard_policies,
)
from invarlock.core.types import RunStatus


def test_compute_paired_delta_log_ci_returns_zero_for_empty_trimmed_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = iter(
        [
            np.asarray([1.0], dtype=float),
            np.asarray([], dtype=float),
        ]
    )
    monkeypatch.setattr(bootstrap_mod, "_ensure_array", lambda _samples: next(arrays))

    assert bootstrap_mod.compute_paired_delta_log_ci([1.0], [2.0]) == (0.0, 0.0)


@dataclass
class _NoExtraSection(config_mod.SectionMixin):
    name: str = "section"
    _extra: object | None = None


def test_config_runtime_helpers_cover_non_dict_extra_and_bootstrap_passthrough() -> (
    None
):
    section = _NoExtraSection(_extra=None)

    assert config_mod._section_dataclass_payload(section) == {"name": "section"}
    with pytest.raises(KeyError):
        section["other"] = 1

    bootstrap = config_mod.EvalBootstrapConfig(replicates=8)
    cfg = config_mod.EvalConfig(bootstrap=bootstrap, loss="keep-as-is")

    assert cfg.bootstrap is bootstrap
    assert cfg.loss == "keep-as-is"


class _TruthyEmpty(list):
    def __bool__(self) -> bool:
        return True


class _FakeEvalRunner:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, dict[str, object]]] = []

    def _log_event(
        self,
        component: str,
        operation: str,
        level: str,
        data: dict[str, object] | None = None,
    ) -> None:
        self.events.append((component, operation, level, data or {}))


def _runtime(**overrides: object) -> object:
    payload = {
        "bootstrap_enabled": False,
        "bootstrap_replicates": 8,
        "bootstrap_alpha": 0.05,
        "bootstrap_seed": 11,
        "single_method": "bca",
        "delta_method": "paired",
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _slices(**overrides: object) -> object:
    payload = {
        "preview_mean_log": 0.1,
        "final_mean_log": 0.2,
        "delta_mean_log": 0.1,
        "ppl_ratio": 1.1,
        "pm_invalid": False,
        "preview_log_losses": [0.1, 0.2],
        "final_log_losses": [0.3, 0.4],
        "preview_token_counts": [2, 3],
        "final_token_counts": [5, 7],
        "pm_preview": 1.0,
        "pm_final": 1.1,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_compute_bootstrap_delta_stats_covers_truthy_empty_and_final_count_fallback() -> (
    None
):
    no_pairs = runner_eval_stats._compute_bootstrap_delta_stats(
        _FakeEvalRunner(),
        _runtime(),
        _slices(
            preview_log_losses=_TruthyEmpty(),
            final_log_losses=_TruthyEmpty(),
            preview_token_counts=[],
            final_token_counts=[],
        ),
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.0),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.0),
    )
    assert no_pairs.degenerate_reason == "no_pairs"

    weighted = runner_eval_stats._compute_bootstrap_delta_stats(
        _FakeEvalRunner(),
        _runtime(),
        _slices(preview_token_counts=[], final_token_counts=[5, 7]),
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.0),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.0),
    )
    assert weighted.delta_weights == [5.0, 7.0]


def _patch_doctor_preflight_common(
    monkeypatch: pytest.MonkeyPatch, cfg: object
) -> None:
    monkeypatch.setattr(
        "invarlock.core.config_runtime.load_config", lambda _path: cfg, raising=False
    )
    monkeypatch.setattr(
        "invarlock.core.config_runtime.apply_profile",
        lambda cfg_obj, _profile: cfg_obj,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.detect_model_profile",
        lambda **_kwargs: SimpleNamespace(default_loss="classification"),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.model_profile.resolve_tokenizer",
        lambda _profile: (
            SimpleNamespace(__class__=SimpleNamespace(__name__="Tok")),
            "tok",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.metric_provider_resolution.resolve_metric_and_provider",
        lambda *_args, **_kwargs: ("accuracy", "synthetic", {}),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda _kind: SimpleNamespace(
            estimate_capacity=lambda **_kwargs: {
                "available_nonoverlap": 2,
                "tokens_available": 16,
                "examples_available": 1,
            }
        ),
        raising=False,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _base_doctor_cfg() -> object:
    return SimpleNamespace(
        dataset=SimpleNamespace(
            provider={"kind": "synthetic", "workers": 0, "deterministic_shards": True},
            seq_len=16,
            stride=8,
            preview_n=1,
            final_n=1,
            split="validation",
        ),
        model=SimpleNamespace(adapter="hf_fake", id="fake-model", device="cpu"),
    )


def test_doctor_preflight_records_eval_section_and_attribute_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _SectionFailCfg:
        def __init__(self) -> None:
            base = _base_doctor_cfg()
            self.dataset = base.dataset
            self.model = base.model

        def section(self, _name: str) -> object:
            raise RuntimeError("section boom")

    section_cfg = _SectionFailCfg()
    _patch_doctor_preflight_common(monkeypatch, section_cfg)
    section_result = run_doctor_config_preflight(
        config_path=str(tmp_path / "cfg.yaml"),
        profile="dev",
        tier="balanced",
        baseline=None,
    )
    assert "D019" in {finding.code for finding in section_result.findings}
    assert section_result.had_error is True

    class _AttrFailCfg:
        def __init__(self) -> None:
            base = _base_doctor_cfg()
            self.dataset = base.dataset
            self.model = base.model

        @property
        def eval(self) -> object:
            raise RuntimeError("attr boom")

    attr_cfg = _AttrFailCfg()
    _patch_doctor_preflight_common(monkeypatch, attr_cfg)
    attr_result = run_doctor_config_preflight(
        config_path=str(tmp_path / "cfg-attr.yaml"),
        profile="dev",
        tier="balanced",
        baseline=None,
    )
    assert "D019" in {finding.code for finding in attr_result.findings}
    assert attr_result.had_error is True


class _BadListLike:
    def tolist(self) -> list[object]:
        raise RuntimeError("boom")


class _BadSet(set[int]):
    def __iter__(self):  # type: ignore[override]
        raise RuntimeError("boom")


def test_event_logger_handles_serializer_fallbacks_and_destructor_without_file(
    tmp_path: Path,
) -> None:
    logger = EventLogger(tmp_path / "events.jsonl", auto_flush=False)

    payload = logger._sanitize_data({"array_like": _BadListLike(), "items": _BadSet()})
    assert payload["array_like"] == "<_BadListLike>"
    assert payload["items"] == "<_BadSet>"
    assert isinstance(logger._json_serializer(object()), str)

    logger.log_error("runner", "oops", RuntimeError("boom"), context=None)
    logger.close()

    orphan = EventLogger.__new__(EventLogger)
    orphan.__del__()


class _InventoryRegistry:
    def list_adapters(self) -> list[str]:
        return ["hf_bnb"]

    def list_guards(self) -> list[str]:
        return ["remote_guard"]

    def list_edits(self) -> list[str]:
        return []

    def get_plugin_info(self, name: str, kind: str) -> dict[str, str]:
        mapping = {
            ("hf_bnb", "adapters"): {
                "module": "invarlock.plugins.bitsandbytes",
                "entry_point": "bnb",
            },
            ("remote_guard", "guards"): {
                "module": "vendor.guard",
                "entry_point": "remote",
            },
        }
        return mapping[(name, kind)]


def test_plugins_inventory_covers_runtime_enable_and_missing_hint_paths() -> None:
    registry = _InventoryRegistry()

    adapter_rows = gather_adapter_inventory_rows(
        registry=registry,
        minimal=False,
        has_cuda=False,
        is_linux=True,
        extras_checker=lambda _name, _kind: "",
        provenance_extractor=lambda _name: SimpleNamespace(
            library="bitsandbytes",
            version="0.1",
        ),
        bitsandbytes_runtime_available=lambda: False,
    )
    assert adapter_rows[0]["status"] == "unsupported"
    assert (
        adapter_rows[0]["enable"]
        == "Requires CUDA or a compatible bitsandbytes runtime"
    )

    guard_rows = gather_generic_inventory_rows(
        registry=registry,
        plugin_type="guards",
        extras_checker=lambda _name, _kind: "⚠️ missing",
    )
    assert guard_rows[0]["status"] == "needs_extra"
    assert guard_rows[0]["enable"] == ""


def test_finalize_phase_covers_missing_preview_and_warn_mode_tail() -> None:
    runner = CoreRunner()
    report = RunReport()
    cfg = RunConfig(max_pm_ratio=1.01, spike_threshold=1.05)

    missing_preview_status = runner._finalize_phase(
        object(),
        object(),
        {"spectral": {"passed": True}},
        {"primary_metric": {"kind": "ppl_causal", "final": 2.0}},
        cfg,
        report,
    )
    assert missing_preview_status == RunStatus.SUCCESS.value

    warn_tail_status = runner._finalize_phase(
        object(),
        object(),
        {"spectral": {"passed": True}},
        {
            "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.01},
            "primary_metric_tail": {"mode": "warn", "evaluated": True, "passed": False},
        },
        cfg,
        RunReport(),
    )
    assert warn_tail_status == RunStatus.SUCCESS.value


def test_runner_guard_helpers_cover_skipped_items_and_meta_auto_resolution() -> None:
    assert _coerce_diagnostics([123]) == []

    normalized = _normalize_guard_result(
        {"passed": False, "decision": "monitor", "violations": "not-a-sequence"}
    )
    assert normalized["decision"] == "monitor"
    assert normalized["violations"] == []

    seen: dict[str, object] = {}

    def _resolver(
        tier: str, edit_name: str | None, overrides: dict[str, object]
    ) -> dict[str, dict[str, object]]:
        seen["tier"] = tier
        seen["edit_name"] = edit_name
        seen["overrides"] = dict(overrides)
        return {"spectral": {"deadband": 0.1}}

    report = RunReport(meta={"config": {"auto": {"tier": "conservative"}}})
    runner = SimpleNamespace(
        _log_event=lambda *_args, **_kwargs: None,
    )
    resolve_guard_policies(runner, report, None, resolver=_resolver)

    assert seen["tier"] == "conservative"
    assert seen["edit_name"] is None
    assert seen["overrides"] == {}

    seen.clear()
    report_fallback = RunReport(meta={"config": "bad"})
    report_fallback.__dict__["auto_config"] = "bad"
    resolve_guard_policies(runner, report_fallback, None, resolver=_resolver)
    assert seen["tier"] == "balanced"


def test_adapter_auto_helpers_cover_quant_fallbacks_and_apply_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadQuantConfig(dict):
        def get(self, key: object, default: object = None) -> object:  # type: ignore[override]
            if key in {"quant_method", "quant_method_full"}:
                raise TypeError("boom")
            return super().get(key, default)

    assert (
        adapter_auto_mod._detect_quant_family_from_cfg(  # noqa: SLF001
            {"quantization_config": _BadQuantConfig()}
        )
        is None
    )
    assert (
        adapter_auto_mod._detect_quant_family_from_cfg(  # noqa: SLF001
            {"quantization_config": "bnb"}
        )
        is None
    )
    monkeypatch.setattr(
        adapter_auto_mod,
        "_read_local_hf_config",
        lambda _model_id: {"architectures": ["ToyForCausalLM"], "model_type": "toy"},
    )
    assert adapter_auto_mod.resolve_auto_adapter("toy/model") == "hf_causal"

    class _Cfg:
        def __init__(self) -> None:
            self.model = SimpleNamespace(adapter="auto", id="model")

        def model_dump(self) -> dict[str, object]:
            raise TypeError("boom")

    cfg = _Cfg()
    monkeypatch.setattr(
        adapter_auto_mod,
        "resolve_auto_adapter",
        lambda _model_id, default="hf_causal": default,
    )
    assert adapter_auto_mod.apply_auto_adapter_if_needed(cfg) is cfg


def test_metric_provider_resolution_and_helper_normalizers_cover_fallbacks() -> None:
    class _BrokenMetric:
        def get(self, _key: str) -> object:
            raise TypeError("boom")

        def __getattr__(self, _name: str) -> object:
            raise AttributeError("missing")

    class _Dataset:
        @property
        def provider(self) -> object:
            raise TypeError("boom")

    class _Cfg:
        dataset = _Dataset()

        def section(self, _name: str) -> object:
            return {"metric": _BrokenMetric()}

    metric_kind, provider_kind, metric_opts = (
        metric_provider_mod.resolve_metric_and_provider(
            _Cfg(),
            SimpleNamespace(default_provider="default-provider", default_metric=None),
            resolved_loss_type="seq2seq",
        )
    )
    assert (metric_kind, provider_kind, metric_opts) == (
        "ppl_seq2seq",
        "default-provider",
        {},
    )

    cfg2 = SimpleNamespace(
        dataset=SimpleNamespace(provider=None),
        section=lambda _name: {
            "metric": {"kind": "accuracy", "reps": "bad", "ci_level": "bad"}
        },
    )
    metric_kind2, provider_kind2, metric_opts2 = (
        metric_provider_mod.resolve_metric_and_provider(
            cfg2,
            SimpleNamespace(default_provider=None, default_metric=None),
            resolved_loss_type="classification",
        )
    )
    assert (metric_kind2, provider_kind2, metric_opts2) == (
        "accuracy",
        "wikitext2",
        {},
    )

    assert auto_tuning_mod.normalize_family_caps(
        {"skip": "x", "bad_dict": {"kappa": "nope"}, "ok": 1.5}
    ) == {"ok": {"kappa": 1.5}}
    assert auto_tuning_mod._normalize_multiple_testing(  # noqa: SLF001
        {"method": "BH", "alpha": "bad", "m": "bad"}
    ) == {"method": "bh"}
    assert auto_tuning_mod._normalize_multiple_testing({"method": "BH"}) == {  # noqa: SLF001
        "method": "bh"
    }

    class _NonMappingCfg:
        dataset = SimpleNamespace(provider=None)

        def section(self, _name: str) -> object:
            return "bad"

    metric_kind3, provider_kind3, metric_opts3 = (
        metric_provider_mod.resolve_metric_and_provider(
            _NonMappingCfg(),
            SimpleNamespace(default_provider=None, default_metric=None),
            resolved_loss_type="causal",
        )
    )
    assert (metric_kind3, provider_kind3, metric_opts3) == (
        "ppl_causal",
        "wikitext2",
        {},
    )

    class _SectionBoomCfg:
        dataset = SimpleNamespace(provider=None)

        def section(self, _name: str) -> object:
            raise TypeError("boom")

    metric_kind4, provider_kind4, metric_opts4 = (
        metric_provider_mod.resolve_metric_and_provider(
            _SectionBoomCfg(),
            SimpleNamespace(default_provider=None, default_metric=None),
            resolved_loss_type="causal",
        )
    )
    assert (metric_kind4, provider_kind4, metric_opts4) == (
        "ppl_causal",
        "wikitext2",
        {},
    )


def test_run_payload_snapshot_and_runtime_helpers_cover_remaining_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_edit, context_edit = run_report_payload_mod.build_edit_payload(
        core_edit={"deltas": "bad"},
        edit_name="quant_rtn",
        edit_label=None,
    )
    assert report_edit["deltas"]["params_changed"] == 0
    assert context_edit is not None
    assert context_edit["params_changed"] == 0

    metrics_payload = run_report_payload_mod.build_metrics_payload(
        core_metrics={"stats": [], "primary_metric_tail": {"passed": True}},
        resolved_loss_type=None,
        dataset_meta_context={"loss_type": "classification", "masked_tokens_total": 5},
        window_plan_context={"capacity": {"tokens": 3}, "requested_preview": 1},
    )
    assert metrics_payload["window_capacity"] == {"tokens": 3}
    assert metrics_payload["stats"]["requested_preview"] == 1
    assert metrics_payload["loss_type"] == "classification"
    assert metrics_payload["masked_tokens_total"] == 5
    assert run_report_payload_mod.build_guard_entries(["bad"]) == []

    observed: dict[str, object] = {}

    def _choose_snapshot_mode(**kwargs: object) -> str:
        observed.update(kwargs)
        return "off"

    plan = run_snapshot_contract_mod.build_snapshot_execution_plan(
        adapter=SimpleNamespace(),
        model=object(),
        cfg_snapshot={"temp_dir": ""},
        direct_reuse_loaded_model=False,
        skip_overhead_source=None,
        choose_snapshot_mode_fn=_choose_snapshot_mode,
        estimate_model_bytes_fn=lambda _model: 1024,
        psutil_module=None,
        environ={},
        tempfile_gettempdir_fn=lambda: str(tmp_path),
        disk_usage_fn=lambda _path: (_ for _ in ()).throw(TypeError("boom")),
        free_model_memory_fn=lambda _model: None,
    )
    assert observed["disk_free_mb"] == 0.0
    assert plan.snapshot_enabled is False

    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "runtime.manifest.json"
    manifest_payload = json.loads(
        Path("tests/fixtures/runtime_attestation/runtime.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_payload["report"] = "bad"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    monkeypatch.setattr(
        runtime_manifest_verify_mod.jsonschema,
        "validate",
        lambda **_kwargs: None,
    )
    errors = runtime_manifest_verify_mod.verify_report_manifest(
        report_path, manifest_path
    )
    assert "manifest is missing report.sha256" in errors

    monkeypatch.setenv("INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE", "true")
    flags = runner_context_mod.resolve_policy_flags(
        RunConfig(context={"eval": {"materialize_calibration": False}})
    )
    assert flags["allow_calibration_materialize"] is True

    timing = run_timing_policy_mod.build_timing_summary_payload(
        timings={"execute": 1.5},
        total_duration=None,
        report={"metrics": []},
    )
    assert timing is not None
    assert timing.memory_mb_peak is None


def test_finalize_related_types_and_decisions_cover_fallback_paths() -> None:
    invalid_diags = run_guard_overhead_mod._coerce_guard_overhead_diagnostics(None)  # noqa: SLF001
    assert invalid_diags == []

    class _BadPayload(dict):
        def get(self, key: object, default: object = None) -> object:  # type: ignore[override]
            if key == "overhead_ratio":
                raise TypeError("boom")
            return super().get(key, default)

    normalized = run_guard_overhead_mod.normalize_guard_overhead_result(_BadPayload())
    assert normalized["evaluated"] is False
    assert normalized["passed"] is True

    guard_result = core_types_mod.GuardValidationResult(passed=True, decision="allow")
    guard_result["diagnostics"] = "bad"
    assert guard_result.diagnostics == ()
    assert (
        core_types_mod.normalize_guard_decision(None, fallback_action="warn")
        == "monitor"
    )
    assert (
        core_types_mod.normalize_guard_decision(
            None, fallback_action="  ", passed=False
        )
        == "block"
    )


def test_finalize_phase_marks_non_finite_primary_metric_payload_invalid() -> None:
    runner = CoreRunner()
    cfg = RunConfig(max_pm_ratio=1.01, spike_threshold=1.05)

    status = runner._finalize_phase(
        object(),
        object(),
        {"spectral": {"passed": True}},
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": float("nan"),
                "final": 1.0,
            }
        },
        cfg,
        RunReport(),
    )
    assert status == RunStatus.ROLLBACK.value
