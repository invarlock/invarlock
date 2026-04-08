from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from invarlock.core.run_orchestrator import (
    RunDiagnosticEvent,
    RunExecutionRequest,
    RunExecutionServices,
    RunFailureEvent,
    execute_run_request,
)
from tests.core._support_run_orchestrator_request import (
    _Config,
    _install_common_monkeypatches,
    _make_services,
)


def test_execute_run_request_reports_missing_baseline_windows(
    monkeypatch, tmp_path: Path
) -> None:
    config = _Config()
    _install_common_monkeypatches(monkeypatch)
    services = _make_services(
        tmp_path,
        config,
        provenance_result=SimpleNamespace(
            missing_evaluation_windows_for_baseline=True,
            missing_evaluation_windows_message="baseline windows missing",
        ),
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
            baseline=str(tmp_path / "baseline.json"),
        ),
        services=services,
    )

    failure_codes = [
        event.failure.code
        for event in outcome.events
        if isinstance(event, RunFailureEvent)
    ]
    assert outcome.ok is False
    assert outcome.failure is not None
    assert outcome.failure.code == "pipeline_failed"
    assert failure_codes == ["baseline_windows_missing", "pipeline_failed"]


def test_execute_run_request_halts_before_report_assembly_when_guarded_run_fails(
    monkeypatch, tmp_path: Path
) -> None:
    config = _Config()
    _install_common_monkeypatches(monkeypatch)
    persist_calls: list[object] = []
    assemble_calls: list[object] = []
    base_services = _make_services(tmp_path, config)
    services = RunExecutionServices(
        **{
            **base_services.__dict__,
            "execute_guarded_run": lambda **_kwargs: (
                SimpleNamespace(
                    edit={},
                    metrics={},
                    guards={},
                    context={},
                    evaluation_windows={},
                    status="failed",
                    error="[INVARLOCK:E321] RTN quantization matched no target modules.",
                ),
                object(),
            ),
            "assemble_run_report": lambda **_kwargs: (
                assemble_calls.append(object()),
                base_services.assemble_run_report(**_kwargs),
            )[1],
            "persist_run_report_outputs": lambda **_kwargs: (
                persist_calls.append(object()),
                base_services.persist_run_report_outputs(**_kwargs),
            )[1],
        }
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
        ),
        services=services,
    )

    failure_codes = [
        event.failure.code
        for event in outcome.events
        if isinstance(event, RunFailureEvent)
    ]
    assert outcome.ok is False
    assert outcome.result is None
    assert outcome.failure is not None
    assert (
        outcome.failure.summary
        == "[INVARLOCK:E321] RTN quantization matched no target modules."
    )
    assert failure_codes == ["pipeline_failed"]
    assert assemble_calls == []
    assert persist_calls == []


def test_execute_run_request_emits_export_and_guard_overhead_failure(
    monkeypatch, tmp_path: Path
) -> None:
    class _Adapter:
        name = "stub"

        def save_pretrained(self, _model, export_dir: Path) -> bool:
            export_dir.mkdir(parents=True, exist_ok=True)
            return True

    class _Tokenizer:
        def save_pretrained(self, _path: str) -> None:
            raise RuntimeError("missing tokenizer export support")

    config = _Config()
    config.output.save_model = True
    config.output.model_subdir = "hf-export"

    _install_common_monkeypatches(
        monkeypatch,
        adapter=_Adapter(),
        should_measure_overhead=(True, True, None),
    )
    services = _make_services(
        tmp_path,
        config,
        tokenizer=_Tokenizer(),
        report={
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 1.0,
                    "final": 1.2,
                    "ratio_vs_baseline": float("nan"),
                }
            },
            "guard_overhead": {
                "passed": False,
                "evaluated": True,
                "overhead_threshold": 0.02,
            },
            "artifacts": {},
        },
        metrics_enrichment=SimpleNamespace(
            pairing_violations=(),
            debug_diffs_line="debug metric diff",
        ),
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
            export_model_requested=True,
        ),
        services=services,
    )

    diagnostic_codes = {
        event.code for event in outcome.events if isinstance(event, RunDiagnosticEvent)
    }
    assert outcome.ok is False
    assert outcome.failure is not None
    assert outcome.failure.code == "guard_overhead_budget_exceeded"
    assert "export_tokenizer_missing" in diagnostic_codes
    assert "metric_diffs_debug" in diagnostic_codes


def test_execute_run_request_covers_diagnostic_and_seed_fallback_branches(
    monkeypatch, tmp_path: Path
) -> None:
    class _DatasetWithFallbacks:
        provider = "synthetic"
        preview_n = 1
        final_n = 1
        seq_len = 8
        stride = 4

        @property
        def seed(self) -> int:
            raise TypeError("seed unavailable")

        @property
        def split(self) -> str:
            raise TypeError("split unavailable")

    class _ConfigWithFallbacks(_Config):
        def __init__(self) -> None:
            super().__init__()
            self.dataset = _DatasetWithFallbacks()

        def model_dump(self) -> dict[str, object]:
            raise TypeError("model_dump disabled")

    class _Cudnn:
        benchmark = True
        deterministic = False

    class _Torch:
        backends = SimpleNamespace(cudnn=_Cudnn())

        def use_deterministic_algorithms(self, *_args, **_kwargs) -> None:
            raise RuntimeError("deterministic algorithms unavailable")

        def initial_seed(self) -> int:
            raise ValueError("torch seed unavailable")

    config = _ConfigWithFallbacks()
    _install_common_monkeypatches(monkeypatch)

    monkeypatch.setattr(
        "invarlock.core.run_orchestrator_execute._build_run_context_payload_impl",
        lambda **_kwargs: {"dataset": object(), "eval": {}},
    )
    monkeypatch.setattr(
        "invarlock.core.registry.get_registry",
        lambda: SimpleNamespace(
            get_adapter=lambda _name: SimpleNamespace(name="stub"),
            get_edit=lambda _name: SimpleNamespace(name="noop"),
            get_guard=lambda _name: (_ for _ in ()).throw(KeyError(_name)),
            get_plugin_metadata=lambda name, plugin_type: (
                (_ for _ in ()).throw(KeyError(name))
                if plugin_type == "edits"
                else {
                    "name": name,
                    "module": f"{plugin_type}.{name}",
                    "version": "test",
                }
            ),
        ),
    )
    monkeypatch.setattr(
        "invarlock.core.run_orchestrator_execute.np.random.get_state",
        lambda: (_ for _ in ()).throw(ValueError("no numpy state")),
    )
    monkeypatch.setattr(
        "invarlock.core.determinism_policy.apply_determinism_preset",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("preset unavailable")),
    )

    dataset_result = SimpleNamespace(
        diagnostics=(
            SimpleNamespace(
                kind="dataset_kind",
                metadata="not-a-dict",
                context={"ctx": 1},
                level="warning",
                summary="kind summary",
            ),
            SimpleNamespace(
                metadata={"meta": 1},
                details={"detail": 2},
                context={"ctx2": 3},
                level="info",
                summary="generic summary",
            ),
            SimpleNamespace(),
        ),
        resolved_split="validation",
        used_fallback_split=False,
        tokenizer=None,
        tokenizer_hash=None,
        calibration_data=[],
        dataset_meta={},
        window_plan=None,
        preview_count=1,
        final_count=1,
        effective_preview=1,
        effective_final=1,
        preview_mask_counts=[],
        final_mask_counts=[],
        preview_records=[],
        final_records=[],
    )
    base_services = _make_services(tmp_path, config)
    services = RunExecutionServices(
        **{
            **base_services.__dict__,
            "materialize_run_dataset": lambda **_kwargs: dataset_result,
            "load_baseline_pairing_evidence": lambda **_kwargs: SimpleNamespace(
                status="fallback",
                message=None,
                report_data=None,
                pairing_schedule=None,
                tokenizer_hash=None,
            ),
            "resolve_snapshot_config": lambda _context: (_ for _ in ()).throw(
                RuntimeError("snapshot config unavailable")
            ),
            "get_torch": lambda: _Torch(),
        }
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="ci",
            baseline=str(tmp_path / "baseline.json"),
            eval_device_override="cuda:1",
        ),
        services=services,
    )

    diagnostics = [
        event for event in outcome.events if isinstance(event, RunDiagnosticEvent)
    ]
    diagnostic_codes = {event.code for event in diagnostics}
    generic = next(
        event for event in diagnostics if event.code == "transition_diagnostic"
    )

    assert outcome.ok is True
    assert outcome.failure is None
    assert "dataset_kind" in diagnostic_codes
    assert generic.level == "info"
    assert generic.summary == "generic summary"
    assert generic.context["meta"] == 1
    assert generic.context["detail"] == 2
    assert generic.context["ctx2"] == 3


def test_execute_run_request_covers_retry_validation_and_timing_none(
    monkeypatch, tmp_path: Path
) -> None:
    class _ExplodingMetrics(dict):
        def get(self, key, default=None):
            if key == "primary_metric":
                raise TypeError("primary metric lookup failed")
            return super().get(key, default)

    config = _Config()
    _install_common_monkeypatches(monkeypatch)
    monkeypatch.setattr(
        "invarlock.core.run_orchestrator_execute._build_timing_summary_payload_impl",
        lambda **_kwargs: None,
    )

    base_services = _make_services(
        tmp_path,
        config,
        report={
            "metrics": _ExplodingMetrics(),
            "guard_overhead": {"passed": False, "evaluated": False},
            "artifacts": {},
        },
    )
    services = RunExecutionServices(
        **{
            **base_services.__dict__,
            "init_retry_controller": lambda **_kwargs: object(),
            "validate_retry_evaluation_report": lambda **_kwargs: SimpleNamespace(
                telemetry_summary="retry telemetry"
            ),
        }
    )

    monkeypatch.setattr(
        "invarlock.core.run_orchestrator_execute._resolve_retry_validation_transition_impl",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="unexpected",
            validation_gates=("gate-a",),
            error=SimpleNamespace(message="unexpected retry state"),
            updated_edit_config={},
            head_adjustment=None,
            diagnostics=(
                SimpleNamespace(
                    kind="retry_kind",
                    metadata={"from": "retry"},
                    level="warning",
                    summary="retry diagnostic",
                ),
            ),
            next_attempt=None,
        ),
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
            baseline=str(tmp_path / "baseline.json"),
            until_pass=True,
            capture_timings=True,
        ),
        services=services,
    )

    diagnostics = [
        event for event in outcome.events if isinstance(event, RunDiagnosticEvent)
    ]
    summaries = {event.code: event.summary for event in diagnostics}

    assert outcome.ok is True
    assert outcome.failure is None
    assert summaries["retry_validation_telemetry_summary"] == "retry telemetry"
    assert outcome.result is not None
    assert outcome.result.timing_summary is None


def test_execute_run_request_covers_export_fallback_branches(
    monkeypatch, tmp_path: Path
) -> None:
    class _OutputConfig:
        dir = "runs"

        @property
        def save_model(self) -> bool:
            raise TypeError("save_model unavailable")

        @property
        def model_dir(self) -> str:
            raise RuntimeError("model_dir unavailable")

        @property
        def model_path(self) -> str | None:
            return None

        @property
        def model_subdir(self) -> str:
            raise RuntimeError("model_subdir unavailable")

    config = _Config()
    config.output = _OutputConfig()

    _install_common_monkeypatches(monkeypatch, adapter=SimpleNamespace(name="stub"))
    services = _make_services(
        tmp_path,
        config,
        report={
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 1.0,
                    "final": 1.0,
                }
            },
            "artifacts": {},
        },
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
            export_model_requested=True,
            export_dir="   ",
        ),
        services=services,
    )

    diagnostic_codes = {
        event.code for event in outcome.events if isinstance(event, RunDiagnosticEvent)
    }

    assert outcome.ok is True
    assert outcome.failure is None
    assert "export_adapter_directory_missing" in diagnostic_codes


def test_execute_run_request_fails_fast_when_torch_is_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    config = _Config()
    _install_common_monkeypatches(monkeypatch)
    base_services = _make_services(tmp_path, config)
    services = RunExecutionServices(
        **{
            **base_services.__dict__,
            "get_torch": lambda: None,
        }
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
        ),
        services=services,
    )

    assert outcome.ok is False
    assert outcome.failure is not None
    assert outcome.failure.code == "torch_missing"


def test_execute_run_request_maps_torch_import_errors_to_typed_failure(
    monkeypatch, tmp_path: Path
) -> None:
    config = _Config()
    _install_common_monkeypatches(monkeypatch)
    base_services = _make_services(tmp_path, config)
    services = RunExecutionServices(
        **{
            **base_services.__dict__,
            "load_model_with_cfg": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ImportError("torch backend unavailable")
            ),
        }
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
        ),
        services=services,
    )

    failure_codes = [
        event.failure.code
        for event in outcome.events
        if isinstance(event, RunFailureEvent)
    ]
    assert outcome.ok is False
    assert outcome.failure is not None
    assert outcome.failure.code == "torch_missing"
    assert failure_codes == ["torch_missing"]


def test_execute_run_request_applies_preset_seed_overrides_and_model_path_attr(
    monkeypatch, tmp_path: Path
) -> None:
    class _Adapter:
        name = "stub"

        def __init__(self) -> None:
            self.saved: list[Path] = []

        def save_pretrained(self, _model, export_dir: Path) -> bool:
            self.saved.append(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            return True

    class _OutputConfig:
        dir = "runs"
        save_model = True
        model_dir = None
        model_path = "hf-export"
        model_subdir = "ignored"

    config = _Config()
    config.output = _OutputConfig()
    adapter = _Adapter()
    captured_seed_bundles: list[dict[str, object]] = []
    _install_common_monkeypatches(monkeypatch, adapter=adapter)
    monkeypatch.setattr(
        "invarlock.core.determinism_policy.apply_determinism_preset",
        lambda **_kwargs: {
            "mode": "strict",
            "seeds": {"python": 101, "numpy": 202, "torch": 303},
        },
    )
    base_services = _make_services(tmp_path, config)
    services = RunExecutionServices(
        **{
            **base_services.__dict__,
            "assemble_run_report": lambda **kwargs: (
                captured_seed_bundles.append(dict(kwargs["seed_bundle"])),
                SimpleNamespace(
                    report={
                        "metrics": {
                            "primary_metric": {
                                "kind": "ppl_causal",
                                "preview": 1.0,
                                "final": 1.0,
                            }
                        },
                        "artifacts": {},
                    },
                    timings={},
                    provenance_result=SimpleNamespace(
                        missing_evaluation_windows_for_baseline=False
                    ),
                    metrics_enrichment=SimpleNamespace(
                        pairing_violations=(),
                        debug_diffs_line=None,
                    ),
                ),
            )[1],
        }
    )

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="ci",
            export_model_requested=True,
        ),
        services=services,
    )

    assert outcome.ok is True
    assert outcome.failure is None
    assert captured_seed_bundles == [{"python": 101, "numpy": 202, "torch": 303}]


def test_execute_run_request_marks_export_failure_when_adapter_save_raises(
    monkeypatch, tmp_path: Path
) -> None:
    class _Adapter:
        name = "stub"

        def save_pretrained(self, _model, _export_dir: Path) -> bool:
            raise RuntimeError("export boom")

    config = _Config()
    config.output.save_model = True

    _install_common_monkeypatches(monkeypatch, adapter=_Adapter())
    services = _make_services(tmp_path, config)

    outcome = execute_run_request(
        RunExecutionRequest(
            config=str(tmp_path / "config.yaml"),
            device="cpu",
            profile="dev",
            export_model_requested=True,
        ),
        services=services,
    )

    diagnostic_codes = {
        event.code for event in outcome.events if isinstance(event, RunDiagnosticEvent)
    }
    assert outcome.ok is True
    assert "export_failed" in diagnostic_codes
