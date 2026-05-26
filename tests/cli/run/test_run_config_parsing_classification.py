from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from invarlock.cli.commands.run import run_command
from invarlock.eval.data import EvaluationWindow


class _DictNoItems(dict):
    def __getattribute__(self, name: str):  # noqa: ANN001
        if name == "items":
            raise AttributeError(name)
        return super().__getattribute__(name)


class _TruthyEmptyDict(dict):
    def __bool__(self) -> bool:
        return True


def _detect_profile(model_id: str, adapter: str) -> SimpleNamespace:
    return SimpleNamespace(
        default_loss="ce",
        default_provider=None,
        default_metric=None,
        model_id=model_id,
        adapter=adapter,
        family="gpt",
        module_selectors={},
        invariants=[],
        cert_lints=[],
    )


def _tok():
    return (
        SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50_000),
        "tokhash123",
    )


def _pm_stub(*_a, **_k):
    return {
        "kind": "ppl_causal",
        "preview": 1.0,
        "final": 1.0,
        "ratio_vs_baseline": 1.0,
    }


def _core_report(*, evaluation_windows: dict[str, object] | None) -> SimpleNamespace:
    return SimpleNamespace(
        edit={"plan_digest": "abcd", "deltas": {"heads_pruned": 0}},
        metrics={
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "ppl_ratio": 1.0,
            "window_overlap_fraction": 0.0,
            "window_match_fraction": 1.0,
            "paired_windows": 1,
            "loss_type": "ce",
        },
        guards={},
        context={"dataset_meta": {}},
        evaluation_windows=evaluation_windows,
        status="success",
    )


def _provider_windows(
    preview_n: int, final_n: int
) -> tuple[EvaluationWindow, EvaluationWindow]:
    prev = EvaluationWindow(
        input_ids=[
            [1 + 4 * i, 2 + 4 * i, 3 + 4 * i, 4 + 4 * i] for i in range(preview_n)
        ],
        attention_masks=[[1, 1, 1, 1] for _ in range(preview_n)],
        indices=list(range(preview_n)),
    )
    fin = EvaluationWindow(
        input_ids=[
            [101 + 4 * i, 102 + 4 * i, 103 + 4 * i, 104 + 4 * i] for i in range(final_n)
        ],
        attention_masks=[[1, 1, 1, 1] for _ in range(final_n)],
        indices=[1000 + i for i in range(final_n)],
    )
    return prev, fin


class _Eval:
    def __init__(self, *, spike_threshold: float, loss_type: str, capacity_fast: bool):
        self.spike_threshold = float(spike_threshold)
        self.loss = SimpleNamespace(type=loss_type)
        self.capacity_fast = bool(capacity_fast)

    def model_dump(self) -> dict[str, object]:
        return {
            "spike_threshold": float(self.spike_threshold),
            "loss": {"type": str(getattr(self.loss, "type", "auto"))},
            "capacity_fast": bool(self.capacity_fast),
        }


class _Cfg:
    def __init__(
        self,
        *,
        outdir: Path,
        dataset_provider: object,
        loss_type: str = "ce",
        edit_plan: object | None = None,
        output: dict[str, object] | None = None,
    ) -> None:
        self.model = SimpleNamespace(id="gpt2", adapter="hf_causal", device="cpu")
        self.edit = SimpleNamespace(name="quant_rtn", plan=(edit_plan or {}))
        self.auto = SimpleNamespace(
            enabled=False, tier="balanced", probes=0, target_pm_ratio=None
        )
        self.guards = SimpleNamespace(order=[])
        self.dataset = SimpleNamespace(
            provider=dataset_provider,
            id="synthetic",
            split="validation",
            seq_len=8,
            stride=4,
            preview_n=2,
            final_n=2,
            seed=42,
        )
        self.eval = _Eval(spike_threshold=2.0, loss_type=loss_type, capacity_fast=True)
        out = {"dir": outdir}
        if output:
            out.update(output)
        self.output = SimpleNamespace(**out)

    def model_dump(self) -> dict[str, object]:
        out = {
            "dir": str(getattr(self.output, "dir", "")),
            "save_model": getattr(self.output, "save_model", False),
            "model_dir": getattr(self.output, "model_dir", None),
            "model_subdir": getattr(self.output, "model_subdir", None),
        }
        return {
            "model": {
                "id": self.model.id,
                "adapter": self.model.adapter,
                "device": self.model.device,
            },
            "edit": {
                "name": self.edit.name,
                "plan": getattr(self.edit, "plan", {}),
            },
            "auto": {
                "enabled": self.auto.enabled,
                "tier": self.auto.tier,
                "probes": self.auto.probes,
                "target_pm_ratio": self.auto.target_pm_ratio,
            },
            "guards": {"order": list(self.guards.order)},
            "dataset": {
                "provider": self.dataset.provider,
                "id": self.dataset.id,
                "split": self.dataset.split,
                "seq_len": self.dataset.seq_len,
                "stride": self.dataset.stride,
                "preview_n": self.dataset.preview_n,
                "final_n": self.dataset.final_n,
                "seed": self.dataset.seed,
            },
            "eval": {
                "spike_threshold": self.eval.spike_threshold,
                "loss": {"type": getattr(self.eval.loss, "type", None)},
            },
            "output": out,
        }


def _run_with_common_patches(
    *, cfg: _Cfg, exec_stub, post_stub, extra_patches=(), run_kwargs=None
):
    run_kwargs = run_kwargs or {}

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id: str, device: str | None = None):  # noqa: ARG002
            return object()

    adapter = Adapter()

    class Registry:
        def get_adapter(self, name):  # noqa: ARG002
            return adapter

        def get_edit(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, name, plugin_type):  # noqa: ARG002
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    patches = [
        patch("invarlock.cli.run_config.prepare_config_for_run", lambda **k: cfg),
        patch("invarlock.cli.run_runtime.detect_model_profile", _detect_profile),
        patch("invarlock.cli.run_runtime.resolve_tokenizer", lambda *_a, **_k: _tok()),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch("invarlock.core.registry.get_registry", lambda: Registry()),
        patch(
            "invarlock.core.run_orchestrator_execute._should_measure_overhead_impl",
            lambda *_a: (False, False, None),
        ),
        patch("invarlock.cli.run_runtime_exec.execute_guarded_run", exec_stub),
        patch("invarlock.cli.run_artifact_output.postprocess_and_summarize", post_stub),
        patch(
            "invarlock.cli.run_pairing.resolve_metric_and_provider",
            lambda *_a, **_k: ("ppl_causal", None, {}),
        ),
        patch(
            "invarlock.eval.primary_metric.compute_primary_metric_from_report", _pm_stub
        ),
    ]
    patches.extend(extra_patches)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        run_command(
            config="dummy.yaml",
            device="cpu",
            profile=None,
            out=str(cfg.output.dir),
            until_pass=False,
            **run_kwargs,
        )


def test_run_command_classification_pseudo_counts_and_export_env_dir(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id: str, device: str | None = None):  # noqa: ARG002
            return object()

        def save_pretrained(self, model, export_dir: Path):  # noqa: ANN001,ARG002
            return False

    adapter = Adapter()

    class Registry:
        def get_adapter(self, name):  # noqa: ARG002
            return adapter

        def get_edit(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, name, plugin_type):  # noqa: ARG002
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    def exec_stub(**kwargs):  # noqa: ANN001,ARG001
        return _core_report(evaluation_windows=None), object()

    def post_stub(**kwargs):  # noqa: ANN001
        captured["report"] = kwargs.get("report")
        return {"json": str(tmp_path / "report.json")}

    monkeypatch.setenv("INVARLOCK_EXPORT_MODEL", "1")
    monkeypatch.setenv("INVARLOCK_EXPORT_DIR", "env_export")
    monkeypatch.setenv("DEBUG_METRIC_DIFFS", "1")
    monkeypatch.setenv("INVARLOCK_ALLOW_PSEUDO_ACCURACY", "1")

    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        loss_type="classification",
        output={"save_model": True},
    )

    with ExitStack() as stack:
        for p in (
            patch("invarlock.cli.run_config.prepare_config_for_run", lambda **k: cfg),
            patch("invarlock.cli.run_runtime.detect_model_profile", _detect_profile),
            patch(
                "invarlock.cli.run_runtime.resolve_tokenizer", lambda *_a, **_k: _tok()
            ),
            patch("invarlock.cli.device.resolve_device", lambda d: d),
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            ),
            patch(
                "invarlock.core.run_orchestrator_execute._should_measure_overhead_impl",
                lambda *_a: (False, False, None),
            ),
            patch(
                "invarlock.cli.run_config.resolve_provider_and_split",
                lambda *_a, **_k: (
                    SimpleNamespace(
                        windows=lambda **_kw: (
                            SimpleNamespace(
                                input_ids=[], attention_masks=[], indices=[]
                            ),
                            SimpleNamespace(
                                input_ids=[], attention_masks=[], indices=[]
                            ),
                        )
                    ),
                    "validation",
                    False,
                ),
            ),
            patch("invarlock.cli.run_runtime_exec.execute_guarded_run", exec_stub),
            patch(
                "invarlock.cli.run_artifact_output.postprocess_and_summarize", post_stub
            ),
            patch(
                "invarlock.cli.run_pairing.resolve_metric_and_provider",
                lambda *_a, **_k: ("ppl_causal", None, {}),
            ),
            patch(
                "invarlock.eval.primary_metric.compute_primary_metric_from_report",
                _pm_stub,
            ),
            patch(
                "invarlock.reporting.run_metric_utils.format_debug_metric_diffs",
                lambda *_a, **_k: "",
            ),
            patch("invarlock.core.registry.get_registry", lambda: Registry()),
        ):
            stack.enter_context(p)
        run_command(
            config="dummy.yaml",
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    report = captured["report"]
    assert isinstance(report, dict)
    clf = report.get("metrics", {}).get("classification", {})
    assert clf.get("counts_source") == "pseudo_config"
    assert "metric_notes" in (report.get("provenance", {}) or {})


def test_run_command_export_saves_tokenizer_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class _Tokenizer:
        name_or_path = "Qwen/Qwen3-8B"
        eos_token = "</s>"
        pad_token = "</s>"
        vocab_size = 151_936

        def save_pretrained(self, output_dir: str) -> None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            captured["tokenizer_saved_to"] = str(path)

    tokenizer = _Tokenizer()

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id: str, device: str | None = None):  # noqa: ARG002
            return object()

        def save_pretrained(self, model, output_dir: Path):  # noqa: ANN001,ARG002
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text("{}", encoding="utf-8")
            captured["model_saved_to"] = str(output_dir)
            return True

    adapter = Adapter()

    class Registry:
        def get_adapter(self, name):  # noqa: ARG002
            return adapter

        def get_edit(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, name, plugin_type):  # noqa: ARG002
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    def exec_stub(**kwargs):  # noqa: ANN001,ARG001
        return _core_report(evaluation_windows=None), object()

    def post_stub(**kwargs):  # noqa: ANN001
        captured["report"] = kwargs.get("report")
        return {"json": str(tmp_path / "report.json")}

    monkeypatch.setenv("INVARLOCK_EXPORT_MODEL", "1")
    monkeypatch.delenv("INVARLOCK_EXPORT_DIR", raising=False)

    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        output={"model_dir": "exported_model"},
    )

    with ExitStack() as stack:
        for p in (
            patch("invarlock.cli.run_config.prepare_config_for_run", lambda **k: cfg),
            patch("invarlock.cli.run_runtime.detect_model_profile", _detect_profile),
            patch(
                "invarlock.cli.run_runtime.resolve_tokenizer",
                lambda *_a, **_k: (tokenizer, "tokhash123"),
            ),
            patch("invarlock.cli.device.resolve_device", lambda d: d),
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            ),
            patch(
                "invarlock.core.run_orchestrator_execute._should_measure_overhead_impl",
                lambda *_a: (False, False, None),
            ),
            patch(
                "invarlock.cli.run_config.resolve_provider_and_split",
                lambda *_a, **_k: (
                    SimpleNamespace(
                        windows=lambda **_kw: (
                            SimpleNamespace(
                                input_ids=[], attention_masks=[], indices=[]
                            ),
                            SimpleNamespace(
                                input_ids=[], attention_masks=[], indices=[]
                            ),
                        )
                    ),
                    "validation",
                    False,
                ),
            ),
            patch("invarlock.cli.run_runtime_exec.execute_guarded_run", exec_stub),
            patch(
                "invarlock.cli.run_artifact_output.postprocess_and_summarize", post_stub
            ),
            patch(
                "invarlock.cli.run_pairing.resolve_metric_and_provider",
                lambda *_a, **_k: ("ppl_causal", None, {}),
            ),
            patch(
                "invarlock.eval.primary_metric.compute_primary_metric_from_report",
                _pm_stub,
            ),
            patch("invarlock.core.registry.get_registry", lambda: Registry()),
        ):
            stack.enter_context(p)
        run_command(
            config="dummy.yaml",
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    report = captured["report"]
    assert isinstance(report, dict)
    checkpoint_path = report.get("artifacts", {}).get("checkpoint_path")
    assert isinstance(checkpoint_path, str)
    export_dir = Path(checkpoint_path)
    assert captured["model_saved_to"] == checkpoint_path
    assert captured["tokenizer_saved_to"] == checkpoint_path
    assert export_dir.name == "exported_model"
    assert (export_dir / "config.json").is_file()
    assert (export_dir / "tokenizer_config.json").is_file()
