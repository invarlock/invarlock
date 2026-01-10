from __future__ import annotations

import json
from collections import UserDict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

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
        input_ids=[[1 + 4 * i, 2 + 4 * i, 3 + 4 * i, 4 + 4 * i] for i in range(preview_n)],
        attention_masks=[[1, 1, 1, 1] for _ in range(preview_n)],
        indices=list(range(preview_n)),
    )
    fin = EvaluationWindow(
        input_ids=[
            [101 + 4 * i, 102 + 4 * i, 103 + 4 * i, 104 + 4 * i]
            for i in range(final_n)
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
        edit_parameters: object | None = None,
        output: dict[str, object] | None = None,
    ) -> None:
        self.model = SimpleNamespace(id="gpt2", adapter="hf_gpt2", device="cpu")
        self.edit = SimpleNamespace(name="quant_rtn", plan=(edit_plan or {}))
        if edit_parameters is not None:
            self.edit.parameters = edit_parameters
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
                "parameters": getattr(self.edit, "parameters", None),
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


def _run_with_common_patches(*, cfg: _Cfg, exec_stub, post_stub, extra_patches=()):
    class Adapter:
        name = "hf_gpt2"

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
        patch("invarlock.cli.commands.run._prepare_config_for_run", lambda **k: cfg),
        patch("invarlock.cli.commands.run.detect_model_profile", _detect_profile),
        patch("invarlock.cli.commands.run.resolve_tokenizer", lambda *_a, **_k: _tok()),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch("invarlock.core.registry.get_registry", lambda: Registry()),
        patch(
            "invarlock.cli.commands.run._should_measure_overhead", lambda *_a: (False, False)
        ),
        patch("invarlock.cli.commands.run._execute_guarded_run", exec_stub),
        patch("invarlock.cli.commands.run._postprocess_and_summarize", post_stub),
        patch(
            "invarlock.cli.commands.run._resolve_metric_and_provider",
            lambda *_a, **_k: ("ppl_causal", None, {}),
        ),
        patch("invarlock.eval.primary_metric.compute_primary_metric_from_report", _pm_stub),
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
        )


def test_run_command_provider_dict_unwraps_nested_kwargs(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def resolver(*_a, **kwargs):  # noqa: ANN001
        captured["provider_kwargs"] = kwargs.get("provider_kwargs")
        return (
            SimpleNamespace(
                windows=lambda **kw: _provider_windows(
                    int(kw.get("preview_n", 0) or 0), int(kw.get("final_n", 0) or 0)
                )
            ),
            "validation",
            False,
        )

    def exec_stub(**kwargs):  # noqa: ANN001
        return _core_report(evaluation_windows={}), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider={
            "kind": "hf",
            "dataset_name": "wikitext",
            "cache_dir": "",
            "max_samples": None,
        },
    )

    _run_with_common_patches(
        cfg=cfg,
        exec_stub=exec_stub,
        post_stub=post_stub,
        extra_patches=(patch("invarlock.cli.commands.run._resolve_provider_and_split", resolver),),
    )

    provider_kwargs = captured["provider_kwargs"]
    assert isinstance(provider_kwargs, dict)
    assert provider_kwargs.get("dataset_name") == "wikitext"
    assert "cache_dir" not in provider_kwargs
    assert "max_samples" not in provider_kwargs


def test_run_command_provider_mapping_like_unwraps_data_and_items_fallback(
    tmp_path: Path,
) -> None:
    captured: list[dict[str, object]] = []

    def resolver(*_a, **kwargs):  # noqa: ANN001
        captured.append(dict(kwargs.get("provider_kwargs") or {}))
        return (
            SimpleNamespace(
                windows=lambda **kw: _provider_windows(
                    int(kw.get("preview_n", 0) or 0), int(kw.get("final_n", 0) or 0)
                )
            ),
            "validation",
            False,
        )

    def exec_stub(**kwargs):  # noqa: ANN001
        return _core_report(evaluation_windows={}), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    class ProviderObj:
        def __init__(self, data: dict[str, object], *, break_data: bool = False) -> None:
            self._data = None if break_data else data
            self._items = data

        def get(self, key: str, default=None):  # noqa: ANN001
            return self._items.get(key, default)

        def items(self):
            return self._items.items()

        def __bool__(self) -> bool:
            return True

    cfg_data = {"kind": "hf", "dataset_name": "wikitext", "cache_dir": ""}
    cfg = _Cfg(outdir=tmp_path / "runs", dataset_provider=ProviderObj(cfg_data))
    cfg2 = _Cfg(
        outdir=tmp_path / "runs2", dataset_provider=ProviderObj(cfg_data, break_data=True)
    )

    extra = (patch("invarlock.cli.commands.run._resolve_provider_and_split", resolver),)
    _run_with_common_patches(
        cfg=cfg, exec_stub=exec_stub, post_stub=post_stub, extra_patches=extra
    )
    _run_with_common_patches(
        cfg=cfg2, exec_stub=exec_stub, post_stub=post_stub, extra_patches=extra
    )

    assert captured and captured[0].get("dataset_name") == "wikitext"
    assert captured[1].get("dataset_name") == "wikitext"


def test_run_command_extracts_edit_config_from_plan_and_parameters(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    def resolver(*_a, **_k):
        return (
            SimpleNamespace(
                windows=lambda **kw: _provider_windows(
                    int(kw.get("preview_n", 0) or 0), int(kw.get("final_n", 0) or 0)
                )
            ),
            "validation",
            False,
        )

    def exec_stub(**kwargs):  # noqa: ANN001
        ec = kwargs.get("edit_config") or {}
        captured.append(dict(ec))
        return _core_report(evaluation_windows={}), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    # plan unwrap: non-dict mapping with .items and no _data
    plan = UserDict({"alpha": 1, "beta": 2})
    cfg_plan = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        edit_plan=plan,
    )

    # parameters unwrap: mapping with items()
    cfg_params_items = _Cfg(
        outdir=tmp_path / "runs2",
        dataset_provider="synthetic",
        edit_plan={},
        edit_parameters={"gamma": 3},
    )

    # parameters unwrap: dict subclass with hidden .items to hit isinstance(dict) branch
    cfg_params_dict = _Cfg(
        outdir=tmp_path / "runs3",
        dataset_provider="synthetic",
        edit_plan={},
        edit_parameters=_DictNoItems({"delta": 4}),
    )

    extra = (
        patch("invarlock.cli.commands.run._resolve_provider_and_split", resolver),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *_a, **_k: SimpleNamespace(
                windows=lambda **_kw: _provider_windows(1, 1)
            ),
        ),
    )
    _run_with_common_patches(
        cfg=cfg_plan, exec_stub=exec_stub, post_stub=post_stub, extra_patches=extra
    )
    _run_with_common_patches(
        cfg=cfg_params_items,
        exec_stub=exec_stub,
        post_stub=post_stub,
        extra_patches=extra,
    )
    _run_with_common_patches(
        cfg=cfg_params_dict,
        exec_stub=exec_stub,
        post_stub=post_stub,
        extra_patches=extra,
    )

    assert captured[0] == {"alpha": 1, "beta": 2}
    assert captured[1] == {"gamma": 3}
    assert captured[2] == {"delta": 4}


def test_run_command_baseline_token_counts_provider_parity_export_and_classification(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "provenance": {
                    "provider_digest": {"dataset": "synthetic", "split": "validation"}
                },
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0, 1],
                        "input_ids": [[1, 2], [3, 4]],
                        "attention_masks": [[1, 1], [1, 1]],
                        "logloss": [1.0, 1.0],
                        "token_counts": [2, 2],
                    },
                    "final": {
                        "window_ids": [2, 3],
                        "input_ids": [[5, 6], [7, 8]],
                        "attention_masks": [[1, 1], [1, 1]],
                        "logloss": [1.0, 1.0],
                        "token_counts": [2, 2],
                    },
                },
            }
        )
    )

    class Adapter:
        name = "hf_gpt2"

        def load_model(self, model_id: str, device: str | None = None):  # noqa: ARG002
            return object()

        def snapshot(self, model):  # noqa: ANN001
            return b"x"

        def restore(self, model, blob):  # noqa: ANN001,ARG002
            return None

        def save_pretrained(self, model, export_dir: Path):  # noqa: ANN001
            captured["save_pretrained_called"] = True
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "config.json").write_text("{}", encoding="utf-8")
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

    def exec_stub(**kwargs):  # noqa: ANN001
        captured["run_config"] = kwargs.get("run_config")
        captured["model_in_exec"] = kwargs.get("model")
        return _core_report(
            evaluation_windows={
                "preview": {"input_ids": [[1, 2], [3, 4]]},
                "final": {"input_ids": [[5, 6], [7, 8]]},
            }
        ), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        captured["report"] = kwargs.get("report")
        return {"json": str(tmp_path / "report.json")}

    def provider_digest(_report):  # noqa: ANN001
        return {"dataset": "synthetic", "split": "validation"}

    def enforce_parity(digest, base_digest, profile=None):  # noqa: ANN001,ARG001
        captured["base_digest"] = base_digest

    monkeypatch.setenv("INVARLOCK_EXPORT_MODEL", "1")
    monkeypatch.delenv("INVARLOCK_EXPORT_DIR", raising=False)
    monkeypatch.setenv("DEBUG_METRIC_DIFFS", "1")

    rec_console = Console(record=True)
    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        loss_type="classification",
        output={"model_dir": "exported_model"},
    )

    with ExitStack() as stack:
        for p in (
            patch("invarlock.cli.commands.run.console", rec_console),
            patch("invarlock.cli.commands.run._prepare_config_for_run", lambda **k: cfg),
            patch("invarlock.cli.commands.run.detect_model_profile", _detect_profile),
            patch("invarlock.cli.commands.run.resolve_tokenizer", lambda *_a, **_k: _tok()),
            patch("invarlock.cli.device.resolve_device", lambda d: d),
            patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
            patch(
                "invarlock.cli.commands.run._should_measure_overhead", lambda *_a: (False, False)
            ),
            patch("invarlock.cli.commands.run._execute_guarded_run", exec_stub),
            patch("invarlock.cli.commands.run._postprocess_and_summarize", post_stub),
            patch(
                "invarlock.cli.commands.run._resolve_metric_and_provider",
                lambda *_a, **_k: ("ppl_causal", None, {}),
            ),
            patch(
                "invarlock.eval.primary_metric.compute_primary_metric_from_report", _pm_stub
            ),
            patch("invarlock.core.registry.get_registry", lambda: Registry()),
            patch("invarlock.cli.commands.run._compute_provider_digest", provider_digest),
            patch("invarlock.cli.commands.run._enforce_provider_parity", enforce_parity),
            patch(
                "invarlock.cli.commands.run._format_debug_metric_diffs",
                lambda *_a, **_k: "diffs",
            ),
        ):
            stack.enter_context(p)
        run_command(
            config="dummy.yaml",
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            baseline=str(baseline),
            until_pass=False,
        )

    rc = captured["run_config"]
    assert rc is not None
    ctx = rc.context
    assert isinstance(ctx.get("baseline_eval_windows"), dict)
    assert ctx["baseline_eval_windows"]["final"]["token_counts"] == [2, 2]

    report = captured["report"]
    assert isinstance(report, dict)
    assert captured.get("base_digest") == {"dataset": "synthetic", "split": "validation"}
    assert captured.get("save_pretrained_called") is True
    assert report.get("artifacts", {}).get("checkpoint_path")
    assert report.get("metrics", {}).get("classification", {}).get("counts_source") == "measured"
    assert "DEBUG_METRIC_DIFFS" in rec_console.export_text()


def test_run_command_classification_pseudo_counts_and_export_env_dir(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class Adapter:
        name = "hf_gpt2"

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

    def exec_stub(**kwargs):  # noqa: ANN001
        return _core_report(evaluation_windows=None), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        captured["report"] = kwargs.get("report")
        return {"json": str(tmp_path / "report.json")}

    monkeypatch.setenv("INVARLOCK_EXPORT_MODEL", "1")
    monkeypatch.setenv("INVARLOCK_EXPORT_DIR", "env_export")
    monkeypatch.setenv("DEBUG_METRIC_DIFFS", "1")

    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        loss_type="classification",
        output={"save_model": True},
    )

    with ExitStack() as stack:
        for p in (
            patch("invarlock.cli.commands.run._prepare_config_for_run", lambda **k: cfg),
            patch("invarlock.cli.commands.run.detect_model_profile", _detect_profile),
            patch("invarlock.cli.commands.run.resolve_tokenizer", lambda *_a, **_k: _tok()),
            patch("invarlock.cli.device.resolve_device", lambda d: d),
            patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
            patch(
                "invarlock.cli.commands.run._should_measure_overhead", lambda *_a: (False, False)
            ),
            patch(
                "invarlock.cli.commands.run._resolve_provider_and_split",
                lambda *_a, **_k: (
                    SimpleNamespace(
                        windows=lambda **_kw: (
                            SimpleNamespace(input_ids=[], attention_masks=[], indices=[]),
                            SimpleNamespace(input_ids=[], attention_masks=[], indices=[]),
                        )
                    ),
                    "validation",
                    False,
                ),
            ),
            patch("invarlock.cli.commands.run._execute_guarded_run", exec_stub),
            patch("invarlock.cli.commands.run._postprocess_and_summarize", post_stub),
            patch(
                "invarlock.cli.commands.run._resolve_metric_and_provider",
                lambda *_a, **_k: ("ppl_causal", None, {}),
            ),
            patch(
                "invarlock.eval.primary_metric.compute_primary_metric_from_report", _pm_stub
            ),
            patch(
                "invarlock.cli.commands.run._format_debug_metric_diffs",
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


def test_run_command_until_pass_auto_tune_head_budget_paths(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2]],
                        "attention_masks": [[1, 1]],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[3, 4]],
                        "attention_masks": [[1, 1]],
                    },
                },
            }
        )
    )

    class Adapter:
        name = "hf_gpt2"

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

    class RC:
        def __init__(self, *args, **kwargs):  # noqa: ANN001,ARG002
            self.attempt_history = []

        def record_attempt(self, attempt, result, edit_config):  # noqa: ANN001
            self.attempt_history.append((attempt, result, edit_config))

        def should_retry(self, passed: bool) -> bool:  # noqa: ARG002
            return False

    def exec_stub(**kwargs):  # noqa: ANN001
        return _core_report(
            evaluation_windows={
                "preview": {"input_ids": [[1, 2]]},
                "final": {"input_ids": [[3, 4]]},
            }
        ), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        edit_plan={
            "heads": {
                "mask_only": True,
                "_auto_search": {"keep_low": 0, "keep_high": 8, "keep_current": 4},
            }
        },
    )

    def cert_fail_pm(_report, _baseline_report):  # noqa: ANN001
        return {"validation": {"primary_metric_acceptable": False, "drift_ok": True}}

    def cert_fail_other(_report, _baseline_report):  # noqa: ANN001
        return {"validation": {"primary_metric_acceptable": True, "drift_ok": False}}

    for make_cert in (cert_fail_pm, cert_fail_other):
        with ExitStack() as stack:
            for p in (
                patch("invarlock.cli.commands.run._prepare_config_for_run", lambda **k: cfg),
                patch("invarlock.cli.commands.run.detect_model_profile", _detect_profile),
                patch(
                    "invarlock.cli.commands.run.resolve_tokenizer", lambda *_a, **_k: _tok()
                ),
                patch("invarlock.cli.device.resolve_device", lambda d: d),
                patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
                patch("invarlock.core.registry.get_registry", lambda: Registry()),
                patch(
                    "invarlock.cli.commands.run._should_measure_overhead",
                    lambda *_a: (False, False),
                ),
                patch("invarlock.cli.commands.run._execute_guarded_run", exec_stub),
                patch("invarlock.cli.commands.run._postprocess_and_summarize", post_stub),
                patch("invarlock.core.retry.RetryController", RC),
                patch("invarlock.reporting.certificate.make_certificate", make_cert),
            ):
                stack.enter_context(p)
            run_command(
                config="dummy.yaml",
                device="cpu",
                profile=None,
                out=str(tmp_path / "runs"),
                baseline=str(baseline),
                until_pass=True,
                max_attempts=1,
            )
